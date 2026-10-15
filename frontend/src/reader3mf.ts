import { unzipSync, strFromU8 } from 'fflate';

/**
 * Minimal 3MF reader that handles what slicers write and three's loader
 * skips: Bambu/Orca split files (components with p:path), component and
 * build-item transforms (scale!), and per-triangle paint codes.
 * Regex scanning instead of DOMParser keeps 80 MB model files cheap.
 */

type Mat = number[]; // 12 values, 3MF row-vector order: m00 m01 m02 m10 m11 m12 m20 m21 m22 m30 m31 m32
const IDENTITY: Mat = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0];

function parseMat(s: string | undefined): Mat {
  if (!s) return IDENTITY;
  const v = s.trim().split(/\s+/).map(Number);
  return v.length === 12 && v.every(Number.isFinite) ? v : IDENTITY;
}

/** a then b (apply a first). */
function mul(a: Mat, b: Mat): Mat {
  const r = new Array(12);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 3; j++) {
      r[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j] + (i === 3 ? b[9 + j] : 0);
    }
  }
  return r;
}

const attr = (tag: string, name: string) => {
  const m = tag.match(new RegExp(`(?:^|\\s)${name}="([^"]*)"`));
  return m ? m[1] : undefined;
};

interface ObjectDef {
  positions?: Float32Array;
  faces?: Uint32Array;
  paint?: (string | undefined)[];
  components?: { path?: string; objectid: string; transform: Mat }[];
}

function parseModel(xml: string, path: string, objects: Map<string, ObjectDef>) {
  const objRe = /<object\b([^>]*)>([\s\S]*?)<\/object>/g;
  let m: RegExpExecArray | null;
  while ((m = objRe.exec(xml))) {
    const id = attr(m[1], 'id')!;
    const body = m[2];
    const def: ObjectDef = {};
    if (body.includes('<mesh')) {
      const pos: number[] = [];
      const vRe = /<vertex\b([^>]*)\/?>/g;
      let v: RegExpExecArray | null;
      while ((v = vRe.exec(body))) pos.push(+attr(v[1], 'x')!, +attr(v[1], 'y')!, +attr(v[1], 'z')!);
      const faces: number[] = [];
      const paint: (string | undefined)[] = [];
      let anyPaint = false;
      const tRe = /<triangle\b([^>]*)\/?>/g;
      let t: RegExpExecArray | null;
      while ((t = tRe.exec(body))) {
        faces.push(+attr(t[1], 'v1')!, +attr(t[1], 'v2')!, +attr(t[1], 'v3')!);
        const p = attr(t[1], 'paint_color') ?? attr(t[1], 'slic3rpe:mmu_segmentation');
        if (p) anyPaint = true;
        paint.push(p);
      }
      def.positions = new Float32Array(pos);
      def.faces = new Uint32Array(faces);
      if (anyPaint) def.paint = paint;
    }
    const cRe = /<component\b([^>]*)\/?>/g;
    let c: RegExpExecArray | null;
    while ((c = cRe.exec(body))) {
      (def.components ??= []).push({
        path: attr(c[1], 'p:path'),
        objectid: attr(c[1], 'objectid')!,
        transform: parseMat(attr(c[1], 'transform')),
      });
    }
    objects.set(`${path}#${id}`, def);
  }
}

/**
 * Extruder from an unsplit-triangle paint code (see encodePaint).
 * Codes with split sub-triangles take their first leaf's state.
 */
export function decodePaint(code: string | undefined): number {
  if (!code) return 0;
  // Bitstream is the hex string read back to front, 4 bits per char.
  const nibbles = Array.from(code).reverse().map((ch) => parseInt(ch, 16));
  let i = 0;
  const readLeaf = (): number => {
    const n = nibbles[i++] ?? 0;
    const split = n & 3;
    if (split) {
      // Skip into the first child (children are serialised in order).
      return readLeaf();
    }
    const st = n >> 2;
    if (st < 3) return st;
    let state = 3;
    for (;;) {
      const x = nibbles[i++] ?? 0;
      state += x;
      if (x !== 15) break;
    }
    return state;
  };
  return readLeaf();
}

export interface Read3mf {
  positions: Float32Array;
  faces: Uint32Array;
  /** 1-based extruder per face (0 = unpainted), when the file had paint. */
  extruders?: Uint8Array;
}

export function read3mf(buf: ArrayBuffer): Read3mf {
  const files = unzipSync(new Uint8Array(buf));
  const norm = (p: string) => p.replace(/^\//, '');
  const objects = new Map<string, ObjectDef>();
  for (const [name, data] of Object.entries(files)) {
    if (name.toLowerCase().endsWith('.model')) parseModel(strFromU8(data), '/' + norm(name), objects);
  }
  const rootName = Object.keys(files).find((n) => /^3D\/3dmodel\.model$/i.test(n)) ?? Object.keys(files).find((n) => n.endsWith('.model'));
  if (!rootName) throw new Error('No 3D model in this 3MF');
  const root = '/' + rootName;
  const rootXml = strFromU8(files[rootName]);

  const pos: number[] = [];
  const faces: number[] = [];
  const ext: number[] = [];
  let painted = false;

  const emit = (key: string, m: Mat, depth: number) => {
    const def = objects.get(key);
    if (!def || depth > 16) return;
    if (def.positions && def.faces) {
      const base = pos.length / 3;
      const p = def.positions;
      for (let i = 0; i < p.length; i += 3) {
        const x = p[i], y = p[i + 1], z = p[i + 2];
        pos.push(x * m[0] + y * m[3] + z * m[6] + m[9], x * m[1] + y * m[4] + z * m[7] + m[10], x * m[2] + y * m[5] + z * m[8] + m[11]);
      }
      // Mirroring transforms flip winding; restore outward normals.
      const det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
      for (let i = 0; i < def.faces.length; i += 3) {
        if (det < 0) faces.push(base + def.faces[i], base + def.faces[i + 2], base + def.faces[i + 1]);
        else faces.push(base + def.faces[i], base + def.faces[i + 1], base + def.faces[i + 2]);
        const code = def.paint?.[i / 3];
        if (code) painted = true;
        ext.push(decodePaint(code));
      }
    }
    for (const c of def.components ?? []) {
      const path = c.path ? '/' + norm(c.path) : key.split('#')[0];
      emit(`${path}#${c.objectid}`, mul(c.transform, m), depth + 1);
    }
  };

  const itemRe = /<item\b([^>]*)\/?>/g;
  let it: RegExpExecArray | null;
  let items = 0;
  while ((it = itemRe.exec(rootXml))) {
    if (attr(it[1], 'printable') === '0') continue;
    emit(`${root}#${attr(it[1], 'objectid')}`, parseMat(attr(it[1], 'transform')), 0);
    items++;
  }
  if (!items) throw new Error('3MF has no build items');
  if (!faces.length) throw new Error('3MF contains no triangles');
  return {
    positions: new Float32Array(pos),
    faces: new Uint32Array(faces),
    extruders: painted ? new Uint8Array(ext) : undefined,
  };
}
