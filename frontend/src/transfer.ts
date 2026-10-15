import type { PaintMesh } from './mesh';

/**
 * Copy paint between repeated parts. Articulated prints are many near-copies
 * of one segment (the tail ones shrinking), each its own separate shell. To
 * repeat a paint job, each shell is put in a canonical frame (centroid +
 * principal axes + per-axis spread), the best of the 8 axis-sign choices is
 * picked by fit error, and every target face takes the slot of the nearest
 * source face in that frame.
 */

export interface Shell {
  faces: Uint32Array;
  area: number;
  center: [number, number, number];
  axes: number[][]; // 3 unit vectors, largest spread first
  spread: [number, number, number]; // sqrt eigenvalues
}

const cache = new WeakMap<PaintMesh, { shellOf: Int32Array; shells: Shell[] }>();

function symEigen(m: number[][]): { values: number[]; vectors: number[][] } {
  // Jacobi rotations on a 3x3 symmetric matrix.
  const a = m.map((r) => r.slice());
  const v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 30; sweep++) {
    let off = 0;
    for (let p = 0; p < 2; p++) for (let q = p + 1; q < 3; q++) off += a[p][q] * a[p][q];
    if (off < 1e-20) break;
    for (let p = 0; p < 2; p++) {
      for (let q = p + 1; q < 3; q++) {
        if (Math.abs(a[p][q]) < 1e-30) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < 3; k++) {
          const akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < 3; k++) {
          const apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < 3; k++) {
          const vkp = v[k][p], vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq; v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const order = [0, 1, 2].sort((i, j) => a[j][j] - a[i][i]);
  return { values: order.map((i) => a[i][i]), vectors: order.map((i) => [v[0][i], v[1][i], v[2][i]]) };
}

export function shells(mesh: PaintMesh): { shellOf: Int32Array; shells: Shell[] } {
  const hit = cache.get(mesh);
  if (hit) return hit;
  const F = mesh.faceCount, adj = mesh.adjacency, c = mesh.centroids;
  const shellOf = new Int32Array(F).fill(-1);
  const list: Shell[] = [];
  for (let s = 0; s < F; s++) {
    if (shellOf[s] >= 0) continue;
    const id = list.length;
    const faces: number[] = [];
    const stack = [s];
    shellOf[s] = id;
    while (stack.length) {
      const f = stack.pop()!;
      faces.push(f);
      for (let k = 0; k < 3; k++) {
        const g = adj[f * 3 + k];
        if (g >= 0 && shellOf[g] < 0) { shellOf[g] = id; stack.push(g); }
      }
    }
    let A = 0, mx = 0, my = 0, mz = 0;
    for (const f of faces) { const w = mesh.areas[f]; A += w; mx += w * c[f * 3]; my += w * c[f * 3 + 1]; mz += w * c[f * 3 + 2]; }
    mx /= A; my /= A; mz /= A;
    const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (const f of faces) {
      const w = mesh.areas[f] / A;
      const d = [c[f * 3] - mx, c[f * 3 + 1] - my, c[f * 3 + 2] - mz];
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[i][j] += w * d[i] * d[j];
    }
    const e = symEigen(C);
    list.push({
      faces: Uint32Array.from(faces),
      area: A,
      center: [mx, my, mz],
      axes: e.vectors,
      spread: e.values.map((x) => Math.sqrt(Math.max(x, 1e-12))) as [number, number, number],
    });
  }
  const out = { shellOf, shells: list };
  cache.set(mesh, out);
  return out;
}

/** Canonical coordinates of a face centroid in a shell's frame, with axis signs. */
function canon(mesh: PaintMesh, sh: Shell, f: number, signs: number[], out: number[]) {
  const c = mesh.centroids;
  const d0 = c[f * 3] - sh.center[0], d1 = c[f * 3 + 1] - sh.center[1], d2 = c[f * 3 + 2] - sh.center[2];
  for (let k = 0; k < 3; k++) {
    const ax = sh.axes[k];
    out[k] = (signs[k] * (d0 * ax[0] + d1 * ax[1] + d2 * ax[2])) / sh.spread[k];
  }
}

/** Nearest-neighbour lookup over points in canonical space (uniform grid, numeric keys). */
class Grid {
  private cells = new Map<number, number[]>();
  private pts: Float32Array;
  private h: number;
  constructor(pts: Float32Array, h: number) {
    this.pts = pts; this.h = h;
    for (let i = 0; i < pts.length / 3; i++) {
      const k = this.key(Math.floor(pts[i * 3] / h), Math.floor(pts[i * 3 + 1] / h), Math.floor(pts[i * 3 + 2] / h));
      let cell = this.cells.get(k);
      if (!cell) { cell = []; this.cells.set(k, cell); }
      cell.push(i);
    }
  }
  private key(i: number, j: number, k: number) {
    return ((i + 512) * 1024 + (j + 512)) * 1024 + (k + 512);
  }
  nearest(x: number, y: number, z: number): { i: number; d: number } {
    const cx = Math.floor(x / this.h), cy = Math.floor(y / this.h), cz = Math.floor(z / this.h);
    let best = -1, bd = Infinity;
    for (let r = 1; r <= 4 && best < 0; r++) {
      for (let i = -r; i <= r; i++) for (let j = -r; j <= r; j++) for (let k = -r; k <= r; k++) {
        // Shells already searched at a smaller radius are skipped.
        if (r > 1 && Math.max(Math.abs(i), Math.abs(j), Math.abs(k)) < r) continue;
        const cell = this.cells.get(this.key(cx + i, cy + j, cz + k));
        if (!cell) continue;
        for (const p of cell) {
          const dx = this.pts[p * 3] - x, dy = this.pts[p * 3 + 1] - y, dz = this.pts[p * 3 + 2] - z;
          const d = dx * dx + dy * dy + dz * dz;
          if (d < bd) { bd = d; best = p; }
        }
      }
    }
    return { i: best, d: Math.sqrt(bd) };
  }
}

const SIGNS = [-1, 1].flatMap((a) => [-1, 1].flatMap((b) => [-1, 1].map((c) => [a, b, c])));

export interface TransferReport {
  source: number;
  copied: number[]; // target shell ids
  skipped: { shell: number; reason: string }[];
  faces: Map<number, number>; // face -> old slot, for undo
}

/** Shells that look like the source: similar area and spread ratios. */
export function matchingShells(mesh: PaintMesh, source: number): number[] {
  const { shells: list } = shells(mesh);
  const s = list[source];
  const r1 = s.spread[1] / s.spread[0], r2 = s.spread[2] / s.spread[0];
  return list
    .map((t, i) => ({ t, i }))
    .filter(({ t, i }) => {
      if (i === source) return false;
      const ar = t.area / s.area;
      if (ar < 0.35 || ar > 2.8) return false;
      const t1 = t.spread[1] / t.spread[0], t2 = t.spread[2] / t.spread[0];
      return Math.abs(Math.log(t1 / r1)) < 0.35 && Math.abs(Math.log(t2 / r2)) < 0.35;
    })
    .map(({ i }) => i);
}

/**
 * Copy the source shell's paint onto each matching shell. Shells whose best
 * alignment still fits poorly are skipped rather than painted wrongly.
 */
export function copyPaint(mesh: PaintMesh, source: number, targets?: number[]): TransferReport {
  const { shells: list } = shells(mesh);
  const src = list[source];
  const tgts = targets ?? matchingShells(mesh, source);
  const report: TransferReport = { source, copied: [], skipped: [], faces: new Map() };

  const srcPts = new Float32Array(src.faces.length * 3);
  const tmp = [0, 0, 0];
  for (let i = 0; i < src.faces.length; i++) {
    canon(mesh, src, src.faces[i], [1, 1, 1], tmp);
    srcPts[i * 3] = tmp[0]; srcPts[i * 3 + 1] = tmp[1]; srcPts[i * 3 + 2] = tmp[2];
  }
  const grid = new Grid(srcPts, 0.06);

  for (const t of tgts) {
    const tgt = list[t];
    // Choose the axis signs that fit best on a sample of target faces.
    const step = Math.max(1, Math.floor(tgt.faces.length / 600));
    let bestSigns = SIGNS[0], bestErr = Infinity;
    for (const signs of SIGNS) {
      let err = 0, n = 0;
      for (let i = 0; i < tgt.faces.length; i += step) {
        canon(mesh, tgt, tgt.faces[i], signs, tmp);
        err += grid.nearest(tmp[0], tmp[1], tmp[2]).d; n++;
      }
      err /= n;
      if (err < bestErr) { bestErr = err; bestSigns = signs; }
    }
    if (bestErr > 0.12) {
      report.skipped.push({ shell: t, reason: `shape differs too much (fit error ${bestErr.toFixed(2)})` });
      continue;
    }
    for (const f of tgt.faces) {
      canon(mesh, tgt, f, bestSigns, tmp);
      const nn = grid.nearest(tmp[0], tmp[1], tmp[2]);
      if (nn.i < 0) continue;
      const slot = mesh.slots[src.faces[nn.i]];
      if (mesh.slots[f] !== slot) {
        if (!report.faces.has(f)) report.faces.set(f, mesh.slots[f]);
        mesh.slots[f] = slot;
      }
    }
    report.copied.push(t);
  }
  return report;
}

/**
 * The source part plus every part that really has its shape (passes the
 * same alignment check as copyPaint), e.g. all the body segments but not
 * the legs that happen to be a similar size.
 */
export function similarShells(mesh: PaintMesh, source: number, maxErr = 0.12): number[] {
  const { shells: list } = shells(mesh);
  const src = list[source];
  const tmp = [0, 0, 0];
  const srcPts = new Float32Array(src.faces.length * 3);
  for (let i = 0; i < src.faces.length; i++) {
    canon(mesh, src, src.faces[i], [1, 1, 1], tmp);
    srcPts[i * 3] = tmp[0]; srcPts[i * 3 + 1] = tmp[1]; srcPts[i * 3 + 2] = tmp[2];
  }
  const grid = new Grid(srcPts, 0.06);
  const out = [source];
  for (const t of matchingShells(mesh, source)) {
    const tgt = list[t];
    const step = Math.max(1, Math.floor(tgt.faces.length / 600));
    let best = Infinity;
    for (const signs of SIGNS) {
      let err = 0, n = 0;
      for (let i = 0; i < tgt.faces.length; i += step) {
        canon(mesh, tgt, tgt.faces[i], signs, tmp);
        err += grid.nearest(tmp[0], tmp[1], tmp[2]).d; n++;
      }
      best = Math.min(best, err / n);
    }
    if (best <= maxErr) out.push(t);
  }
  return out;
}

/**
 * Spikes of a segmented body by position: in each part, faces above
 * `heightFrac` of the part's own height, or further sideways than
 * `reachFrac` of its own half-width, plus optionally a spine stripe of
 * `stripeFrac` of the half-width down the middle of the top. Scaled per part, so tapering tail
 * segments get the same cut as big ones.
 */
export function segmentRuleFaces(mesh: PaintMesh, parts: number[], heightFrac: number, reachFrac: number, stripeFrac = 0): number[] {
  const { shells: list } = shells(mesh);
  const c = mesh.centroids;
  const out: number[] = [];
  for (const i of parts) {
    const sh = list[i];
    // Sideways = the part's longest axis, flattened to horizontal (spike to spike).
    let lx = sh.axes[0][0], ly = sh.axes[0][1];
    const l = Math.hypot(lx, ly) || 1;
    lx /= l; ly /= l;
    let zlo = Infinity, zhi = -Infinity, smax = 0;
    for (const f of sh.faces) {
      const z = c[f * 3 + 2];
      zlo = Math.min(zlo, z); zhi = Math.max(zhi, z);
      smax = Math.max(smax, Math.abs((c[f * 3] - sh.center[0]) * lx + (c[f * 3 + 1] - sh.center[1]) * ly));
    }
    const zspan = zhi - zlo || 1;
    for (const f of sh.faces) {
      const zr = (c[f * 3 + 2] - zlo) / zspan;
      const sr = Math.abs((c[f * 3] - sh.center[0]) * lx + (c[f * 3 + 1] - sh.center[1]) * ly) / (smax || 1);
      // Spine stripe: a band down the middle of the segment's upward-facing top.
      const stripe = stripeFrac > 0 && sr < stripeFrac && zr > 0.4 && mesh.faceNormals[f * 3 + 2] > 0.25;
      if (zr > heightFrac || sr > reachFrac || stripe) out.push(f);
    }
  }
  return out;
}
