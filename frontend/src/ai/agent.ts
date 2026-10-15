import * as THREE from 'three';
import type { PaintMesh } from '../mesh';
import type { History } from '../history';
import type { Filament } from './client';
import { brushFaces, creaseFill, rangeFaces, smoothEdges } from '../tools';
import { similarFaces, similarThinFaces } from './similar';
import { computeThickness, thinFill } from '../thickness';
import { copyPaint, segmentRuleFaces, shells, similarShells } from '../transfer';

/**
 * Browser half of the hand-painting agent: renders views for the model,
 * turns its pixel clicks and outlines into surface selections, and runs the
 * same tools a person would use in the slicer. One instance is one chat
 * session; follow-up requests continue the same conversation.
 */

const SIZE = 900;

export interface Shot {
  id: string;
  caption: string;
  image: string;
}

interface ToolCall {
  id: string;
  name: string;
  input: Record<string, unknown>;
}

interface ToolResult {
  id: string;
  text: string;
  error?: boolean;
  images?: Shot[];
}

interface Turn {
  session?: string;
  calls: ToolCall[];
  text: string;
  finished: boolean;
  cost?: number;
}

export interface AgentEvents {
  onStatus(msg: string): void;
  onImage(shot: Shot): void;
  /** The model's own words to the user. */
  onSay(text: string): void;
  /** One line per tool call. */
  onAction(line: string): void;
}

interface Pt { x: number; y: number }

async function post<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) {
    let msg = `${r.status}`;
    try { msg = (await r.json()).detail ?? msg; } catch { /* keep status */ }
    throw new Error(msg);
  }
  return r.json();
}

const ID_VERT = /* glsl */ `
attribute float fid;
varying float vFid;
void main() {
  vFid = fid;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;
const ID_FRAG = /* glsl */ `
varying float vFid;
void main() {
  float id = floor(vFid + 0.5) + 1.0;
  gl_FragColor = vec4(mod(id, 256.0) / 255.0, mod(floor(id / 256.0), 256.0) / 255.0, floor(id / 65536.0) / 255.0, 1.0);
}`;

export class PaintAgent {
  private cams = new Map<string, THREE.OrthographicCamera>();
  private shotNo = 0;
  private scene = new THREE.Scene();
  private obj: THREE.Mesh<THREE.BufferGeometry, THREE.Material>;
  private paintMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.55 });
  private idGeo: THREE.BufferGeometry | null = null;
  private idMat = new THREE.ShaderMaterial({ vertexShader: ID_VERT, fragmentShader: ID_FRAG });
  private target: THREE.WebGLRenderTarget;
  private idTarget: THREE.WebGLRenderTarget;
  private raycaster = new THREE.Raycaster();
  private undoable = 0;
  private session: string | null = null;
  /** Results owed for the last turn's tool calls, sent with the next message. */
  private pending: ToolResult[] = [];
  private renderer: THREE.WebGLRenderer;
  private mesh: PaintMesh;
  private history: History;
  private filaments: Filament[];
  private events: AgentEvents;
  stopped = false;
  /** Running API spend for this chat, from the usage each step reports. */
  cost = 0;

  constructor(renderer: THREE.WebGLRenderer, mesh: PaintMesh, history: History, filaments: Filament[], events: AgentEvents) {
    this.renderer = renderer;
    this.mesh = mesh;
    this.history = history;
    this.filaments = filaments;
    this.events = events;
    this.obj = new THREE.Mesh<THREE.BufferGeometry, THREE.Material>(mesh.geometry, this.paintMat);
    this.scene.add(this.obj);
    this.scene.background = new THREE.Color(0xd9dde3);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8f99, 2.2));
    this.target = new THREE.WebGLRenderTarget(SIZE, SIZE, { samples: 4 });
    this.target.texture.colorSpace = THREE.SRGBColorSpace;
    this.idTarget = new THREE.WebGLRenderTarget(SIZE, SIZE, { minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
    this.raycaster.firstHitOnly = true;
  }

  get started() {
    return this.session !== null;
  }

  dispose() {
    this.target.dispose();
    this.idTarget.dispose();
    this.paintMat.dispose();
    this.idMat.dispose();
    this.idGeo?.dispose();
  }

  // ---------- rendering ----------

  private camera(azimuth: number, elevation: number, zoom: number, focus: THREE.Vector3 | null): THREE.OrthographicCamera {
    const sphere = this.mesh.geometry.boundingSphere!;
    const R = sphere.radius;
    const z = Math.min(12, Math.max(1, zoom || 1));
    const half = (R * 1.05) / z;
    const center = focus ?? sphere.center;
    const az = (azimuth * Math.PI) / 180;
    const el = (Math.max(-89.5, Math.min(89.5, elevation)) * Math.PI) / 180;
    const dir = new THREE.Vector3(Math.sin(az) * Math.cos(el), -Math.cos(az) * Math.cos(el), Math.sin(el));
    const cam = new THREE.OrthographicCamera(-half, half, half, -half, 0.01, 1000);
    cam.up.set(0, 0, 1);
    cam.position.copy(center).addScaledVector(dir, R * 3);
    cam.lookAt(center);
    // Zoomed views clip geometry well in front of the focus (e.g. a tail coil in the way).
    cam.near = z > 1.5 ? R * 3 - half * 2.5 : R * 0.05;
    cam.far = R * 6;
    cam.updateProjectionMatrix();
    cam.updateMatrixWorld();
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(0.3, 0.5, 1);
    cam.add(key);
    return cam;
  }

  private readPixels(target: THREE.WebGLRenderTarget, cam: THREE.Camera): Uint8Array {
    this.scene.add(cam);
    const prev = this.renderer.getRenderTarget();
    this.renderer.setRenderTarget(target);
    this.renderer.render(this.scene, cam);
    const buf = new Uint8Array(SIZE * SIZE * 4);
    this.renderer.readRenderTargetPixels(target, 0, 0, SIZE, SIZE, buf);
    this.renderer.setRenderTarget(prev);
    this.scene.remove(cam);
    return buf; // bottom-up rows
  }

  private shoot(cam: THREE.OrthographicCamera, caption: string): Shot {
    const id = `img-${++this.shotNo}`;
    this.cams.set(id, cam);
    const buf = this.readPixels(this.target, cam);
    const img = new ImageData(SIZE, SIZE);
    for (let y = 0; y < SIZE; y++) img.data.set(buf.subarray((SIZE - 1 - y) * SIZE * 4, (SIZE - y) * SIZE * 4), y * SIZE * 4);
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = SIZE;
    const ctx = canvas.getContext('2d')!;
    ctx.putImageData(img, 0, 0);
    // Coordinate grid so pixel positions can be read off the image.
    ctx.strokeStyle = 'rgba(0, 90, 220, 0.28)';
    ctx.lineWidth = 1;
    ctx.fillStyle = 'rgba(0, 70, 180, 0.85)';
    ctx.font = '12px sans-serif';
    for (let p = 100; p < SIZE; p += 100) {
      ctx.beginPath(); ctx.moveTo(p + 0.5, 0); ctx.lineTo(p + 0.5, SIZE); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, p + 0.5); ctx.lineTo(SIZE, p + 0.5); ctx.stroke();
      ctx.fillText(String(p), p + 2, 12);
      ctx.fillText(String(p), 2, p - 2);
    }
    ctx.fillStyle = '#111';
    ctx.font = 'bold 14px sans-serif';
    ctx.fillText(`${id} · ${caption}`, 8, SIZE - 10);
    const shot = { id, caption, image: canvas.toDataURL('image/jpeg', 0.88) };
    this.events.onImage(shot);
    return shot;
  }

  look(azimuth: number, elevation: number, zoom = 1, focus: THREE.Vector3 | null = null): Shot {
    const cam = this.camera(azimuth, elevation, zoom, focus);
    const caption = `azimuth ${Math.round(azimuth)}°, elevation ${Math.round(elevation)}°, zoom ${zoom}${focus ? ' (focused)' : ''}`;
    return this.shoot(cam, caption);
  }

  overview(): Shot[] {
    return [this.look(0, 30), this.look(180, 30), this.look(0, 90), this.look(0, -60)];
  }

  /** Facts about the model for the opening message. */
  facts(): string {
    const size = this.mesh.boundingBox().getSize(new THREE.Vector3());
    const { shells: list } = shells(this.mesh);
    return `Model size ${size.x.toFixed(0)} × ${size.y.toFixed(0)} mm, ${size.z.toFixed(1)} mm tall (heights for range_fill are measured from its lowest point). It is made of ${list.length} separate part${list.length === 1 ? '' : 's'}.`;
  }

  /** Face id visible at every pixel of an image (top-down rows), -1 for background. */
  private faceIdsFor(cam: THREE.OrthographicCamera): Int32Array {
    if (!this.idGeo) {
      const F = this.mesh.faceCount;
      const fid = new Float32Array(F * 3);
      for (let f = 0; f < F; f++) fid[f * 3] = fid[f * 3 + 1] = fid[f * 3 + 2] = f;
      this.idGeo = new THREE.BufferGeometry();
      this.idGeo.setAttribute('position', this.mesh.geometry.getAttribute('position'));
      this.idGeo.setAttribute('fid', new THREE.BufferAttribute(fid, 1));
      this.idGeo.setIndex(this.mesh.geometry.index);
    }
    this.obj.geometry = this.idGeo; this.obj.material = this.idMat;
    const bg = this.scene.background;
    this.scene.background = new THREE.Color(0x000000);
    const buf = this.readPixels(this.idTarget, cam);
    this.scene.background = bg;
    this.obj.geometry = this.mesh.geometry; this.obj.material = this.paintMat;
    const ids = new Int32Array(SIZE * SIZE);
    for (let y = 0; y < SIZE; y++) {
      const row = (SIZE - 1 - y) * SIZE * 4;
      for (let x = 0; x < SIZE; x++) {
        const o = row + x * 4;
        ids[y * SIZE + x] = buf[o] + buf[o + 1] * 256 + buf[o + 2] * 65536 - 1;
      }
    }
    return ids;
  }

  /** Surface hit under a pixel of an earlier image. */
  private hit(imageId: string, x: number, y: number): { face: number; point: THREE.Vector3; cam: THREE.OrthographicCamera } | null {
    const cam = this.cams.get(imageId);
    if (!cam) return null;
    const ndc = new THREE.Vector2((x / SIZE) * 2 - 1, -(y / SIZE) * 2 + 1);
    this.raycaster.setFromCamera(ndc, cam);
    const h = this.raycaster.intersectObject(this.obj, false)[0];
    if (!h || !h.face) return null;
    return { face: Math.floor(h.face.a / 3), point: h.point, cam };
  }

  /** Visible faces inside a pixel polygon, with small hidden gaps closed. */
  private outlineFaces(cam: THREE.OrthographicCamera, pts: Pt[]): number[] {
    const ids = this.faceIdsFor(cam);
    const mask = document.createElement('canvas');
    mask.width = mask.height = SIZE;
    const ctx = mask.getContext('2d', { willReadFrequently: true })!;
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    pts.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
    ctx.closePath();
    ctx.fill();
    const m = ctx.getImageData(0, 0, SIZE, SIZE).data;
    const sel = new Uint8Array(this.mesh.faceCount);
    const out: number[] = [];
    for (let i = 0; i < SIZE * SIZE; i++) {
      const f = ids[i];
      if (f >= 0 && m[i * 4] > 127 && !sel[f]) { sel[f] = 1; out.push(f); }
    }
    // Triangles smaller than a pixel can fall between samples: take faces
    // whose neighbours were mostly selected and whose centre projects inside.
    const v = new THREE.Vector3();
    const adj = this.mesh.adjacency, c = this.mesh.centroids;
    for (let pass = 0; pass < 3; pass++) {
      const add: number[] = [];
      for (const f of out) {
        for (let k = 0; k < 3; k++) {
          const g = adj[f * 3 + k];
          if (g < 0 || sel[g]) continue;
          let n = 0;
          for (let j = 0; j < 3; j++) { const h = adj[g * 3 + j]; if (h >= 0 && sel[h]) n++; }
          if (n < 2) continue;
          v.set(c[g * 3], c[g * 3 + 1], c[g * 3 + 2]).project(cam);
          const px = Math.round((v.x + 1) * 0.5 * SIZE), py = Math.round((1 - v.y) * 0.5 * SIZE);
          if (px >= 0 && py >= 0 && px < SIZE && py < SIZE && m[(py * SIZE + px) * 4] > 127) add.push(g);
        }
      }
      for (const g of add) if (!sel[g]) { sel[g] = 1; out.push(g); }
      if (!add.length) break;
    }
    return out;
  }

  private slot(name: unknown): number {
    const n = String(name ?? '').trim().toLowerCase();
    return this.filaments.findIndex((f) => f.name.trim().toLowerCase() === n || f.hex.toLowerCase() === n);
  }

  private total = 0;
  private totalArea() {
    if (!this.total) for (let f = 0; f < this.mesh.faceCount; f++) this.total += this.mesh.areas[f];
    return this.total;
  }

  /** How much of the model each filament covers now. */
  private shares(): string {
    const acc = new Float64Array(this.filaments.length);
    for (let f = 0; f < this.mesh.faceCount; f++) acc[this.mesh.slots[f]] += this.mesh.areas[f];
    const t = this.totalArea();
    return 'Model now: ' + this.filaments.map((fl, i) => `${fl.name} ${((100 * acc[i]) / t).toFixed(1)}%`).join(', ');
  }

  /** Height and horizontal reach of a part, so range fills can be aimed. */
  private partStats(faces: ArrayLike<number>, center: [number, number]): string {
    const c = this.mesh.centroids, base = this.mesh.boundingBox().min.z;
    let zLo = Infinity, zHi = -Infinity, rHi = 0;
    for (let i = 0; i < faces.length; i++) {
      const f = faces[i];
      const z = c[f * 3 + 2] - base;
      zLo = Math.min(zLo, z); zHi = Math.max(zHi, z);
      rHi = Math.max(rHi, Math.hypot(c[f * 3] - center[0], c[f * 3 + 1] - center[1]));
    }
    return `That part spans heights ${zLo.toFixed(1)}–${zHi.toFixed(1)} mm and reaches ${rHi.toFixed(1)} mm horizontally from its centre.`;
  }

  private areaOf(faces: number[]) {
    let a = 0;
    for (const f of faces) a += this.mesh.areas[f];
    return a;
  }

  // ---------- tools ----------

  private async stroke(s: Record<string, unknown>, i: number): Promise<{ line: string; cam?: THREE.OrthographicCamera }> {
    const slot = this.slot(s.color);
    if (slot < 0) return { line: `${i}. unknown filament "${s.color}"` };
    const image = String(s.image);
    const cam = this.cams.get(image);
    if (!cam) return { line: `${i}. unknown image ${image}` };
    const tool = String(s.tool);
    const pts = (s.points as Pt[] | null) ?? [];
    let faces: number[] = [];
    let extra = '';

    if (tool === 'outline') {
      if (pts.length < 3) return { line: `${i}. outline needs at least 3 points` };
      faces = this.outlineFaces(cam, pts);
    } else if (tool === 'line') {
      if (pts.length < 2) return { line: `${i}. line needs at least 2 points` };
      const r = s.radius_mm == null ? 1.5 : Number(s.radius_mm);
      const d = new THREE.Vector3();
      cam.getWorldDirection(d);
      const set = new Set<number>();
      // Pixels per mm in this view, to space samples about half a radius apart.
      const pxPerMm = SIZE / (cam.right - cam.left);
      const step = Math.max(1, (r * pxPerMm) / 2);
      let misses = 0, samples = 0;
      for (let k = 1; k < pts.length; k++) {
        const a = pts[k - 1], b = pts[k];
        const len = Math.hypot(b.x - a.x, b.y - a.y);
        const n = Math.max(1, Math.ceil(len / step));
        for (let t = 0; t <= n; t++) {
          const h = this.hit(image, a.x + ((b.x - a.x) * t) / n, a.y + ((b.y - a.y) * t) / n);
          samples++;
          if (!h) { misses++; continue; }
          for (const f of brushFaces(this.mesh, h.face, [h.point.x, h.point.y, h.point.z], r, [d.x, d.y, d.z])) set.add(f);
        }
      }
      faces = [...set];
      if (misses) extra = `, ${misses}/${samples} path samples missed the model`;
    } else {
      if (s.x == null || s.y == null) return { line: `${i}. ${tool} needs x and y` };
      const h = this.hit(image, Number(s.x), Number(s.y));
      if (!h) return { line: `${i}. (${s.x}, ${s.y}) on ${image} missed the model` };
      const angle = s.angle_deg == null ? 30 : Number(s.angle_deg);
      const maxT = s.max_thickness_mm == null ? 5 : Number(s.max_thickness_mm);
      if (tool === 'brush') {
        const d = new THREE.Vector3();
        cam.getWorldDirection(d);
        faces = brushFaces(this.mesh, h.face, [h.point.x, h.point.y, h.point.z], s.radius_mm == null ? 1.5 : Number(s.radius_mm), [d.x, d.y, d.z]);
      } else if (tool === 'fill_similar') {
        const r = similarFaces(this.mesh, h.face, angle);
        faces = r.faces;
        extra = `, ${r.matches} matching features`;
      } else if (tool === 'thin_fill' || tool === 'thin_fill_similar') {
        this.events.onStatus('Measuring thickness (first time only)…');
        const th = await computeThickness(this.mesh);
        extra = `, thickness at click ${th[h.face].toFixed(1)} mm`;
        if (tool === 'thin_fill') faces = thinFill(this.mesh, th, h.face, maxT);
        else {
          const r = similarThinFaces(this.mesh, th, h.face, maxT);
          faces = r.faces;
          extra += `, ${r.matches} matching parts`;
        }
      } else {
        faces = creaseFill(this.mesh, h.face, angle);
      }
    }
    this.history.paint(faces, slot);
    const pct = (100 * this.areaOf(faces)) / this.totalArea();
    // One feature (a spike, an eye, a tooth) is a small share of the model; a big share means it leaked.
    const warn = pct > 6 ? ` ⚠ this stroke covered ${pct.toFixed(0)}% of the whole model, far more than one feature: if you meant a single part it leaked, undo` : '';
    return { line: `${i}. ${tool} ${this.filaments[slot].name}: ${this.areaOf(faces).toFixed(0)} mm², ${pct.toFixed(1)}% of model (${faces.length} triangles${extra})${warn}`, cam };
  }

  private async exec(call: ToolCall): Promise<ToolResult> {
    const inp = call.input;
    switch (call.name) {
      case 'look': {
        const f = inp.focus as Pt & { image: string } | null;
        let focus: THREE.Vector3 | null = null;
        if (f) {
          const h = this.hit(f.image, f.x, f.y);
          if (!h) return { id: call.id, text: `focus pixel (${f.x}, ${f.y}) on ${f.image} is not on the model`, error: true };
          focus = h.point;
        }
        const shot = this.look(Number(inp.azimuth), Number(inp.elevation), Number(inp.zoom) || 1, focus);
        return { id: call.id, text: `Rendered ${shot.id}.`, images: [shot] };
      }
      case 'paint': {
        const strokes = (inp.strokes as Record<string, unknown>[]) ?? [];
        const lines: string[] = [];
        let lastCam: THREE.OrthographicCamera | undefined, lastImage = '';
        for (let i = 0; i < strokes.length; i++) {
          const r = await this.stroke(strokes[i], i + 1);
          lines.push(r.line);
          if (r.cam) { lastCam = r.cam; lastImage = String(strokes[i].image); }
        }
        this.history.commit();
        this.undoable++;
        lines.push(this.shares());
        const images = lastCam ? [this.shoot(lastCam.clone() as THREE.OrthographicCamera, `after painting, same view as ${lastImage}`)] : [];
        return { id: call.id, text: lines.join('\n'), images };
      }
      case 'range_fill': {
        const slot = this.slot(inp.color);
        if (slot < 0) return { id: call.id, text: `unknown filament "${inp.color}"`, error: true };
        const { shellOf, shells: list } = shells(this.mesh);
        let partFaces: Uint32Array | undefined;
        let center: [number, number] = [0, 0];
        let where = 'whole model';
        let cam: THREE.OrthographicCamera | undefined;
        if (inp.image != null && inp.x != null && inp.y != null) {
          const h = this.hit(String(inp.image), Number(inp.x), Number(inp.y));
          if (!h) return { id: call.id, text: `(${inp.x}, ${inp.y}) on ${inp.image} missed the model`, error: true };
          const part = list[shellOf[h.face]];
          partFaces = part.faces;
          center = [part.center[0], part.center[1]];
          where = 'clicked part only';
          cam = h.cam;
        } else if (inp.r_min != null || inp.r_max != null) {
          return { id: call.id, text: 'r_min / r_max need a clicked part (image, x, y) to measure from.', error: true };
        }
        const faces = rangeFaces(this.mesh, { zMin: num(inp.z_min), zMax: num(inp.z_max), rMin: num(inp.r_min), rMax: num(inp.r_max), center }, partFaces);
        this.history.paint(faces, slot);
        this.history.commit();
        this.undoable++;
        const pct = (100 * this.areaOf(faces)) / this.totalArea();
        const lines = [`range_fill ${this.filaments[slot].name} (${where}): ${this.areaOf(faces).toFixed(0)} mm², ${pct.toFixed(1)}% of model`];
        if (partFaces) lines.push(this.partStats(partFaces, center));
        lines.push(this.shares());
        const images = cam ? [this.shoot(cam.clone() as THREE.OrthographicCamera, `after range_fill, same view as ${inp.image}`)] : [];
        return { id: call.id, text: lines.join('\n'), images };
      }
      case 'segment_rule_fill': {
        const slot = this.slot(inp.color);
        if (slot < 0) return { id: call.id, text: `unknown filament "${inp.color}"`, error: true };
        const h = this.hit(String(inp.image), Number(inp.x), Number(inp.y));
        if (!h) return { id: call.id, text: `(${inp.x}, ${inp.y}) on ${inp.image} missed the model`, error: true };
        const { shellOf, shells: list } = shells(this.mesh);
        const parts = similarShells(this.mesh, shellOf[h.face]);
        const hf = Number(inp.height_frac), rf = Number(inp.reach_frac), sf = Number(inp.stripe_frac ?? 0);
        const faces = segmentRuleFaces(this.mesh, parts, hf, rf, sf);
        this.history.paint(faces, slot);
        // Smooth the cut lines inside those parts, then fold that into the same undo step.
        const scope: number[] = [];
        for (const p of parts) for (const f of list[p].faces) scope.push(f);
        const smoothed = smoothEdges(this.mesh, 2, scope);
        this.mesh.refreshColors();
        this.history.recordApplied(smoothed);
        this.undoable++;
        const pct = (100 * this.areaOf(faces)) / this.totalArea();
        const shot = this.shoot(h.cam.clone() as THREE.OrthographicCamera, `after segment_rule_fill, same view as ${inp.image}`);
        return {
          id: call.id,
          text: `segment_rule_fill ${this.filaments[slot].name} on ${parts.length} matching parts (height > ${hf}, reach > ${rf}): ${pct.toFixed(1)}% of model.\n${this.shares()}`,
          images: [shot],
        };
      }
      case 'smooth_edges': {
        const changed = smoothEdges(this.mesh, 2);
        this.mesh.refreshColors();
        this.history.recordApplied(changed);
        this.undoable++;
        return { id: call.id, text: `Smoothed ${changed.size} border triangles.` };
      }
      case 'copy_to_matching_parts': {
        const h = this.hit(String(inp.image), Number(inp.x), Number(inp.y));
        if (!h) return { id: call.id, text: `(${inp.x}, ${inp.y}) on ${inp.image} missed the model`, error: true };
        const { shellOf, shells: list } = shells(this.mesh);
        const r = copyPaint(this.mesh, shellOf[h.face]);
        this.mesh.refreshColors();
        this.history.recordApplied(r.faces);
        this.undoable++;
        const lines = [
          `The model has ${list.length} separate parts. Copied the clicked part's paint onto ${r.copied.length} matching parts.`,
          r.skipped.length ? `Skipped ${r.skipped.length} similar-looking parts that didn't align well.` : '',
          this.shares(),
        ];
        return { id: call.id, text: lines.filter(Boolean).join('\n') };
      }
      case 'fill_all': {
        const slot = this.slot(inp.color);
        if (slot < 0) return { id: call.id, text: `unknown filament "${inp.color}"`, error: true };
        this.history.paint(Array.from({ length: this.mesh.faceCount }, (_, i) => i), slot);
        this.history.commit();
        this.undoable++;
        return { id: call.id, text: `Whole model is now ${this.filaments[slot].name}.
${this.shares()}` };
      }
      case 'undo': {
        if (!this.undoable) return { id: call.id, text: 'Nothing to undo.', error: true };
        this.history.undo();
        this.undoable--;
        return { id: call.id, text: `Reverted the last paint call.
${this.shares()}` };
      }
      case 'done':
        return { id: call.id, text: 'ok' };
      default:
        return { id: call.id, text: `unknown tool ${call.name}`, error: true };
    }
  }

  /** First message of the chat: show the model and hand over the brief. */
  async start(brief: string, reference: string | null): Promise<void> {
    this.events.onStatus('Rendering overview…');
    const overview = this.overview();
    this.events.onStatus('Looking at the model…');
    const turn = await post<Turn>('/api/agent/start', { description: brief, filaments: this.filaments, reference_image: reference, overview, facts: this.facts() });
    this.session = turn.session!;
    await this.loop(turn);
  }

  /** A follow-up request in the same chat. */
  async send(message: string): Promise<void> {
    if (!this.session) throw new Error('No session');
    // The previous request ended with calls (done, or ones cut off by Stop) that still need results.
    const results = this.pending;
    this.pending = [];
    this.stopped = false;
    this.events.onStatus('Thinking…');
    const turn = await post<Turn>('/api/agent/step', { session: this.session, results, message });
    await this.loop(turn);
  }

  private async loop(turn: Turn, maxTurns = 45): Promise<void> {
    for (let i = 0; i < maxTurns; i++) {
      this.cost += turn.cost ?? 0;
      if (turn.text) this.events.onSay(turn.text);
      const done = turn.calls.find((c) => c.name === 'done');
      if (done) {
        this.pending = turn.calls.map((c) => ({ id: c.id, text: 'ok' }));
        this.events.onSay(String(done.input.summary ?? 'Done.'));
        return;
      }
      if (turn.finished) return;
      if (this.stopped) {
        // Answer the open calls with the next message so the chat can continue.
        this.pending = turn.calls.map((c) => ({ id: c.id, text: 'Stopped by the user before this ran.', error: true }));
        this.events.onSay('Stopped. Send a message to continue or change course.');
        return;
      }
      const results: ToolResult[] = [];
      for (const call of turn.calls) {
        this.events.onAction(`${call.name} ${summarise(call)}`);
        let res: ToolResult;
        try { res = await this.exec(call); } catch (err) { res = { id: call.id, text: `tool failed: ${(err as Error).message}`, error: true }; }
        results.push(res);
      }
      this.events.onStatus(`Step ${i + 1}: thinking… ($${this.cost.toFixed(2)} so far)`);
      turn = await post<Turn>('/api/agent/step', { session: this.session, results });
    }
    this.pending = turn.calls.map((c) => ({ id: c.id, text: 'Paused at the step limit before this ran.', error: true }));
    this.events.onSay('Paused after the step limit. Send a message to continue.');
  }
}

function num(v: unknown): number | null {
  return v == null ? null : Number(v);
}

function summarise(call: ToolCall): string {
  const i = call.input;
  if (call.name === 'look') return `az ${i.azimuth} el ${i.elevation} zoom ${i.zoom}`;
  if (call.name === 'paint') {
    const s = (i.strokes as { tool: string; color: string }[]) ?? [];
    const counts = new Map<string, number>();
    for (const x of s) counts.set(`${x.tool} ${x.color}`, (counts.get(`${x.tool} ${x.color}`) ?? 0) + 1);
    return [...counts].map(([k, n]) => (n > 1 ? `${n}× ${k}` : k)).join(', ');
  }
  if (call.name === 'fill_all') return String(i.color);
  return '';
}
