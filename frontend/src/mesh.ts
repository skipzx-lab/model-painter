import * as THREE from 'three';
import { MeshBVH } from 'three-mesh-bvh';

/**
 * A welded triangle mesh with one filament slot per face.
 *
 * Coordinates stay in the file's own frame (Z up, millimetres) so export is a
 * straight write-back. The display geometry is non-indexed (3 corners per
 * face) so every face can carry its own flat colour.
 */
export class PaintMesh {
  positions: Float32Array; // welded vertices, xyz
  faces: Uint32Array; // 3 vertex indices per face
  slots: Uint8Array; // palette index per face
  faceNormals!: Float32Array;
  centroids!: Float32Array;
  areas!: Float32Array;
  adjacency!: Int32Array; // 3 neighbouring faces per face, -1 when open

  geometry!: THREE.BufferGeometry;
  colorAttr!: THREE.BufferAttribute;
  bvh!: MeshBVH;
  palette: THREE.Color[] = [];

  constructor(positions: Float32Array, faces: Uint32Array, slots?: Uint8Array) {
    this.positions = positions;
    this.faces = faces;
    this.slots = slots ?? new Uint8Array(faces.length / 3);
    this.rebuild();
  }

  get faceCount() {
    return this.faces.length / 3;
  }

  get vertexCount() {
    return this.positions.length / 3;
  }

  /** Weld triangle soup into an indexed mesh, dropping degenerate faces. */
  static fromGeometry(src: THREE.BufferGeometry): PaintMesh {
    const g = src.index ? src.toNonIndexed() : src;
    const pos = g.getAttribute('position');
    const n = pos.count;
    const box = new THREE.Box3().setFromBufferAttribute(pos as THREE.BufferAttribute);
    const size = box.getSize(new THREE.Vector3()).length() || 1;
    const tol = size * 1e-6;
    const inv = 1 / tol;

    const map = new Map<string, number>();
    const welded: number[] = [];
    const remap = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
      const key = `${Math.round(x * inv)},${Math.round(y * inv)},${Math.round(z * inv)}`;
      let id = map.get(key);
      if (id === undefined) {
        id = welded.length / 3;
        map.set(key, id);
        welded.push(x, y, z);
      }
      remap[i] = id;
    }

    const faces: number[] = [];
    for (let i = 0; i + 2 < n; i += 3) {
      const a = remap[i], b = remap[i + 1], c = remap[i + 2];
      if (a === b || b === c || a === c) continue;
      faces.push(a, b, c);
    }
    return new PaintMesh(new Float32Array(welded), new Uint32Array(faces));
  }

  /**
   * From an indexed mesh (e.g. 3MF): weld coincident vertices and drop
   * degenerate faces. `keep` receives the surviving original face indices.
   */
  static fromIndexed(positions: Float32Array, faces: Uint32Array, keep?: number[]): PaintMesh {
    const n = positions.length / 3;
    let minX = Infinity, minY = Infinity, minZ = Infinity, maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < positions.length; i += 3) {
      minX = Math.min(minX, positions[i]); maxX = Math.max(maxX, positions[i]);
      minY = Math.min(minY, positions[i + 1]); maxY = Math.max(maxY, positions[i + 1]);
      minZ = Math.min(minZ, positions[i + 2]); maxZ = Math.max(maxZ, positions[i + 2]);
    }
    const inv = 1 / ((Math.hypot(maxX - minX, maxY - minY, maxZ - minZ) || 1) * 1e-6);
    const map = new Map<string, number>();
    const welded: number[] = [];
    const remap = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      const x = positions[i * 3], y = positions[i * 3 + 1], z = positions[i * 3 + 2];
      const key = `${Math.round(x * inv)},${Math.round(y * inv)},${Math.round(z * inv)}`;
      let id = map.get(key);
      if (id === undefined) { id = welded.length / 3; map.set(key, id); welded.push(x, y, z); }
      remap[i] = id;
    }
    const out: number[] = [];
    for (let i = 0; i < faces.length; i += 3) {
      const a = remap[faces[i]], b = remap[faces[i + 1]], c = remap[faces[i + 2]];
      if (a === b || b === c || a === c) continue;
      out.push(a, b, c);
      keep?.push(i / 3);
    }
    return new PaintMesh(new Float32Array(welded), new Uint32Array(out));
  }

  /** Recompute derived data after topology changes. */
  rebuild() {
    this.computeFaceData();
    this.computeAdjacency();
    this.buildDisplayGeometry();
  }

  private computeFaceData() {
    const F = this.faceCount;
    const p = this.positions, f = this.faces;
    this.faceNormals = new Float32Array(F * 3);
    this.centroids = new Float32Array(F * 3);
    this.areas = new Float32Array(F);
    for (let i = 0; i < F; i++) {
      const a = f[i * 3] * 3, b = f[i * 3 + 1] * 3, c = f[i * 3 + 2] * 3;
      const e1x = p[b] - p[a], e1y = p[b + 1] - p[a + 1], e1z = p[b + 2] - p[a + 2];
      const e2x = p[c] - p[a], e2y = p[c + 1] - p[a + 1], e2z = p[c + 2] - p[a + 2];
      let nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
      const len = Math.hypot(nx, ny, nz);
      this.areas[i] = len / 2;
      if (len > 0) { nx /= len; ny /= len; nz /= len; }
      this.faceNormals.set([nx, ny, nz], i * 3);
      this.centroids[i * 3] = (p[a] + p[b] + p[c]) / 3;
      this.centroids[i * 3 + 1] = (p[a + 1] + p[b + 1] + p[c + 1]) / 3;
      this.centroids[i * 3 + 2] = (p[a + 2] + p[b + 2] + p[c + 2]) / 3;
    }
  }

  private computeAdjacency() {
    const F = this.faceCount, f = this.faces;
    this.adjacency = new Int32Array(F * 3).fill(-1);
    // Edge key -> (face*3 + corner) of the first face seen on that edge.
    const edges = new Map<number, number>();
    const V = this.vertexCount;
    const useNumericKey = V < 94906265; // V*V must stay below 2^53
    const strEdges = new Map<string, number>();
    for (let i = 0; i < F; i++) {
      for (let k = 0; k < 3; k++) {
        const u = f[i * 3 + k], v = f[i * 3 + ((k + 1) % 3)];
        const lo = Math.min(u, v), hi = Math.max(u, v);
        const slot = i * 3 + k;
        let other: number | undefined;
        if (useNumericKey) {
          const key = lo * V + hi;
          other = edges.get(key);
          if (other === undefined) edges.set(key, slot);
          else edges.delete(key); // manifold edge: pair once, ignore extra faces
        } else {
          const key = `${lo},${hi}`;
          other = strEdges.get(key);
          if (other === undefined) strEdges.set(key, slot);
          else strEdges.delete(key);
        }
        if (other !== undefined) {
          this.adjacency[slot] = Math.floor(other / 3);
          this.adjacency[other] = i;
        }
      }
    }
  }

  private buildDisplayGeometry() {
    const F = this.faceCount, p = this.positions, f = this.faces;

    // Area-weighted smooth vertex normals for nicer shading on organic models.
    const vn = new Float32Array(p.length);
    for (let i = 0; i < F; i++) {
      const w = this.areas[i];
      for (let k = 0; k < 3; k++) {
        const v = f[i * 3 + k] * 3;
        vn[v] += this.faceNormals[i * 3] * w;
        vn[v + 1] += this.faceNormals[i * 3 + 1] * w;
        vn[v + 2] += this.faceNormals[i * 3 + 2] * w;
      }
    }
    for (let v = 0; v < vn.length; v += 3) {
      const l = Math.hypot(vn[v], vn[v + 1], vn[v + 2]) || 1;
      vn[v] /= l; vn[v + 1] /= l; vn[v + 2] /= l;
    }

    const pos = new Float32Array(F * 9);
    const nor = new Float32Array(F * 9);
    for (let i = 0; i < F; i++) {
      for (let k = 0; k < 3; k++) {
        const v = f[i * 3 + k] * 3, o = i * 9 + k * 3;
        pos[o] = p[v]; pos[o + 1] = p[v + 1]; pos[o + 2] = p[v + 2];
        nor[o] = vn[v]; nor[o + 1] = vn[v + 1]; nor[o + 2] = vn[v + 2];
      }
    }
    this.geometry?.dispose();
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
    this.colorAttr = new THREE.BufferAttribute(new Uint8Array(F * 9), 3, true);
    g.setAttribute('color', this.colorAttr);
    // The BVH writes an index that reorders triangles; corner vertex v always
    // belongs to face floor(v / 3), so face ids remain recoverable.
    this.bvh = new MeshBVH(g);
    g.boundsTree = this.bvh;
    g.computeBoundingBox();
    g.computeBoundingSphere();
    this.geometry = g;
    this.refreshColors();
  }

  /** Face id from a raycast hit on the display geometry. */
  faceFromHit(hit: THREE.Intersection): number {
    return Math.floor(hit.face!.a / 3);
  }

  setPalette(colors: string[]) {
    this.palette = colors.map((c) => new THREE.Color(c).convertSRGBToLinear());
    this.refreshColors();
  }

  setSlot(face: number, slot: number) {
    this.slots[face] = slot;
    this.writeColor(face);
  }

  private writeColor(face: number) {
    const c = this.palette[this.slots[face]] ?? this.palette[0];
    if (!c) return;
    const arr = this.colorAttr.array as Uint8Array;
    // three.js treats vertex colours as linear, so palette colours were converted on setPalette.
    const r = Math.round(THREE.MathUtils.clamp(c.r, 0, 1) * 255);
    const gg = Math.round(THREE.MathUtils.clamp(c.g, 0, 1) * 255);
    const b = Math.round(THREE.MathUtils.clamp(c.b, 0, 1) * 255);
    const o = face * 9;
    for (let k = 0; k < 9; k += 3) { arr[o + k] = r; arr[o + k + 1] = gg; arr[o + k + 2] = b; }
  }

  refreshColors() {
    if (!this.colorAttr || !this.palette.length) return;
    for (let i = 0; i < this.faceCount; i++) this.writeColor(i);
    this.colorAttr.needsUpdate = true;
  }

  markColorsDirty() {
    this.colorAttr.needsUpdate = true;
  }

  /**
   * Split long edges until every edge is <= maxEdge, keeping the mesh
   * conforming (no T-junctions). Gives the paint finer resolution on
   * low-poly models. Returns the number of new faces.
   */
  refine(maxEdge: number, maxFaces = 3_000_000): number {
    const before = this.faceCount;
    let pos = Array.from(this.positions);
    let faces = Array.from(this.faces);
    let slots = Array.from(this.slots);
    const max2 = maxEdge * maxEdge;

    for (let pass = 0; pass < 12; pass++) {
      const mid = new Map<string, number>();
      const len2 = (a: number, b: number) => {
        const dx = pos[a * 3] - pos[b * 3], dy = pos[a * 3 + 1] - pos[b * 3 + 1], dz = pos[a * 3 + 2] - pos[b * 3 + 2];
        return dx * dx + dy * dy + dz * dz;
      };
      const key = (a: number, b: number) => (a < b ? `${a},${b}` : `${b},${a}`);
      // Mark long edges.
      for (let i = 0; i < faces.length; i += 3) {
        for (let k = 0; k < 3; k++) {
          const a = faces[i + k], b = faces[i + ((k + 1) % 3)];
          const kk = key(a, b);
          if (!mid.has(kk) && len2(a, b) > max2) mid.set(kk, -1);
        }
      }
      if (!mid.size) break;
      if (faces.length / 3 + mid.size * 2 > maxFaces) break;
      for (const kk of mid.keys()) {
        const [a, b] = kk.split(',').map(Number);
        const id = pos.length / 3;
        pos.push((pos[a * 3] + pos[b * 3]) / 2, (pos[a * 3 + 1] + pos[b * 3 + 1]) / 2, (pos[a * 3 + 2] + pos[b * 3 + 2]) / 2);
        mid.set(kk, id);
      }
      const nf: number[] = [], ns: number[] = [];
      const push = (a: number, b: number, c: number, s: number) => { nf.push(a, b, c); ns.push(s); };
      for (let i = 0; i < faces.length; i += 3) {
        const v = [faces[i], faces[i + 1], faces[i + 2]];
        const s = slots[i / 3];
        const m = [0, 1, 2].map((k) => mid.get(key(v[k], v[(k + 1) % 3])) ?? -1);
        const count = m.filter((x) => x >= 0).length;
        if (count === 0) push(v[0], v[1], v[2], s);
        else if (count === 3) {
          push(v[0], m[0], m[2], s); push(m[0], v[1], m[1], s); push(m[2], m[1], v[2], s); push(m[0], m[1], m[2], s);
        } else if (count === 1) {
          const k = m.findIndex((x) => x >= 0);
          const a = v[k], b = v[(k + 1) % 3], c = v[(k + 2) % 3], ab = m[k];
          push(a, ab, c, s); push(ab, b, c, s);
        } else {
          // Two split edges: the unsplit one is k.
          const k = m.findIndex((x) => x < 0);
          const a = v[k], b = v[(k + 1) % 3], c = v[(k + 2) % 3];
          const bc = m[(k + 1) % 3], ca = m[(k + 2) % 3];
          push(ca, bc, c, s);
          // Split the quad a-b-bc-ca along its shorter diagonal.
          if (len2(a, bc) < len2(b, ca)) { push(a, b, bc, s); push(a, bc, ca, s); }
          else { push(a, b, ca, s); push(b, bc, ca, s); }
        }
      }
      faces = nf; slots = ns;
    }
    this.positions = new Float32Array(pos);
    this.faces = new Uint32Array(faces);
    this.slots = new Uint8Array(slots);
    this.rebuild();
    return this.faceCount - before;
  }

  /** Median edge length, used to pick sensible brush and refine defaults. */
  medianEdge(): number {
    const sample: number[] = [];
    const step = Math.max(1, Math.floor(this.faceCount / 5000));
    for (let i = 0; i < this.faceCount; i += step) {
      const a = this.faces[i * 3] * 3, b = this.faces[i * 3 + 1] * 3;
      const p = this.positions;
      sample.push(Math.hypot(p[a] - p[b], p[a + 1] - p[b + 1], p[a + 2] - p[b + 2]));
    }
    sample.sort((x, y) => x - y);
    return sample[Math.floor(sample.length / 2)] || 1;
  }

  boundingBox(): THREE.Box3 {
    return this.geometry.boundingBox!.clone();
  }
}
