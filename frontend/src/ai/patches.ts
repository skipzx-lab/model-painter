import type { PaintMesh } from '../mesh';

export interface Patches {
  ids: Int32Array; // patch per face, -1 for faces outside the segmented subset
  count: number;
  area: Float64Array;
  /** Patches sharing a border, for filling patches the AI never saw. */
  neighbours: Set<number>[];
}

/**
 * Split the mesh into about `target` regions whose borders follow the
 * sculpt's own creases, so spikes, claws, teeth and eyes come out as their
 * own regions instead of being cut through. Graph-based segmentation
 * (Felzenszwalb-Huttenlocher) over the face graph: edges are weighted by how
 * sharply concave the fold between two faces is, and regions merge across
 * an edge only when it is weaker than both regions' internal variation plus
 * a size allowance. The allowance is tuned by bisection to hit the target.
 *
 * `mask` limits segmentation to some faces (used for close-up detail
 * passes); other faces get id -1.
 */
export function buildPatches(mesh: PaintMesh, target: number, mask?: Uint8Array): Patches {
  const F = mesh.faceCount, c = mesh.centroids, n = mesh.faceNormals, adj = mesh.adjacency, areas = mesh.areas;
  const inSet = (f: number) => !mask || mask[f] === 1;

  // Edge list with crease weights.
  const eu: number[] = [], ev: number[] = [], ew: number[] = [];
  let totalArea = 0;
  for (let f = 0; f < F; f++) {
    if (!inSet(f)) continue;
    totalArea += areas[f];
    for (let k = 0; k < 3; k++) {
      const g = adj[f * 3 + k];
      if (g <= f || !inSet(g)) continue;
      const dot = n[f * 3] * n[g * 3] + n[f * 3 + 1] * n[g * 3 + 1] + n[f * 3 + 2] * n[g * 3 + 2];
      const dx = c[g * 3] - c[f * 3], dy = c[g * 3 + 1] - c[f * 3 + 1], dz = c[g * 3 + 2] - c[f * 3 + 2];
      const concave = dx * n[f * 3] + dy * n[f * 3 + 1] + dz * n[f * 3 + 2] > 0;
      // Valleys separate parts; ridges (spike edges, horn tips) mostly don't.
      eu.push(f); ev.push(g); ew.push((concave ? 1 : 0.12) * Math.max(0, 1 - dot));
    }
  }
  const E = eu.length;
  const W = Float32Array.from(ew);
  const order = new Uint32Array(E);
  for (let i = 0; i < E; i++) order[i] = i;
  order.sort((a, b) => W[a] - W[b]);

  const parent = new Int32Array(F);
  const area = new Float64Array(F);
  const internal = new Float32Array(F);
  const find = (x: number) => {
    while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
  };

  const run = (kk: number) => {
    for (let f = 0; f < F; f++) { parent[f] = f; area[f] = areas[f]; internal[f] = 0; }
    for (let i = 0; i < E; i++) {
      const e = order[i];
      const a = find(eu[e]), b = find(ev[e]);
      if (a === b) continue;
      const w = W[e];
      if (w <= Math.min(internal[a] + kk / area[a], internal[b] + kk / area[b])) {
        parent[b] = a; area[a] += area[b]; internal[a] = w;
      }
    }
    // Fold in crumbs too small to print or name, along their weakest border.
    const minArea = totalArea / (target * 12);
    for (let i = 0; i < E; i++) {
      const e = order[i];
      const a = find(eu[e]), b = find(ev[e]);
      if (a !== b && (area[a] < minArea || area[b] < minArea)) { parent[b] = a; area[a] += area[b]; }
    }
    let count = 0;
    for (let f = 0; f < F; f++) if (inSet(f) && find(f) === f) count++;
    return count;
  };

  // More allowance -> fewer, bigger regions. Bisect in log space.
  const k0 = (0.05 * totalArea) / Math.max(1, target);
  let lo = Math.log(k0 / 1e4), hi = Math.log(k0 * 1e4);
  let bestK = k0, bestErr = Infinity;
  for (let it = 0; it < 16; it++) {
    const mid = (lo + hi) / 2;
    const cnt = run(Math.exp(mid));
    const err = Math.abs(cnt - target);
    if (err < bestErr) { bestErr = err; bestK = Math.exp(mid); }
    if (err <= target * 0.05) break;
    if (cnt > target) lo = mid; else hi = mid;
  }
  run(bestK);

  const ids = new Int32Array(F).fill(-1);
  const label = new Map<number, number>();
  for (let f = 0; f < F; f++) {
    if (!inSet(f)) continue;
    const r = find(f);
    let id = label.get(r);
    if (id === undefined) { id = label.size; label.set(r, id); }
    ids[f] = id;
  }
  const count = label.size;
  const out = new Float64Array(count);
  const neighbours = Array.from({ length: count }, () => new Set<number>());
  for (let f = 0; f < F; f++) {
    const p = ids[f];
    if (p < 0) continue;
    out[p] += areas[f];
    for (let k = 0; k < 3; k++) {
      const g = adj[f * 3 + k];
      if (g >= 0 && ids[g] >= 0 && ids[g] !== p) neighbours[p].add(ids[g]);
    }
  }
  return { ids, count, area: out, neighbours };
}

export function resolveAssignments(patches: Patches, assigned: Map<number, number>, fallback: number): Int32Array {
  const slot = new Int32Array(patches.count).fill(-1);
  for (const [p, s] of assigned) if (p >= 0 && p < patches.count) slot[p] = s;
  for (let round = 0; round < 50; round++) {
    let changed = false;
    const nextSlot = slot.slice();
    for (let p = 0; p < patches.count; p++) {
      if (slot[p] >= 0) continue;
      const votes = new Map<number, number>();
      for (const q of patches.neighbours[p]) if (slot[q] >= 0) votes.set(slot[q], (votes.get(slot[q]) ?? 0) + patches.area[q]);
      let best = -1, bestV = 0;
      for (const [s, v] of votes) if (v > bestV) { best = s; bestV = v; }
      if (best >= 0) { nextSlot[p] = best; changed = true; }
    }
    slot.set(nextSlot);
    if (!changed) break;
  }
  for (let p = 0; p < patches.count; p++) if (slot[p] < 0) slot[p] = fallback;
  return slot;
}

/**
 * Faces near a set of patches: everything within the patches' bounding
 * sphere (grown a little), for a finer close-up pass on eyes, teeth etc.
 */
export function focusArea(mesh: PaintMesh, patches: Patches, regions: number[]): { mask: Uint8Array; center: [number, number, number]; radius: number } | null {
  const want = new Set(regions);
  const c = mesh.centroids;
  let sx = 0, sy = 0, sz = 0, sa = 0;
  for (let f = 0; f < mesh.faceCount; f++) {
    if (!want.has(patches.ids[f])) continue;
    const a = mesh.areas[f];
    sx += c[f * 3] * a; sy += c[f * 3 + 1] * a; sz += c[f * 3 + 2] * a; sa += a;
  }
  if (!sa) return null;
  const center: [number, number, number] = [sx / sa, sy / sa, sz / sa];
  let r2 = 0;
  for (let f = 0; f < mesh.faceCount; f++) {
    if (!want.has(patches.ids[f])) continue;
    r2 = Math.max(r2, (c[f * 3] - center[0]) ** 2 + (c[f * 3 + 1] - center[1]) ** 2 + (c[f * 3 + 2] - center[2]) ** 2);
  }
  const radius = Math.sqrt(r2) * 1.1;
  const mask = new Uint8Array(mesh.faceCount);
  for (let f = 0; f < mesh.faceCount; f++) {
    if ((c[f * 3] - center[0]) ** 2 + (c[f * 3 + 1] - center[1]) ** 2 + (c[f * 3 + 2] - center[2]) ** 2 <= radius * radius) mask[f] = 1;
  }
  return { mask, center, radius };
}
