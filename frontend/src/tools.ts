import type { PaintMesh } from './mesh';

/** Faces whose centroid lies within `radius` of `point`, grown across connected faces from `start`. */
export function brushFaces(
  mesh: PaintMesh,
  start: number,
  point: [number, number, number],
  radius: number,
  viewDir?: [number, number, number],
): number[] {
  const r2 = radius * radius;
  const c = mesh.centroids, n = mesh.faceNormals, adj = mesh.adjacency;
  const seen = new Set<number>([start]);
  const out: number[] = [];
  const stack = [start];
  while (stack.length) {
    const f = stack.pop()!;
    const dx = c[f * 3] - point[0], dy = c[f * 3 + 1] - point[1], dz = c[f * 3 + 2] - point[2];
    if (f !== start && dx * dx + dy * dy + dz * dz > r2) continue;
    // Front-facing only: skip faces pointing away from the camera (e.g. the far side of a wing).
    if (viewDir && f !== start) {
      const d = n[f * 3] * viewDir[0] + n[f * 3 + 1] * viewDir[1] + n[f * 3 + 2] * viewDir[2];
      if (d > 0.2) continue;
    }
    out.push(f);
    for (let k = 0; k < 3; k++) {
      const g = adj[f * 3 + k];
      if (g >= 0 && !seen.has(g)) { seen.add(g); stack.push(g); }
    }
  }
  return out;
}

/** Flood fill bounded by creases sharper than `angleDeg`. */
export function creaseFill(mesh: PaintMesh, start: number, angleDeg: number): number[] {
  const cosT = Math.cos((angleDeg * Math.PI) / 180);
  const n = mesh.faceNormals, adj = mesh.adjacency;
  const seen = new Uint8Array(mesh.faceCount);
  seen[start] = 1;
  const out: number[] = [];
  const stack = [start];
  while (stack.length) {
    const f = stack.pop()!;
    out.push(f);
    for (let k = 0; k < 3; k++) {
      const g = adj[f * 3 + k];
      if (g < 0 || seen[g]) continue;
      const d = n[f * 3] * n[g * 3] + n[f * 3 + 1] * n[g * 3 + 1] + n[f * 3 + 2] * n[g * 3 + 2];
      if (d < cosT) continue;
      seen[g] = 1;
      stack.push(g);
    }
  }
  return out;
}

/** Connected faces sharing the start face's current slot (classic bucket fill). */
export function bucketFill(mesh: PaintMesh, start: number): number[] {
  const target = mesh.slots[start];
  const adj = mesh.adjacency;
  const seen = new Uint8Array(mesh.faceCount);
  seen[start] = 1;
  const out: number[] = [];
  const stack = [start];
  while (stack.length) {
    const f = stack.pop()!;
    out.push(f);
    for (let k = 0; k < 3; k++) {
      const g = adj[f * 3 + k];
      if (g < 0 || seen[g] || mesh.slots[g] !== target) continue;
      seen[g] = 1;
      stack.push(g);
    }
  }
  return out;
}

/**
 * Remove colour islands smaller than `minArea` (mm²) by merging them into
 * the neighbouring colour they share the longest border with. Cleans up
 * speckle left by AI painting that the printer can't reproduce anyway.
 */
export function despeckle(mesh: PaintMesh, minArea: number): Map<number, number> {
  const F = mesh.faceCount, adj = mesh.adjacency, slots = mesh.slots;
  const changes = new Map<number, number>();
  const comp = new Int32Array(F).fill(-1);
  let compId = 0;
  for (let s = 0; s < F; s++) {
    if (comp[s] >= 0) continue;
    const members: number[] = [];
    let area = 0;
    const stack = [s];
    comp[s] = compId;
    const borderCount = new Map<number, number>();
    while (stack.length) {
      const f = stack.pop()!;
      members.push(f);
      area += mesh.areas[f];
      for (let k = 0; k < 3; k++) {
        const g = adj[f * 3 + k];
        if (g < 0) continue;
        if (slots[g] !== slots[s]) { borderCount.set(slots[g], (borderCount.get(slots[g]) ?? 0) + 1); continue; }
        if (comp[g] >= 0) continue;
        comp[g] = compId;
        stack.push(g);
      }
    }
    compId++;
    if (area >= minArea || !borderCount.size) continue;
    let best = -1, bestN = -1;
    for (const [slot, cnt] of borderCount) if (cnt > bestN) { best = slot; bestN = cnt; }
    for (const f of members) { changes.set(f, slots[f]); slots[f] = best; }
  }
  return changes;
}

export interface RangeFilter {
  /** Height above the model's lowest point, mm. */
  zMin?: number | null;
  zMax?: number | null;
  /** Horizontal distance from `center`, mm. */
  rMin?: number | null;
  rMax?: number | null;
  center?: [number, number];
}

/**
 * Faces inside a height band and/or a horizontal distance band, like a
 * slicer's height-range painting. `faces` limits it to one part.
 */
export function rangeFaces(mesh: PaintMesh, filter: RangeFilter, faces?: ArrayLike<number>): number[] {
  const c = mesh.centroids;
  const base = mesh.boundingBox().min.z;
  const zMin = filter.zMin ?? -Infinity, zMax = filter.zMax ?? Infinity;
  const rMin = filter.rMin ?? 0, rMax = filter.rMax ?? Infinity;
  const [cx, cy] = filter.center ?? [0, 0];
  const useR = filter.rMin != null || filter.rMax != null;
  const out: number[] = [];
  const n = faces ? faces.length : mesh.faceCount;
  for (let i = 0; i < n; i++) {
    const f = faces ? faces[i] : i;
    const z = c[f * 3 + 2] - base;
    if (z < zMin || z > zMax) continue;
    if (useR) {
      const r = Math.hypot(c[f * 3] - cx, c[f * 3 + 1] - cy);
      if (r < rMin || r > rMax) continue;
    }
    out.push(f);
  }
  return out;
}

/**
 * Clean stair-stepped colour borders: a face whose neighbours mostly carry
 * another colour takes that colour. Only faces in `scope` (or all) change.
 * Returns face -> old slot for undo.
 */
export function smoothEdges(mesh: PaintMesh, passes = 2, scope?: ArrayLike<number>): Map<number, number> {
  const adj = mesh.adjacency, slots = mesh.slots;
  const changes = new Map<number, number>();
  const list = scope ?? Array.from({ length: mesh.faceCount }, (_, i) => i);
  for (let p = 0; p < passes; p++) {
    const flips: [number, number][] = [];
    for (let i = 0; i < list.length; i++) {
      const f = list[i];
      const a = adj[f * 3], b = adj[f * 3 + 1], c = adj[f * 3 + 2];
      if (a < 0 || b < 0 || c < 0) continue;
      const sa = slots[a], sb = slots[b], sc = slots[c], own = slots[f];
      let maj = -1;
      if (sa === sb || sa === sc) maj = sa; else if (sb === sc) maj = sb;
      if (maj >= 0 && maj !== own) flips.push([f, maj]);
    }
    if (!flips.length) break;
    for (const [f, s] of flips) {
      if (!changes.has(f)) changes.set(f, slots[f]);
      slots[f] = s;
    }
  }
  return changes;
}
