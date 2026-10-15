import type { PaintMesh } from '../mesh';

/** Smart-fill components of the whole mesh at one edge angle, with shape descriptors. */
interface Components {
  ids: Int32Array;
  area: Float64Array;
  /** Principal extents (sqrt of covariance eigenvalues), largest first. */
  extents: Float64Array; // 3 per component
}

const cache = new WeakMap<PaintMesh, Map<number, Components>>();

/** Eigenvalues of a symmetric 3x3 matrix (analytic), descending. */
function eig3(a: number, b: number, c: number, d: number, e: number, f: number): [number, number, number] {
  // | a d e |
  // | d b f |
  // | e f c |
  const p1 = d * d + e * e + f * f;
  if (p1 < 1e-18) return [a, b, c].sort((x, y) => y - x) as [number, number, number];
  const q = (a + b + c) / 3;
  const p2 = (a - q) ** 2 + (b - q) ** 2 + (c - q) ** 2 + 2 * p1;
  const p = Math.sqrt(p2 / 6);
  const B = [(a - q) / p, d / p, e / p, d / p, (b - q) / p, f / p, e / p, f / p, (c - q) / p];
  const detB = B[0] * (B[4] * B[8] - B[5] * B[7]) - B[1] * (B[3] * B[8] - B[5] * B[6]) + B[2] * (B[3] * B[7] - B[4] * B[6]);
  const r = Math.max(-1, Math.min(1, detB / 2));
  const phi = Math.acos(r) / 3;
  const e1 = q + 2 * p * Math.cos(phi);
  const e3 = q + 2 * p * Math.cos(phi + (2 * Math.PI) / 3);
  return [e1, 3 * q - e1 - e3, e3];
}

function components(mesh: PaintMesh, angleDeg: number): Components {
  let byAngle = cache.get(mesh);
  if (!byAngle) { byAngle = new Map(); cache.set(mesh, byAngle); }
  const key = Math.round(angleDeg);
  const hit = byAngle.get(key);
  if (hit) return hit;
  const n = mesh.faceNormals;
  const cosT = Math.cos((key * Math.PI) / 180);
  const out = labelComponents(mesh, () => true, (f, g) =>
    n[f * 3] * n[g * 3] + n[f * 3 + 1] * n[g * 3 + 1] + n[f * 3 + 2] * n[g * 3 + 2] >= cosT);
  byAngle.set(key, out);
  return out;
}

/**
 * Connected components over faces passing `include`, joined across edges
 * passing `cross`; faces outside get id -1. Also area and principal extents.
 */
function labelComponents(mesh: PaintMesh, include: (f: number) => boolean, cross: (f: number, g: number) => boolean): Components {
  const F = mesh.faceCount, adj = mesh.adjacency, c = mesh.centroids;
  const ids = new Int32Array(F).fill(-1);
  let count = 0;
  const stack: number[] = [];
  for (let s = 0; s < F; s++) {
    if (ids[s] >= 0 || !include(s)) continue;
    ids[s] = count;
    stack.push(s);
    while (stack.length) {
      const f = stack.pop()!;
      for (let k = 0; k < 3; k++) {
        const g = adj[f * 3 + k];
        if (g < 0 || ids[g] >= 0 || !include(g) || !cross(f, g)) continue;
        ids[g] = count;
        stack.push(g);
      }
    }
    count++;
  }
  // Area-weighted centroid covariance per component.
  const S = new Float64Array(count * 10); // area, sx, sy, sz, sxx, syy, szz, sxy, sxz, syz
  for (let f = 0; f < F; f++) {
    if (ids[f] < 0) continue;
    const o = ids[f] * 10, w = mesh.areas[f];
    const x = c[f * 3], y = c[f * 3 + 1], z = c[f * 3 + 2];
    S[o] += w; S[o + 1] += w * x; S[o + 2] += w * y; S[o + 3] += w * z;
    S[o + 4] += w * x * x; S[o + 5] += w * y * y; S[o + 6] += w * z * z;
    S[o + 7] += w * x * y; S[o + 8] += w * x * z; S[o + 9] += w * y * z;
  }
  const area = new Float64Array(count);
  const extents = new Float64Array(count * 3);
  for (let i = 0; i < count; i++) {
    const o = i * 10, w = S[o] || 1e-12;
    const mx = S[o + 1] / w, my = S[o + 2] / w, mz = S[o + 3] / w;
    const ev = eig3(
      S[o + 4] / w - mx * mx, S[o + 5] / w - my * my, S[o + 6] / w - mz * mz,
      S[o + 7] / w - mx * my, S[o + 8] / w - mx * mz, S[o + 9] / w - my * mz,
    );
    area[i] = S[o];
    for (let k = 0; k < 3; k++) extents[i * 3 + k] = Math.sqrt(Math.max(0, ev[k]));
  }
  return { ids, area, extents };
}

function matchLike(mesh: PaintMesh, comp: Components, face: number, tolerance: number): { faces: number[]; matches: number } {
  const ref = comp.ids[face];
  if (ref < 0) return { faces: [], matches: 0 };
  const ra = comp.area[ref];
  const re = [comp.extents[ref * 3], comp.extents[ref * 3 + 1], comp.extents[ref * 3 + 2]];
  const close = (a: number, b: number, tol: number) => Math.abs(Math.log((a + 1e-9) / (b + 1e-9))) <= tol;
  const match = new Uint8Array(comp.area.length);
  let matches = 0;
  for (let i = 0; i < comp.area.length; i++) {
    if (
      close(comp.area[i], ra, tolerance * 1.4) &&
      close(comp.extents[i * 3], re[0], tolerance) &&
      close(comp.extents[i * 3 + 1], re[1], tolerance * 1.2)
    ) { match[i] = 1; matches++; }
  }
  const faces: number[] = [];
  for (let f = 0; f < mesh.faceCount; f++) if (comp.ids[f] >= 0 && match[comp.ids[f]]) faces.push(f);
  return { faces, matches };
}

/**
 * Every thin part (thinner than `maxMm`) shaped like the thin part under
 * `face`: all the spikes along the back, all the claws, and so on.
 */
export function similarThinFaces(mesh: PaintMesh, thickness: Float32Array, face: number, maxMm: number, tolerance = 0.45): { faces: number[]; matches: number } {
  const comp = labelComponents(mesh, (f) => thickness[f] <= maxMm, () => true);
  return matchLike(mesh, comp, face, tolerance);
}

/**
 * Faces of every smart-fill component shaped like the one under `face`:
 * similar area and similar principal extents (so all the back spikes, or all
 * the claws, but not the body plates they sit on).
 */
export function similarFaces(mesh: PaintMesh, face: number, angleDeg: number, tolerance = 0.35): { faces: number[]; matches: number } {
  return matchLike(mesh, components(mesh, angleDeg), face, tolerance);
}
