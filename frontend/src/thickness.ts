import * as THREE from 'three';
import type { PaintMesh } from './mesh';

/**
 * Local wall thickness per face (shape diameter): distance from the face,
 * straight inward, to the opposite surface, smoothed over neighbours.
 * Spikes, claws, horns and teeth are thin; the body they grow from is thick,
 * so a thickness threshold finds where a smooth spike ends and the body
 * begins even when there's no crease there.
 */
const cache = new WeakMap<PaintMesh, Float32Array>();

export function hasThickness(mesh: PaintMesh) {
  return cache.has(mesh);
}

export async function computeThickness(mesh: PaintMesh, onProgress?: (frac: number) => void): Promise<Float32Array> {
  const hit = cache.get(mesh);
  if (hit) return hit;
  const F = mesh.faceCount, c = mesh.centroids, n = mesh.faceNormals;
  const size = mesh.boundingBox().getSize(new THREE.Vector3()).length();
  const eps = mesh.medianEdge() * 1e-2;
  const raw = new Float32Array(F);
  const ray = new THREE.Ray();
  // Three rays per face (inward normal plus two tilted ~25°), median distance:
  // a single ray near a spike's tip can slip out sideways and read too long.
  const tilt = Math.tan((25 * Math.PI) / 180);
  const dists = [0, 0, 0];
  const t1 = new THREE.Vector3(), t2 = new THREE.Vector3(), dir = new THREE.Vector3(), nrm = new THREE.Vector3();
  const chunk = 40000;
  for (let start = 0; start < F; start += chunk) {
    const end = Math.min(F, start + chunk);
    for (let f = start; f < end; f++) {
      nrm.set(-n[f * 3], -n[f * 3 + 1], -n[f * 3 + 2]);
      // Any two vectors perpendicular to the normal.
      t1.set(Math.abs(nrm.x) < 0.9 ? 1 : 0, Math.abs(nrm.x) < 0.9 ? 0 : 1, 0).cross(nrm).normalize();
      t2.copy(nrm).cross(t1);
      for (let k = 0; k < 3; k++) {
        dir.copy(nrm);
        if (k === 1) dir.addScaledVector(t1, tilt).normalize();
        if (k === 2) dir.addScaledVector(t2, -tilt).normalize();
        ray.origin.set(c[f * 3], c[f * 3 + 1], c[f * 3 + 2]).addScaledVector(dir, eps);
        ray.direction.copy(dir);
        const h = mesh.bvh.raycastFirst(ray, THREE.DoubleSide);
        dists[k] = h ? h.distance + eps : size;
      }
      dists.sort((a, b) => a - b);
      raw[f] = dists[1];
    }
    onProgress?.(end / F);
    await new Promise((r) => setTimeout(r, 0));
  }
  // Smooth over a few rings so borders follow the shape, not ray noise.
  let cur = raw;
  const adj = mesh.adjacency;
  for (let pass = 0; pass < 4; pass++) {
    const next = new Float32Array(F);
    for (let f = 0; f < F; f++) {
      let s = cur[f], k = 1;
      for (let j = 0; j < 3; j++) { const g = adj[f * 3 + j]; if (g >= 0) { s += cur[g]; k++; } }
      next[f] = s / k;
    }
    cur = next;
  }
  cache.set(mesh, cur);
  return cur;
}

/** Connected faces from `start` that are thinner than `maxMm`. */
export function thinFill(mesh: PaintMesh, thickness: Float32Array, start: number, maxMm: number): number[] {
  if (thickness[start] > maxMm) return [];
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
      if (g < 0 || seen[g] || thickness[g] > maxMm) continue;
      seen[g] = 1;
      stack.push(g);
    }
  }
  return out;
}
