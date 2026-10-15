import * as THREE from 'three';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { PaintMesh } from './mesh';
import { read3mf } from './reader3mf';

/** Merge every mesh in an object tree into one position-only geometry in world space. */
function flatten(root: THREE.Object3D): THREE.BufferGeometry {
  root.updateMatrixWorld(true);
  const parts: THREE.BufferGeometry[] = [];
  root.traverse((o) => {
    const m = o as THREE.Mesh;
    if (!m.isMesh || !m.geometry) return;
    let g = m.geometry.index ? m.geometry.toNonIndexed() : m.geometry.clone();
    g = new THREE.BufferGeometry().setAttribute('position', g.getAttribute('position'));
    g.applyMatrix4(m.matrixWorld);
    parts.push(g);
  });
  if (!parts.length) throw new Error('No meshes found in file');
  return parts.length === 1 ? parts[0] : mergeGeometries(parts)!;
}

export interface Loaded {
  mesh: PaintMesh;
  /** 1-based extruder per face when the file carried slicer paint. */
  extruders?: Uint8Array;
}

export async function loadModelFile(file: File): Promise<Loaded> {
  const ext = file.name.split('.').pop()!.toLowerCase();
  const buf = await file.arrayBuffer();
  if (ext === '3mf') {
    const r = read3mf(buf);
    const keep: number[] = [];
    const mesh = PaintMesh.fromIndexed(r.positions, r.faces, keep);
    return { mesh, extruders: r.extruders && Uint8Array.from(keep, (i) => r.extruders![i]) };
  }
  return { mesh: PaintMesh.fromGeometry(await loadGeometry(ext, buf)) };
}

async function loadGeometry(ext: string, buf: ArrayBuffer): Promise<THREE.BufferGeometry> {
  switch (ext) {
    case 'stl':
      return new STLLoader().parse(buf);
    case 'ply':
      return new PLYLoader().parse(buf);
    case 'obj':
      return flatten(new OBJLoader().parse(new TextDecoder().decode(buf)));
    case 'glb':
    case 'gltf': {
      const gltf = await new GLTFLoader().parseAsync(buf, '');
      // glTF is Y-up; printing files are Z-up.
      gltf.scene.rotation.x = Math.PI / 2;
      return flatten(gltf.scene);
    }
    default:
      throw new Error(`Unsupported file type .${ext}`);
  }
}
