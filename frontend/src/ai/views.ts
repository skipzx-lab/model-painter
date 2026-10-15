import * as THREE from 'three';
import type { PaintMesh } from '../mesh';
import type { Patches } from './patches';

export interface RenderedView {
  name: string;
  /** JPEG data URL: left = shaded model, right = numbered patch map. */
  image: string;
  /** Patch ids labelled in this view. */
  labelled: number[];
}

const VIEWS: { name: string; dir: [number, number, number] }[] = [
  { name: 'front (-Y), from above', dir: [0, -1, 0.8] },
  { name: 'back (+Y), from above', dir: [0, 1, 0.8] },
  { name: 'left (-X), from above', dir: [-1, 0, 0.8] },
  { name: 'right (+X), from above', dir: [1, 0, 0.8] },
  { name: 'top', dir: [0.15, -0.35, 1] },
  { name: 'underside', dir: [0.25, -0.45, -1] },
];

const ID_VERT = /* glsl */ `
attribute float pid;
varying float vPid;
void main() {
  vPid = pid;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;
const ID_FRAG = /* glsl */ `
varying float vPid;
void main() {
  float id = floor(vPid + 0.5) + 1.0;
  float r = mod(id, 256.0);
  float g = mod(floor(id / 256.0), 256.0);
  float b = floor(id / 65536.0);
  gl_FragColor = vec4(r / 255.0, g / 255.0, b / 255.0, 1.0);
}`;

/**
 * Render the model from several directions as image pairs for a vision
 * model: a shaded view (so it can recognise anatomy) beside the same view
 * split into numbered patches it can assign colours to.
 */
export function renderPatchViews(
  renderer: THREE.WebGLRenderer,
  mesh: PaintMesh,
  patches: Patches,
  opts: { size?: number; leftMode?: 'clay' | 'painted'; focus?: { center: THREE.Vector3; radius: number } } = {},
): RenderedView[] {
  const S = opts.size ?? 640;
  const F = mesh.faceCount;
  const display = mesh.geometry;

  // Per-corner patch id and a light grey shade per patch.
  const pid = new Float32Array(F * 3);
  const shade = new Uint8Array(F * 9);
  const tone = new Uint8Array(patches.count);
  for (let p = 0; p < patches.count; p++) tone[p] = 150 + ((p * 97) % 90);
  for (let f = 0; f < F; f++) {
    const p = patches.ids[f];
    pid[f * 3] = pid[f * 3 + 1] = pid[f * 3 + 2] = p; // -1 renders as background: unlabelled
    shade.fill(p >= 0 ? tone[p] : 245, f * 9, f * 9 + 9);
  }
  const idGeo = new THREE.BufferGeometry();
  idGeo.setAttribute('position', display.getAttribute('position'));
  idGeo.setAttribute('pid', new THREE.BufferAttribute(pid, 1));
  idGeo.setIndex(display.index);
  const shadeGeo = new THREE.BufferGeometry();
  shadeGeo.setAttribute('position', display.getAttribute('position'));
  shadeGeo.setAttribute('normal', display.getAttribute('normal'));
  shadeGeo.setAttribute('color', new THREE.BufferAttribute(shade, 3, true));
  shadeGeo.setIndex(display.index);

  const idMat = new THREE.ShaderMaterial({ vertexShader: ID_VERT, fragmentShader: ID_FRAG });
  const clayMat = new THREE.MeshStandardMaterial({ color: 0xb8b8b8, roughness: 0.7 });
  const paintMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.7 });
  const shadeMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.8 });

  const scene = new THREE.Scene();
  const obj = new THREE.Mesh<THREE.BufferGeometry, THREE.Material>(display, clayMat);
  scene.add(obj);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x404050, 1.4));
  const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 1000);
  cam.up.set(0, 0, 1);
  const light = new THREE.DirectionalLight(0xffffff, 2.0);
  light.position.set(0.4, 0.6, 1);
  cam.add(light);
  scene.add(cam);

  // Close-ups clip away anything between the camera and the focus area.
  const sphere = opts.focus ? new THREE.Sphere(opts.focus.center, opts.focus.radius) : display.boundingSphere!;
  const R = sphere.radius;
  cam.left = cam.bottom = -R * 1.05;
  cam.right = cam.top = R * 1.05;
  cam.near = opts.focus ? R * 1.6 : R * 0.01;
  cam.far = opts.focus ? R * 4.4 : R * 10;
  cam.updateProjectionMatrix();

  const visTarget = new THREE.WebGLRenderTarget(S, S, { samples: 4 });
  visTarget.texture.colorSpace = THREE.SRGBColorSpace;
  const idTarget = new THREE.WebGLRenderTarget(S, S, { minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });

  const prevTarget = renderer.getRenderTarget();
  const prevClear = renderer.getClearColor(new THREE.Color());
  const prevAlpha = renderer.getClearAlpha();
  const buf = new Uint8Array(S * S * 4);

  const readInto = (target: THREE.WebGLRenderTarget, out: Uint8ClampedArray) => {
    renderer.readRenderTargetPixels(target, 0, 0, S, S, buf);
    // WebGL rows are bottom-up.
    for (let y = 0; y < S; y++) out.set(buf.subarray((S - 1 - y) * S * 4, (S - y) * S * 4), y * S * 4);
  };

  const results: RenderedView[] = [];
  for (const v of VIEWS) {
    const dir = new THREE.Vector3(...v.dir).normalize();
    cam.position.copy(sphere.center).addScaledVector(dir, R * 3);
    cam.lookAt(sphere.center);
    cam.updateMatrixWorld();

    // 1. Patch id buffer.
    renderer.setRenderTarget(idTarget);
    renderer.setClearColor(0x000000, 1);
    renderer.clear();
    obj.geometry = idGeo; obj.material = idMat;
    renderer.render(scene, cam);
    const idPix = new Uint8ClampedArray(S * S * 4);
    readInto(idTarget, idPix);
    const ids = new Int32Array(S * S);
    for (let i = 0; i < S * S; i++) ids[i] = idPix[i * 4] + idPix[i * 4 + 1] * 256 + idPix[i * 4 + 2] * 65536 - 1;

    // 2. Shaded left panel.
    renderer.setRenderTarget(visTarget);
    renderer.setClearColor(0xffffff, 1);
    renderer.clear();
    obj.geometry = display; obj.material = opts.leftMode === 'painted' ? paintMat : clayMat;
    renderer.render(scene, cam);
    const left = new ImageData(S, S);
    readInto(visTarget, left.data);

    // 3. Grey patch panel.
    renderer.clear();
    obj.geometry = shadeGeo; obj.material = shadeMat;
    renderer.render(scene, cam);
    const right = new ImageData(S, S);
    readInto(visTarget, right.data);

    // Borders where the id changes.
    for (let y = 1; y < S - 1; y++) {
      for (let x = 1; x < S - 1; x++) {
        const i = y * S + x, id = ids[i];
        if (id < 0) continue;
        if (ids[i + 1] !== id || ids[i - 1] !== id || ids[i + S] !== id || ids[i - S] !== id) {
          right.data[i * 4] = 40; right.data[i * 4 + 1] = 40; right.data[i * 4 + 2] = 40;
        }
      }
    }

    // Label anchors: the pixel deepest inside each patch (chamfer distance to its border).
    const dist = new Float32Array(S * S);
    for (let i = 0; i < S * S; i++) dist[i] = ids[i] < 0 ? 0 : 1e9;
    const at = (x: number, y: number) => y * S + x;
    for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
      const i = at(x, y);
      if (!dist[i]) continue;
      const id = ids[i];
      let d = dist[i];
      if (x > 0) d = Math.min(d, ids[i - 1] === id ? dist[i - 1] + 1 : 1);
      else d = 1;
      if (y > 0) d = Math.min(d, ids[i - S] === id ? dist[i - S] + 1 : 1);
      else d = 1;
      dist[i] = d;
    }
    for (let y = S - 1; y >= 0; y--) for (let x = S - 1; x >= 0; x--) {
      const i = at(x, y);
      if (!dist[i]) continue;
      const id = ids[i];
      let d = dist[i];
      if (x < S - 1) d = Math.min(d, ids[i + 1] === id ? dist[i + 1] + 1 : 1);
      else d = 1;
      if (y < S - 1) d = Math.min(d, ids[i + S] === id ? dist[i + S] + 1 : 1);
      else d = 1;
      dist[i] = d;
    }
    const best = new Map<number, { i: number; d: number; n: number }>();
    for (let i = 0; i < S * S; i++) {
      const id = ids[i];
      if (id < 0) continue;
      const b = best.get(id);
      if (!b) best.set(id, { i, d: dist[i], n: 1 });
      else { b.n++; if (dist[i] > b.d) { b.d = dist[i]; b.i = i; } }
    }

    const canvas = document.createElement('canvas');
    canvas.width = S * 2; canvas.height = S;
    const ctx = canvas.getContext('2d')!;
    ctx.putImageData(left, 0, 0);
    ctx.putImageData(right, S, 0);
    ctx.strokeStyle = '#888'; ctx.beginPath(); ctx.moveTo(S, 0); ctx.lineTo(S, S); ctx.stroke();

    const labelled: number[] = [];
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (const [id, b] of best) {
      if (b.d < 5 || b.n < 60) continue; // too small to read here
      labelled.push(id);
      const x = (b.i % S) + S, y = Math.floor(b.i / S);
      const fs = Math.max(10, Math.min(18, b.d * 1.2));
      ctx.font = `bold ${fs}px sans-serif`;
      ctx.lineWidth = 3; ctx.strokeStyle = '#ffffff';
      ctx.strokeText(String(id), x, y);
      ctx.fillStyle = '#c00000';
      ctx.fillText(String(id), x, y);
    }
    ctx.font = 'bold 16px sans-serif'; ctx.textAlign = 'left'; ctx.fillStyle = '#222';
    ctx.fillText(v.name, 8, 16);
    results.push({ name: v.name, image: canvas.toDataURL('image/jpeg', 0.88), labelled });
  }

  renderer.setRenderTarget(prevTarget);
  renderer.setClearColor(prevClear, prevAlpha);
  visTarget.dispose(); idTarget.dispose();
  idGeo.dispose(); shadeGeo.dispose();
  idMat.dispose(); clayMat.dispose(); paintMat.dispose(); shadeMat.dispose();
  return results;
}

/** Downscale an uploaded reference image to a JPEG data URL (max 1024 px side). */
export async function prepareReferenceImage(file: Blob, maxSide = 1024): Promise<string> {
  const bmp = await createImageBitmap(file);
  const k = Math.min(1, maxSide / Math.max(bmp.width, bmp.height));
  const c = document.createElement('canvas');
  c.width = Math.round(bmp.width * k); c.height = Math.round(bmp.height * k);
  c.getContext('2d')!.drawImage(bmp, 0, 0, c.width, c.height);
  return c.toDataURL('image/jpeg', 0.9);
}

/** Dominant colours of an image via k-means on a pixel sample, most common first. */
export async function dominantColours(dataUrl: string, k = 6): Promise<string[]> {
  const img = new Image();
  img.src = dataUrl;
  await img.decode();
  const c = document.createElement('canvas');
  const s = Math.min(1, 160 / Math.max(img.width, img.height));
  c.width = Math.max(1, Math.round(img.width * s)); c.height = Math.max(1, Math.round(img.height * s));
  const ctx = c.getContext('2d')!;
  ctx.drawImage(img, 0, 0, c.width, c.height);
  const d = ctx.getImageData(0, 0, c.width, c.height).data;
  const px: number[][] = [];
  for (let i = 0; i < d.length; i += 4) px.push([d[i], d[i + 1], d[i + 2]]);
  let centers = Array.from({ length: k }, (_, j) => px[Math.floor(((j + 0.5) * px.length) / k)].slice());
  let assign = new Int32Array(px.length);
  for (let it = 0; it < 12; it++) {
    for (let i = 0; i < px.length; i++) {
      let b = 0, bd = Infinity;
      for (let j = 0; j < k; j++) {
        const dd = (px[i][0] - centers[j][0]) ** 2 + (px[i][1] - centers[j][1]) ** 2 + (px[i][2] - centers[j][2]) ** 2;
        if (dd < bd) { bd = dd; b = j; }
      }
      assign[i] = b;
    }
    const sum = centers.map(() => [0, 0, 0, 0]);
    for (let i = 0; i < px.length; i++) { const s2 = sum[assign[i]]; s2[0] += px[i][0]; s2[1] += px[i][1]; s2[2] += px[i][2]; s2[3]++; }
    centers = sum.map((s2, j) => (s2[3] ? [s2[0] / s2[3], s2[1] / s2[3], s2[2] / s2[3]] : centers[j]));
  }
  const counts = new Array(k).fill(0);
  for (const a of assign) counts[a]++;
  return centers
    .map((col, j) => ({ col, n: counts[j] }))
    .sort((a, b) => b.n - a.n)
    .map(({ col }) => '#' + col.map((v) => Math.round(v).toString(16).padStart(2, '0')).join(''));
}
