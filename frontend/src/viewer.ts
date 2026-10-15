import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { acceleratedRaycast } from 'three-mesh-bvh';
import type { PaintMesh } from './mesh';

THREE.Mesh.prototype.raycast = acceleratedRaycast;

export class Viewer {
  renderer: THREE.WebGLRenderer;
  scene = new THREE.Scene();
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  object: THREE.Mesh | null = null;
  overlay: THREE.Mesh | null = null; // patch preview overlay
  cursor: THREE.Mesh;
  private raycaster = new THREE.Raycaster();
  private keyLight = new THREE.DirectionalLight(0xffffff, 2.2);
  private container: HTMLElement;

  constructor(container: HTMLElement) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(this.renderer.domElement);

    this.scene.background = new THREE.Color(0x1d2026);
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.1, 10000);
    this.camera.up.set(0, 0, 1); // printing models are Z-up
    this.camera.position.set(150, -200, 120);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x445066, 1.1));
    this.camera.add(this.keyLight);
    this.keyLight.position.set(0.5, 0.8, 0.3);
    this.scene.add(this.camera);
    this.raycaster.firstHitOnly = true;

    const ring = new THREE.RingGeometry(0.85, 1, 48);
    this.cursor = new THREE.Mesh(
      ring,
      new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.85, depthTest: false, side: THREE.DoubleSide }),
    );
    this.cursor.renderOrder = 10;
    this.cursor.visible = false;
    this.scene.add(this.cursor);

    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
    this.renderer.setAnimationLoop(() => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    });
  }

  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / Math.max(h, 1);
    this.camera.updateProjectionMatrix();
  }

  setMesh(mesh: PaintMesh) {
    if (this.object) this.scene.remove(this.object);
    this.object = new THREE.Mesh(
      mesh.geometry,
      new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.65, metalness: 0.05 }),
    );
    this.scene.add(this.object);
    this.frame(mesh);
  }

  /** Swap geometry after the mesh was rebuilt (refine) without moving the camera. */
  refreshGeometry(mesh: PaintMesh) {
    if (this.object) this.object.geometry = mesh.geometry;
  }

  frame(mesh: PaintMesh) {
    const box = mesh.boundingBox();
    const center = box.getCenter(new THREE.Vector3());
    const radius = box.getSize(new THREE.Vector3()).length() / 2;
    this.controls.target.copy(center);
    const dir = new THREE.Vector3(0.6, -1, 0.5).normalize();
    this.camera.position.copy(center).addScaledVector(dir, radius * 3.3);
    this.camera.near = radius / 100;
    this.camera.far = radius * 50;
    this.camera.updateProjectionMatrix();
  }

  pick(clientX: number, clientY: number): THREE.Intersection | null {
    if (!this.object) return null;
    const rect = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
    this.raycaster.setFromCamera(ndc, this.camera);
    return this.raycaster.intersectObject(this.object, false)[0] ?? null;
  }

  showCursor(hit: THREE.Intersection | null, radius: number) {
    if (!hit || !hit.face) { this.cursor.visible = false; return; }
    this.cursor.visible = true;
    this.cursor.position.copy(hit.point);
    this.cursor.scale.setScalar(radius);
    this.cursor.lookAt(hit.point.clone().add(hit.face.normal));
  }

  viewDirection(): [number, number, number] {
    const d = new THREE.Vector3();
    this.camera.getWorldDirection(d);
    return [d.x, d.y, d.z];
  }
}
