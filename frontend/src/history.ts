import type { PaintMesh } from './mesh';

interface Step {
  faces: Uint32Array;
  before: Uint8Array;
  after: Uint8Array;
}

/** Undo/redo over per-face slot changes. Topology changes (refine) clear it. */
export class History {
  private undoStack: Step[] = [];
  private redoStack: Step[] = [];
  private pending = new Map<number, number>();
  private mesh: PaintMesh;

  constructor(mesh: PaintMesh) {
    this.mesh = mesh;
  }

  /** Paint faces with slot inside the current stroke, recording old values. */
  paint(faces: Iterable<number>, slot: number) {
    for (const f of faces) {
      if (this.mesh.slots[f] === slot) continue;
      if (!this.pending.has(f)) this.pending.set(f, this.mesh.slots[f]);
      this.mesh.setSlot(f, slot);
    }
    this.mesh.markColorsDirty();
  }

  /** Record changes that were already applied, as a map face -> old slot. */
  recordApplied(oldSlots: Map<number, number>) {
    for (const [f, s] of oldSlots) if (!this.pending.has(f)) this.pending.set(f, s);
    this.commit();
  }

  commit() {
    if (!this.pending.size) return;
    const faces = new Uint32Array(this.pending.size);
    const before = new Uint8Array(this.pending.size);
    const after = new Uint8Array(this.pending.size);
    let i = 0;
    for (const [f, s] of this.pending) {
      faces[i] = f; before[i] = s; after[i] = this.mesh.slots[f]; i++;
    }
    this.undoStack.push({ faces, before, after });
    if (this.undoStack.length > 200) this.undoStack.shift();
    this.redoStack = [];
    this.pending.clear();
  }

  private apply(step: Step, values: Uint8Array) {
    for (let i = 0; i < step.faces.length; i++) this.mesh.setSlot(step.faces[i], values[i]);
    this.mesh.markColorsDirty();
  }

  undo() {
    this.commit();
    const s = this.undoStack.pop();
    if (!s) return;
    this.apply(s, s.before);
    this.redoStack.push(s);
  }

  redo() {
    const s = this.redoStack.pop();
    if (!s) return;
    this.apply(s, s.after);
    this.undoStack.push(s);
  }
}
