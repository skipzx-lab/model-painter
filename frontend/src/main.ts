import './style.css';
import * as THREE from 'three';
import { Viewer } from './viewer';
import type { PaintMesh } from './mesh';
import { History } from './history';
import { loadModelFile } from './loader';
import { brushFaces, bucketFill, creaseFill, despeckle, smoothEdges } from './tools';
import { export3mf } from './export3mf';
import { buildPatches, focusArea, resolveAssignments, type Patches } from './ai/patches';
import { dominantColours, prepareReferenceImage, renderPatchViews } from './ai/views';
import { listProviders, requestPaint, type Filament } from './ai/client';
import { PaintAgent } from './ai/agent';
import { computeThickness, hasThickness, thinFill } from './thickness';
import { similarThinFaces } from './ai/similar';
import { copyPaint, segmentRuleFaces, shells, similarShells } from './transfer';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

type Tool = 'orbit' | 'brush' | 'crease' | 'bucket' | 'patch' | 'thin' | 'copy' | 'rule';

const state = {
  mesh: null as PaintMesh | null,
  history: null as History | null,
  fileName: 'model',
  filaments: [
    { name: 'Black', hex: '#1b1b1d' },
    { name: 'Poison green', hex: '#4cc22a' },
    { name: 'Yellow', hex: '#f2c418' },
    { name: 'Bone white', hex: '#e6dcc3' },
  ] as Filament[],
  active: 0,
  tool: 'brush' as Tool,
  patches: null as Patches | null,
  patchKey: '',
  meshVersion: 0,
  showingPatches: false,
  reference: null as string | null,
  lastPatchSlots: null as Int32Array | null,
  busy: false,
};

const viewer = new Viewer($('view'));

// ---------- filaments ----------

function renderFilaments() {
  const list = $('filaments');
  list.innerHTML = '';
  state.filaments.forEach((f, i) => {
    const li = document.createElement('li');
    li.className = i === state.active ? 'active' : '';
    li.innerHTML = `<span class="idx">${i + 1}</span><input type="color" value="${f.hex}"><input type="text" value="${f.name}"><button class="del" title="Remove">×</button>`;
    const [color, name] = li.querySelectorAll('input');
    li.addEventListener('click', (e) => {
      if ((e.target as HTMLElement).tagName === 'INPUT') return;
      state.active = i;
      renderFilaments();
    });
    color.addEventListener('input', () => { f.hex = color.value; applyPalette(); });
    name.addEventListener('change', () => { f.name = name.value.trim() || `Filament ${i + 1}`; });
    li.querySelector('.del')!.addEventListener('click', (e) => { e.stopPropagation(); removeFilament(i); });
    list.appendChild(li);
  });
  ($('add-filament') as HTMLButtonElement).disabled = state.filaments.length >= 16;
}

function applyPalette() {
  if (!state.mesh) return;
  state.mesh.setPalette(state.filaments.map((f) => f.hex));
  state.showingPatches = false;
}

function removeFilament(i: number) {
  if (state.filaments.length <= 1) return;
  state.filaments.splice(i, 1);
  if (state.mesh) {
    const s = state.mesh.slots;
    for (let f = 0; f < s.length; f++) {
      if (s[f] === i) s[f] = 0;
      else if (s[f] > i) s[f]--;
    }
    state.history = new History(state.mesh); // slot numbers changed; old undo steps no longer apply
  }
  state.active = Math.min(state.active, state.filaments.length - 1);
  renderFilaments();
  applyPalette();
}

$('add-filament').addEventListener('click', () => addFilament('#888888', `Filament ${state.filaments.length + 1}`));

function addFilament(hex: string, name: string) {
  if (state.filaments.length >= 16) return;
  state.filaments.push({ name, hex });
  state.active = state.filaments.length - 1;
  renderFilaments();
  applyPalette();
}

$('paint-all').addEventListener('click', () => {
  if (!state.mesh || !state.history) return;
  exitPatchView();
  state.history.paint(Array.from({ length: state.mesh.faceCount }, (_, i) => i), state.active);
  state.history.commit();
});

// ---------- loading ----------

async function openFile(file: File) {
  setStatus(`Loading ${file.name}…`);
  try {
    const { mesh, extruders } = await loadModelFile(file);
    if (extruders) {
      // Bring in paint from a slicer project, adding filament slots as needed.
      let maxExt = 0;
      for (let f = 0; f < extruders.length; f++) maxExt = Math.max(maxExt, extruders[f]);
      while (state.filaments.length < Math.min(16, maxExt)) state.filaments.push({ name: `Filament ${state.filaments.length + 1}`, hex: '#888888' });
      for (let f = 0; f < extruders.length; f++) mesh.slots[f] = Math.min(15, Math.max(0, extruders[f] - 1));
      renderFilaments();
    }
    state.mesh = mesh;
    state.history = new History(mesh);
    state.fileName = file.name.replace(/\.[^.]+$/, '');
    state.patches = null;
    state.lastPatchSlots = null;
    state.meshVersion++;
    state.showingPatches = false;
    mesh.setPalette(state.filaments.map((f) => f.hex));
    viewer.setMesh(mesh);
    $('drop-hint').hidden = true;
    ($('export') as HTMLButtonElement).disabled = false;
    ($('ai-paint') as HTMLButtonElement).disabled = false;
    ($('chat-send') as HTMLButtonElement).disabled = false;
    resetChat('New model loaded. Describe how to paint it.');
    ($('ai-refine') as HTMLButtonElement).disabled = true;
    const edge = mesh.medianEdge();
    const size = mesh.boundingBox().getSize(new THREE.Vector3());
    ($('brush-size') as HTMLInputElement).value = String(Math.max(0.2, +(size.length() / 60).toFixed(1)));
    ($('refine-edge') as HTMLInputElement).value = String(+Math.max(0.2, Math.min(edge, size.length() / 150)).toFixed(2));
    updateLabels();
    updateStats();
    setStatus('');
  } catch (err) {
    console.error(err);
    setStatus(`Could not load: ${(err as Error).message}`, true);
  }
}

function updateStats() {
  const m = state.mesh;
  if (!m) return;
  const s = m.boundingBox().getSize(new THREE.Vector3());
  $('stats').textContent = `${state.fileName}: ${m.faceCount.toLocaleString()} triangles · ${s.x.toFixed(1)} × ${s.y.toFixed(1)} × ${s.z.toFixed(1)} mm`;
}

$('file').addEventListener('change', (e) => {
  const f = (e.target as HTMLInputElement).files?.[0];
  if (f) openFile(f);
});

window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', (e) => {
  e.preventDefault();
  const f = e.dataTransfer?.files?.[0];
  if (!f) return;
  if (f.type.startsWith('image/')) setReference(f);
  else openFile(f);
});

// ---------- export ----------

$('export').addEventListener('click', () => {
  if (!state.mesh) return;
  const data = export3mf(state.mesh, { name: state.fileName, palette: state.filaments.map((f) => f.hex) });
  const blob = new Blob([data as BlobPart], { type: 'model/3mf' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${state.fileName}-painted.3mf`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
});

// ---------- tools ----------

function setTool(t: Tool) {
  state.tool = t;
  document.querySelectorAll<HTMLButtonElement>('#tools button').forEach((b) => b.classList.toggle('active', b.dataset.tool === t));
  const c = viewer.controls;
  c.mouseButtons = t === 'orbit'
    ? { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN }
    : { LEFT: null as unknown as THREE.MOUSE, MIDDLE: THREE.MOUSE.PAN, RIGHT: THREE.MOUSE.ROTATE };
  if (t !== 'brush') viewer.cursor.visible = false;
}
document.querySelectorAll<HTMLButtonElement>('#tools button').forEach((b) => b.addEventListener('click', () => setTool(b.dataset.tool as Tool)));

function updateLabels() {
  $('brush-size-v').textContent = ($('brush-size') as HTMLInputElement).value;
  $('crease-angle-v').textContent = ($('crease-angle') as HTMLInputElement).value;
  $('patch-count-v').textContent = ($('patch-count') as HTMLInputElement).value;
  $('thin-max-v').textContent = ($('thin-max') as HTMLInputElement).value;
  $('rule-height-v').textContent = ($('rule-height') as HTMLInputElement).value;
  $('rule-reach-v').textContent = ($('rule-reach') as HTMLInputElement).value;
  $('rule-stripe-v').textContent = ($('rule-stripe') as HTMLInputElement).value;
}
['brush-size', 'crease-angle', 'patch-count', 'thin-max', 'rule-height', 'rule-reach', 'rule-stripe'].forEach((id) => $(id).addEventListener('input', updateLabels));

const brushRadius = () => +($('brush-size') as HTMLInputElement).value;

function exitPatchView() {
  if (state.showingPatches && state.mesh) {
    state.mesh.refreshColors();
    state.showingPatches = false;
    $('show-patches').textContent = 'Show regions';
  }
}

let painting = false;
const canvas = viewer.renderer.domElement;

function paintAt(e: PointerEvent) {
  const m = state.mesh, h = state.history;
  if (!m || !h) return;
  const hit = viewer.pick(e.clientX, e.clientY);
  if (state.tool === 'brush') viewer.showCursor(hit, brushRadius());
  if (!hit) return;
  const face = m.faceFromHit(hit);
  const p: [number, number, number] = [hit.point.x, hit.point.y, hit.point.z];
  const front = ($('front-only') as HTMLInputElement).checked ? viewer.viewDirection() : undefined;
  exitPatchView();
  switch (state.tool) {
    case 'brush': h.paint(brushFaces(m, face, p, brushRadius(), front), state.active); break;
    case 'crease': h.paint(creaseFill(m, face, +($('crease-angle') as HTMLInputElement).value), state.active); break;
    case 'bucket': h.paint(bucketFill(m, face), state.active); break;
    case 'rule': {
      const { shellOf, shells: list } = shells(m);
      const parts = similarShells(m, shellOf[face]);
      const hf = +($('rule-height') as HTMLInputElement).value / 100, rf = +($('rule-reach') as HTMLInputElement).value / 100;
      h.paint(segmentRuleFaces(m, parts, hf, rf, +($('rule-stripe') as HTMLInputElement).value / 100), state.active);
      const scope: number[] = [];
      for (const p of parts) for (const f of list[p].faces) scope.push(f);
      const sm = smoothEdges(m, 2, scope);
      m.refreshColors();
      h.recordApplied(sm);
      setStatus(`Spike fill applied to ${parts.length} matching part${parts.length === 1 ? '' : 's'}.`);
      break;
    }
    case 'copy': {
      const { shellOf } = shells(m);
      const r = copyPaint(m, shellOf[face]);
      m.refreshColors();
      h.recordApplied(r.faces);
      setStatus(`Copied this part's paint onto ${r.copied.length} matching parts${r.skipped.length ? `, skipped ${r.skipped.length} that didn't fit` : ''}.`);
      break;
    }
    case 'thin': {
      if (!hasThickness(m)) { prepareThickness(); return; }
      computeThickness(m).then((th) => {
        const max = +($('thin-max') as HTMLInputElement).value;
        const faces = ($('thin-similar') as HTMLInputElement).checked
          ? similarThinFaces(m, th, face, max).faces
          : thinFill(m, th, face, max);
        if (!faces.length) setStatus(`That spot is ${th[face].toFixed(1)} mm thick, above the ${max} mm limit.`);
        h.paint(faces, state.active);
        h.commit();
      });
      break;
    }
    case 'patch': {
      const patches = ensurePatches();
      const id = patches.ids[face];
      const faces: number[] = [];
      for (let f = 0; f < m.faceCount; f++) if (patches.ids[f] === id) faces.push(f);
      h.paint(faces, state.active);
      break;
    }
  }
}

canvas.addEventListener('pointerdown', (e) => {
  if (e.button !== 0 || state.tool === 'orbit' || !state.mesh) return;
  painting = state.tool === 'brush';
  canvas.setPointerCapture(e.pointerId);
  paintAt(e);
});
canvas.addEventListener('pointermove', (e) => {
  if (state.tool === 'brush' && state.mesh) {
    if (painting) paintAt(e);
    else viewer.showCursor(viewer.pick(e.clientX, e.clientY), brushRadius());
  }
});
const endStroke = () => { painting = false; state.history?.commit(); };
canvas.addEventListener('pointerup', endStroke);
canvas.addEventListener('pointercancel', endStroke);
canvas.addEventListener('pointerleave', () => { viewer.cursor.visible = false; });
canvas.addEventListener('contextmenu', (e) => e.preventDefault());

window.addEventListener('keydown', (e) => {
  const tag = (e.target as HTMLElement).tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
  if (e.ctrlKey && e.key.toLowerCase() === 'z') { exitPatchView(); state.history?.undo(); e.preventDefault(); return; }
  if (e.ctrlKey && e.key.toLowerCase() === 'y') { exitPatchView(); state.history?.redo(); e.preventDefault(); return; }
  const n = parseInt(e.key, 10);
  if (n >= 1 && n <= state.filaments.length) { state.active = n - 1; renderFilaments(); return; }
  const tools: Record<string, Tool> = { o: 'orbit', b: 'brush', f: 'crease', g: 'bucket', p: 'patch', t: 'thin', c: 'copy', s: 'rule' };
  if (tools[e.key]) setTool(tools[e.key]);
});

async function prepareThickness() {
  const m = state.mesh;
  if (!m) return null;
  if (!hasThickness(m)) setStatus('Measuring thickness (one-off, a few seconds)…');
  const th = await computeThickness(m, (f) => setStatus(`Measuring thickness… ${Math.round(f * 100)}%`));
  setStatus('');
  return th;
}

$('thin-preview').addEventListener('click', async () => {
  const m = state.mesh;
  if (!m) return;
  if (state.showingPatches) { exitPatchView(); return; }
  const th = await prepareThickness();
  if (!th) return;
  const max = +($('thin-max') as HTMLInputElement).value;
  const arr = m.colorAttr.array as Uint8Array;
  for (let f = 0; f < m.faceCount; f++) {
    const thin = th[f] <= max;
    for (let k = 0; k < 9; k += 3) {
      arr[f * 9 + k] = thin ? 255 : 60; arr[f * 9 + k + 1] = thin ? 40 : 60; arr[f * 9 + k + 2] = thin ? 200 : 66;
    }
  }
  m.markColorsDirty();
  state.showingPatches = true;
  $('show-patches').textContent = 'Hide regions';
});

// ---------- mesh ops ----------

$('refine').addEventListener('click', () => {
  const m = state.mesh;
  if (!m) return;
  const edge = +($('refine-edge') as HTMLInputElement).value;
  if (!(edge > 0)) return;
  exitPatchView();
  setStatus('Subdividing…');
  setTimeout(() => {
    const added = m.refine(edge);
    m.refreshColors();
    viewer.refreshGeometry(m);
    state.history = new History(m);
    state.patches = null;
    state.lastPatchSlots = null;
    state.meshVersion++;
    resetChat('Mesh was subdivided, so the chat restarted.');
    updateStats();
    setStatus(added ? `Added ${added.toLocaleString()} triangles.` : 'Already fine enough.');
  }, 20);
});

$('despeckle').addEventListener('click', () => {
  const m = state.mesh, h = state.history;
  if (!m || !h) return;
  exitPatchView();
  const changed = despeckle(m, +($('speck-area') as HTMLInputElement).value);
  m.refreshColors();
  h.recordApplied(changed);
  setStatus(`Cleaned ${changed.size.toLocaleString()} triangles.`);
});

// ---------- AI ----------

function ensurePatches(): Patches {
  const m = state.mesh!;
  const count = +($('patch-count') as HTMLInputElement).value;
  const key = `${state.meshVersion}:${count}`;
  if (!state.patches || state.patchKey !== key) {
    state.patches = buildPatches(m, count);
    state.patchKey = key;
    state.lastPatchSlots = null;
  }
  return state.patches;
}

$('show-patches').addEventListener('click', () => {
  const m = state.mesh;
  if (!m) return;
  if (state.showingPatches) { exitPatchView(); return; }
  const p = ensurePatches();
  const arr = m.colorAttr.array as Uint8Array;
  const col = new THREE.Color();
  for (let f = 0; f < m.faceCount; f++) {
    col.setHSL(((p.ids[f] * 0.618034) % 1), 0.55, 0.55);
    for (let k = 0; k < 9; k += 3) { arr[f * 9 + k] = col.r * 255; arr[f * 9 + k + 1] = col.g * 255; arr[f * 9 + k + 2] = col.b * 255; }
  }
  m.markColorsDirty();
  state.showingPatches = true;
  $('show-patches').textContent = 'Hide regions';
});

async function setReference(file: File) {
  state.reference = await prepareReferenceImage(file);
  const img = $('ref-img') as HTMLImageElement;
  img.src = state.reference;
  img.hidden = false;
  $('ref-hint').hidden = true;
  $('ref-tools').hidden = false;
}
$('ref-drop').addEventListener('click', () => $('ref-file').click());
$('ref-file').addEventListener('change', (e) => {
  const f = (e.target as HTMLInputElement).files?.[0];
  if (f) setReference(f);
});
$('ref-drop').addEventListener('dragover', (e) => { e.preventDefault(); $('ref-drop').classList.add('over'); });
$('ref-drop').addEventListener('dragleave', () => $('ref-drop').classList.remove('over'));
$('ref-drop').addEventListener('drop', (e) => {
  e.preventDefault(); e.stopPropagation();
  $('ref-drop').classList.remove('over');
  const f = e.dataTransfer?.files?.[0];
  if (f) setReference(f);
});
$('ref-clear').addEventListener('click', () => {
  state.reference = null;
  ($('ref-img') as HTMLImageElement).hidden = true;
  $('ref-hint').hidden = false;
  $('ref-tools').hidden = true;
  $('ref-swatches').innerHTML = '';
});
$('ref-suggest').addEventListener('click', async () => {
  if (!state.reference) return;
  const cols = await dominantColours(state.reference, 8);
  const box = $('ref-swatches');
  box.innerHTML = '<div class="muted small" style="width:100%">Click a colour to add it as a filament:</div>';
  for (const c of cols) {
    const b = document.createElement('button');
    b.style.background = c;
    b.title = c;
    b.addEventListener('click', () => addFilament(c, c));
    box.appendChild(b);
  }
});

function setStatus(msg: string, error = false) {
  const s = $('ai-status');
  s.textContent = msg;
  s.classList.toggle('error', error);
}

async function loadProviders() {
  const sel = $('provider') as HTMLSelectElement;
  try {
    const providers = await listProviders();
    sel.innerHTML = providers
      .map((p) => `<option value="${p.id}" ${p.configured ? '' : 'disabled'}>${p.label} — ${p.model}${p.configured ? '' : ' (not configured)'}</option>`)
      .join('');
    const first = providers.find((p) => p.configured);
    if (first) sel.value = first.id;
  } catch (err) {
    sel.innerHTML = '<option disabled>Backend offline</option>';
    console.warn(err);
  }
}

function slotForColour(name: string): number {
  const n = name.trim().toLowerCase();
  let i = state.filaments.findIndex((f) => f.name.trim().toLowerCase() === n);
  if (i < 0) i = state.filaments.findIndex((f) => f.hex.toLowerCase() === n);
  return i;
}

/** Most common slot per patch, for telling the model what is painted now. */
function currentPatchColours(patches: Patches): { patch: number; color: string }[] {
  const m = state.mesh!;
  const votes = Array.from({ length: patches.count }, () => new Float64Array(state.filaments.length));
  for (let f = 0; f < m.faceCount; f++) {
    const p = patches.ids[f];
    if (p >= 0) votes[p][m.slots[f]] += m.areas[f];
  }
  return votes.map((v, patch) => {
    let best = 0;
    for (let i = 1; i < v.length; i++) if (v[i] > v[best]) best = i;
    return { patch, color: state.filaments[best].name };
  });
}

function showViews(views: { name: string; image: string }[], append = false) {
  if (!append) $('ai-views').innerHTML = '';
  for (const v of views) {
    const img = document.createElement('img');
    img.src = v.image; img.title = v.name;
    img.addEventListener('click', () => window.open()?.document.write(`<img src="${v.image}">`));
    $('ai-views').appendChild(img);
  }
  $('ai-details').hidden = false;
}

/** Paint faces from a patch -> slot table; slot -1 leaves faces untouched. */
function applyPatchSlots(patches: Patches, patchSlots: Int32Array) {
  const m = state.mesh!, h = state.history!;
  const faces = new Map<number, number[]>();
  for (let f = 0; f < m.faceCount; f++) {
    const p = patches.ids[f];
    if (p < 0) continue;
    const s = patchSlots[p];
    if (s < 0) continue;
    if (!faces.has(s)) faces.set(s, []);
    faces.get(s)!.push(f);
  }
  for (const [s, list] of faces) h.paint(list, s);
  h.commit();
}

const tick = () => new Promise((r) => setTimeout(r, 20));

async function runAi(mode: 'initial' | 'refine') {
  const m = state.mesh, h = state.history;
  if (!m || !h || state.busy) return;
  const description = ($('chat-input') as HTMLTextAreaElement).value.trim();
  if (!description && !state.reference) { setStatus('Add a description or a reference image first.', true); return; }
  state.busy = true;
  ($('ai-paint') as HTMLButtonElement).disabled = true;
  ($('ai-refine') as HTMLButtonElement).disabled = true;
  const t0 = performance.now();
  const provider = ($('provider') as HTMLSelectElement).value;
  const base = { provider, description, filaments: state.filaments, reference_image: state.reference };
  try {
    setStatus('Splitting model into regions…');
    await tick();
    const patches = ensurePatches();
    exitPatchView();
    setStatus('Rendering views…');
    await tick();
    const views = renderPatchViews(viewer.renderer, m, patches, { leftMode: mode === 'refine' ? 'painted' : 'clay' });

    setStatus(`Pass 1: asking the model (${views.length} views, ${patches.count} regions)…`);
    const res = await requestPaint({
      ...base,
      views,
      patch_count: patches.count,
      mode,
      current: mode === 'refine' ? currentPatchColours(patches) : undefined,
    });

    const assigned = new Map<number, number>();
    for (const a of res.assignments) {
      const s = slotForColour(a.color);
      if (s >= 0) assigned.set(a.patch, s);
    }
    const patchSlots = resolveAssignments(patches, assigned, Math.max(0, slotForColour(res.default_color)));
    state.lastPatchSlots = patchSlots;
    applyPatchSlots(patches, patchSlots);
    $('ai-plan').textContent = res.plan;
    showViews(views);
    let summary = `Pass 1 painted ${assigned.size}/${patches.count} regions`;

    // Pass 2: close-up on the areas the model flagged (eyes, teeth...).
    const flagged = (res.detail_regions ?? []).filter((r) => r >= 0 && r < patches.count);
    if (mode === 'initial' && flagged.length && ($('detail-pass') as HTMLInputElement).checked) {
      const area = focusArea(m, patches, flagged);
      if (area) {
        setStatus(`${summary}. Pass 2: re-cutting the detail area finer…`);
        await tick();
        const fine = buildPatches(m, +($('detail-count') as HTMLInputElement).value, area.mask);
        const close = renderPatchViews(viewer.renderer, m, fine, {
          leftMode: 'painted',
          focus: { center: new THREE.Vector3(...area.center), radius: area.radius },
        }).map((v) => ({ ...v, name: `close-up ${v.name}` }));
        setStatus(`${summary}. Pass 2: asking about details (${fine.count} small regions)…`);
        const det = await requestPaint({
          ...base,
          views: close,
          patch_count: fine.count,
          mode: 'detail',
          current: currentPatchColours(fine),
        });
        const fineSlots = new Int32Array(fine.count).fill(-1); // unassigned keeps pass-1 paint
        for (const a of det.assignments) {
          const s = slotForColour(a.color);
          if (s >= 0 && a.patch < fine.count) fineSlots[a.patch] = s;
        }
        applyPatchSlots(fine, fineSlots);
        $('ai-plan').textContent = `${res.plan}\n\nDetail pass: ${det.plan}`;
        showViews(close, true);
        summary += `; pass 2 repainted ${det.assignments.length}/${fine.count} detail regions`;
      }
    }
    const secs = ((performance.now() - t0) / 1000).toFixed(0);
    setStatus(`${summary} · ${res.model} · ${secs}s`);
  } catch (err) {
    console.error(err);
    setStatus(`AI paint failed: ${(err as Error).message}`, true);
  } finally {
    state.busy = false;
    ($('ai-paint') as HTMLButtonElement).disabled = false;
    ($('ai-refine') as HTMLButtonElement).disabled = !state.mesh;
  }
}
let agent: PaintAgent | null = null;

function chatMsg(kind: 'user' | 'agent' | 'sys' | 'err' | 'act', text: string) {
  const d = document.createElement('div');
  d.className = `msg ${kind}`;
  d.textContent = text;
  $('chat-log').appendChild(d);
  $('chat-log').scrollTop = $('chat-log').scrollHeight;
}

function chatStatus(msg: string) {
  $('chat-status').textContent = msg;
}

function updateCost() {
  $('chat-cost').textContent = agent && agent.cost > 0 ? `$${agent.cost.toFixed(2)} this chat` : '';
}

function resetChat(note?: string) {
  agent?.dispose();
  agent = null;
  $('chat-log').innerHTML = '';
  if (note) chatMsg('sys', note);
  ($('agent-img') as HTMLImageElement).hidden = true;
  updateCost();
}

async function sendChat() {
  const m = state.mesh, h = state.history;
  const input = $('chat-input') as HTMLTextAreaElement;
  const text = input.value.trim();
  if (!m || !h || state.busy || !text) return;
  if (($('provider') as HTMLSelectElement).value !== 'anthropic') {
    chatMsg('err', 'The painter chat needs Claude for now: pick Claude in the AI model list.');
    return;
  }
  exitPatchView();
  input.value = '';
  chatMsg('user', text);
  state.busy = true;
  ($('chat-send') as HTMLButtonElement).disabled = true;
  $('chat-stop').hidden = false;
  const t0 = performance.now();
  if (!agent) {
    agent = new PaintAgent(viewer.renderer, m, h, state.filaments, {
      onStatus: (msg) => { chatStatus(msg); updateCost(); },
      onSay: (t) => chatMsg('agent', t),
      onAction: (line) => chatMsg('act', line),
      onImage: (shot) => {
        const img = $('agent-img') as HTMLImageElement;
        img.src = shot.image; img.title = `${shot.id} · ${shot.caption}`; img.hidden = false;
      },
    });
  }
  try {
    if (!agent.started) await agent.start(text, state.reference);
    else await agent.send(text);
    chatStatus(`Done in ${((performance.now() - t0) / 1000).toFixed(0)}s. Ctrl+Z steps back through strokes.`);
  } catch (err) {
    console.error(err);
    chatMsg('err', `Painter failed: ${(err as Error).message}`);
    chatStatus('');
  } finally {
    updateCost();
    state.busy = false;
    ($('chat-send') as HTMLButtonElement).disabled = false;
    $('chat-stop').hidden = true;
  }
}

$('chat-send').addEventListener('click', sendChat);
$('chat-input').addEventListener('keydown', (e) => {
  const k = e as KeyboardEvent;
  if (k.key === 'Enter' && !k.shiftKey) { k.preventDefault(); sendChat(); }
});
$('chat-stop').addEventListener('click', () => { if (agent) { agent.stopped = true; chatStatus('Stopping after the current step…'); } });
$('chat-new').addEventListener('click', () => { if (!state.busy) resetChat('New chat. The current paint stays; describe what to do.'); });
$('agent-img').addEventListener('click', () => {
  const src = ($('agent-img') as HTMLImageElement).src;
  window.open()?.document.write(`<img src="${src}">`);
});

$('ai-paint').addEventListener('click', () => runAi('initial'));
$('ai-refine').addEventListener('click', () => runAi('refine'));

// ---------- init ----------

renderFilaments();
setTool('brush');
updateLabels();
loadProviders();

// Expose for debugging in the browser console.
(window as unknown as Record<string, unknown>).painter = {
  state, viewer, openFile, setReference,
  thickness: () => computeThickness(state.mesh!),
  shells: () => shells(state.mesh!),
  copyPaint: (source: number) => { const r = copyPaint(state.mesh!, source); state.mesh!.refreshColors(); state.history!.recordApplied(r.faces); return r; },
  renderViews: (leftMode: "clay" | "painted" = "clay") => renderPatchViews(viewer.renderer, state.mesh!, ensurePatches(), { leftMode }),
};
