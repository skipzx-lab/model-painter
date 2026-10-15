// Round-trip check: read a 3MF, paint faces by a simple rule, export, re-read
// and confirm every face's extruder survived. Optionally writes the export.
//   npx tsx tests/roundtrip.ts <in.3mf> [out.3mf]
import { readFileSync, writeFileSync } from 'node:fs';
import { read3mf, decodePaint } from '../src/reader3mf';
import { PaintMesh } from '../src/mesh';
import { export3mf, encodePaint } from '../src/export3mf';

for (let s = 0; s <= 40; s++) {
  if (decodePaint(encodePaint(s, 'bambu')) !== s) throw new Error(`paint code mismatch for ${s}: ${encodePaint(s, 'bambu')}`);
}
console.log('paint codes 0-40 round-trip ok; 1..4 =', [1, 2, 3, 4].map((s) => encodePaint(s)).join(' '));

const [inPath, outPath] = process.argv.slice(2);
const src = read3mf(readFileSync(inPath).buffer as ArrayBuffer);
const mesh = PaintMesh.fromIndexed(src.positions, src.faces);
// Four bands along X so the result is easy to recognise in a slicer.
const bb = mesh.boundingBox();
for (let f = 0; f < mesh.faceCount; f++) {
  const t = (mesh.centroids[f * 3] - bb.min.x) / (bb.max.x - bb.min.x);
  mesh.slots[f] = Math.min(3, Math.floor(t * 4));
}
const bytes = export3mf(mesh, { name: 'roundtrip', palette: ['#1b1b1d', '#6cd12c', '#4a4f57', '#e6dcc3'] });
const back = read3mf(bytes.buffer as ArrayBuffer);
if (back.faces.length !== mesh.faces.length) throw new Error('face count changed');
let bad = 0;
for (let f = 0; f < mesh.faceCount; f++) if (back.extruders![f] !== mesh.slots[f] + 1) bad++;
console.log(`faces ${mesh.faceCount}, mismatched extruders ${bad}, size ${(bytes.length / 1e6).toFixed(1)} MB`);
if (bad) process.exit(1);
if (outPath) { writeFileSync(outPath, bytes); console.log('wrote', outPath); }
