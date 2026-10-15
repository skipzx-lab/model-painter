import { strToU8, zipSync } from 'fflate';
import type { PaintMesh } from './mesh';

/**
 * Per-triangle paint code as OrcaSlicer / Bambu Studio store it in the
 * `paint_color` attribute (TriangleSelector serialisation of an unsplit
 * triangle). `state` is the 1-based extruder; 0 means "object default".
 *   1 -> "4", 2 -> "8", 3 -> "0C", 4 -> "1C", ... 18 -> "0FC"
 * Same scheme as the encoder in Texture2Paint's three-mf-kit (MIT).
 */
export function encodePaint(state: number, dialect: 'bambu' | 'prusa' = 'bambu'): string {
  if (!Number.isInteger(state) || state < 0 || state > 255) throw new Error('Invalid extruder index');
  if (state < 3) return (state * 4).toString(16).toUpperCase();
  if (dialect === 'bambu') {
    return ((state - 3) % 15).toString(16).toUpperCase() + 'F'.repeat(Math.floor((state - 3) / 15)) + 'C';
  }
  return state < 17
    ? (state - 3).toString(16).toUpperCase() + 'C'
    : (state - 17).toString(16).toUpperCase().padStart(2, '0') + 'EC';
}

const xmlEscape = (s: string) =>
  s.replace(/[<>&"']/g, (c) => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;', "'": '&apos;' })[c]!);

export interface ExportOptions {
  name: string;
  palette: string[]; // '#rrggbb' per slot
}

/**
 * Build a 3MF that Elegoo Slicer / OrcaSlicer opens with the paint already
 * applied. Slot i (0-based) becomes extruder i+1. Also writes a standard
 * 3MF colour group so generic viewers show the colours, and the PrusaSlicer
 * attribute for the first 16 slots.
 */
export function export3mf(mesh: PaintMesh, opts: ExportOptions): Uint8Array {
  if (opts.palette.length > 16) throw new Error('OrcaSlicer-family slicers support up to 16 painted filaments.');
  const p = mesh.positions, f = mesh.faces, s = mesh.slots;
  const codes = opts.palette.map((_, i) => encodePaint(i + 1, 'bambu'));
  const prusa = opts.palette.map((_, i) => encodePaint(i + 1, 'prusa'));

  const parts: string[] = [];
  parts.push(
    '<?xml version="1.0" encoding="UTF-8"?>\n',
    '<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" ',
    'xmlns:slic3rpe="http://schemas.slic3r.org/3mf/2017/06" xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02">\n',
    '<metadata name="Application">3dModelPainter</metadata>\n',
    '<metadata name="slic3rpe:Version3mf">1</metadata>\n',
    '<metadata name="slic3rpe:MmPaintingVersion">1</metadata>\n',
    '<resources>\n<m:colorgroup id="1">',
    opts.palette.map((c) => `<m:color color="${c.toUpperCase()}FF"/>`).join(''),
    '</m:colorgroup>\n',
    `<object id="2" type="model" name="${xmlEscape(opts.name)}" pid="1" pindex="0"><mesh><vertices>\n`,
  );
  const chunk: string[] = [];
  for (let v = 0; v < p.length; v += 3) {
    chunk.push(`<vertex x="${+p[v].toFixed(5)}" y="${+p[v + 1].toFixed(5)}" z="${+p[v + 2].toFixed(5)}"/>`);
    if (chunk.length >= 10000) { parts.push(chunk.join('\n'), '\n'); chunk.length = 0; }
  }
  parts.push(chunk.join('\n'), '\n</vertices><triangles>\n');
  chunk.length = 0;
  for (let i = 0; i < s.length; i++) {
    const slot = s[i];
    chunk.push(
      `<triangle v1="${f[i * 3]}" v2="${f[i * 3 + 1]}" v3="${f[i * 3 + 2]}" pid="1" p1="${slot}" paint_color="${codes[slot]}" slic3rpe:mmu_segmentation="${prusa[slot]}"/>`,
    );
    if (chunk.length >= 10000) { parts.push(chunk.join('\n'), '\n'); chunk.length = 0; }
  }
  parts.push(chunk.join('\n'), '\n</triangles></mesh></object>\n</resources>\n<build><item objectid="2"/></build>\n</model>\n');

  const contentTypes =
    '<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">' +
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>' +
    '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>' +
    '<Default Extension="config" ContentType="text/plain"/></Types>';
  const rels =
    '<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' +
    '<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>';

  return zipSync(
    {
      '[Content_Types].xml': strToU8(contentTypes),
      '_rels/.rels': strToU8(rels),
      '3D/3dmodel.model': strToU8(parts.join('')),
    },
    { level: 6 },
  );
}
