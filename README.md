# Model Painter

A local web app that paints 3D-print models for multi-colour (multi-filament) printing.

Load an STL / OBJ / 3MF, list the filaments you have loaded, then either paint by hand with
slicer-style tools or describe the look in a chat ("make me a poison dragon: green spikes, black
body, yellow eyes with black slit pupils") and let Claude paint it. Export a 3MF that
**Elegoo Slicer, OrcaSlicer and Bambu Studio open already painted**, one filament per colour.

It runs on your own machine: the app is served on `127.0.0.1` and your models never leave it,
except for the rendered views sent to the AI when you use the chat painter.

> Status: personal project, works well on articulated / segmented models (dragons!). Fine one-off
> details are still often quicker to finish by hand in the slicer.

## Requirements

| | Version | Notes |
|---|---|---|
| [Python](https://www.python.org/downloads/) | 3.11+ | |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | any recent | Python package/env manager |
| [Node.js](https://nodejs.org/) | 22.12+ (or 20.19+) | only to build the web UI |
| Browser | recent Chrome / Edge / Firefox | needs WebGL2; large models (~1M triangles) want 8 GB+ RAM |
| Anthropic API key | optional | needed for the AI chat painter; manual tools work without it |

## Install

```bash
git clone https://github.com/skipzx-lab/model-painter.git
cd model-painter

# 1. Build the web UI (once, and again after pulling updates)
cd frontend
npm install
npm run build
cd ..

# 2. Install the Python side
uv sync

# 3. Configure (optional, for the AI painter)
cp local.env.example local.env        # Windows PowerShell: Copy-Item local.env.example local.env
# then edit local.env and set ANTHROPIC_API_KEY
```

## Run

```bash
uv run painter
```

Open **http://127.0.0.1:8517**. Stop it with Ctrl+C.

To use a different port: `PAINTER_PORT=9000 uv run painter`
(PowerShell: `$env:PAINTER_PORT=9000; uv run painter`).

## Configuration

Settings are read from `local.env` (or `.env`) in the project root. Both are git-ignored.
See `local.env.example`.

| Variable | Default | What it does |
|---|---|---|
| `ANTHROPIC_API_KEY` | none | Enables the Claude chat painter. Get one at console.anthropic.com |
| `ANTHROPIC_MODEL` | `claude-opus-5-5` | `claude-sonnet-5-5` costs about half |
| `ANTHROPIC_AGENT_EFFORT` | `medium` | `low` / `medium` / `high` thinking effort for the chat painter |
| `ANTHROPIC_EFFORT` | `high` | effort for the one-shot region painter |
| `LOCAL_LLM_BASE_URL` | `http://localhost:11434/v1` | any OpenAI-compatible server (Ollama, Open WebUI `/api`, llama.cpp, vLLM). One-shot mode only |
| `LOCAL_LLM_API_KEY` | none | bearer token for that server, if it needs one |
| `LOCAL_LLM_MODEL` | none | a vision-capable model name; local mode is off until this is set |
| `PAINTER_PORT` | `8517` | port the app listens on |

**Cost:** a full chat paint with Claude Opus is typically $0.50–2; a follow-up correction or a
head-only pass is usually $0.10–0.30. The chat panel shows the running cost.

## Using it

1. **Open model**: STL, OBJ, PLY, GLB, or 3MF (including Bambu/Orca project files; existing paint is
   imported). Or drop the file on the window.
2. **Filaments**: one row per extruder slot, in the order they're loaded in your printer (slot 1 =
   extruder 1). Up to 16. Number keys pick the active one.
3. **Paint by hand** with the tools on the left:
   - **Brush** (B), **Smart fill** (F, stops at edges sharper than the angle), **Bucket** (G),
     **Thin fill** (T, spikes/claws/teeth: fills through thin parts up to a thickness).
   - **Spike fill** (S), for articulated / segmented bodies: click one body segment and it colours
     everything above the height % or beyond the sideways-reach % (plus an optional spine stripe) on
     that segment and every same-shaped segment, each scaled to its own size, with smoothed edges.
     Odd-shaped segments (e.g. where legs attach) need their own click.
   - **Copy part** (C) copies one part's paint onto every matching part.
   - Left-drag paints, right-drag orbits, middle-drag pans. Ctrl+Z / Ctrl+Y undo and redo.
4. **Or ask the painter** in the chat panel on the right. It looks at renders of the model, uses the
   same tools, checks its work and fixes it. Follow up with changes ("make the claws white"). Stop
   pauses it; every stroke is on the undo stack. Optionally drop a reference image first.
5. **Export 3MF** and open it in your slicer.

### Opening the export in your slicer

- Load the **same number of filaments, in the same order** as the app's filament list, before or
  after importing. Painted colours map to extruders by slot number.
- Elegoo Slicer / OrcaSlicer / Bambu Studio read the paint (`paint_color`); PrusaSlicer reads the
  `slic3rpe:mmu_segmentation` copy for slots 1–16.
- The export contains the geometry and paint only, with no printer or print settings, so your
  slicer's current profile is used.

## Security

- The server listens on `127.0.0.1` only and has **no authentication**. Don't expose it to a
  network or the internet (no port forwarding, no `0.0.0.0`).
- Your API key stays in the Python server; the browser never sees it.
- The chat painter sends rendered images of your model (and any reference image) to the Anthropic
  API. The one-shot mode also saves its requests to `runs/` locally for debugging (git-ignored).

## Troubleshooting

| Problem | Fix |
|---|---|
| Blank page / 404 at the root URL | Build the UI: `cd frontend && npm install && npm run build` |
| `address already in use` | Another program has the port: set `PAINTER_PORT` |
| Model list says "Backend offline" | The Python server isn't running, or you opened a different port |
| Chat says it needs Claude | Set `ANTHROPIC_API_KEY` in `local.env` and restart |
| Very slow first Thin fill | It measures wall thickness once per model (10–20 s for ~1M triangles) |
| Colours look wrong in the slicer | Filament count/order in the slicer doesn't match the app's list |
| Browser tab runs out of memory | Very dense meshes: decimate in your slicer or Blender first |

## Development

```bash
# terminal 1: API server
uv run painter
# terminal 2: UI with hot reload on http://localhost:5173 (proxies /api to :8517)
cd frontend && npm run dev
```

Checks:

```bash
cd frontend
npx tsc --noEmit                                    # type-check
npx tsx tests/roundtrip.ts model.3mf [out.3mf]      # paint survives export + re-import
```

### Layout

- `frontend/src/mesh.ts`: welded mesh, per-face filament slots, display geometry
- `frontend/src/tools.ts`: brush, smart fill, bucket, range fill, edge smoothing, despeckle
- `frontend/src/thickness.ts`: wall-thickness measurement and thin fill
- `frontend/src/transfer.ts`: part detection, copy-to-matching-parts, spike fill
- `frontend/src/reader3mf.ts`, `export3mf.ts`: 3MF in/out, `paint_color` encoding
- `frontend/src/ai/agent.ts`: browser side of the chat painter (renders, raycast clicks, runs tools)
- `backend/painter/agent.py`: chat painter conversation with Claude (tool definitions, prompt)
- `backend/painter/llm.py`, `prompts.py`: one-shot region painting (Claude or a local model)
- `backend/painter/server.py`: FastAPI app, serves `frontend/dist`

## Credits and licence

MIT licensed (see `LICENSE`). The per-triangle `paint_color` encoding follows the scheme used by
[Texture2Paint](https://github.com/EllsworOpan/Texture2Paint)'s three-mf-kit (MIT, © 2026 EllsworOpan).
Built on [three.js](https://threejs.org), [three-mesh-bvh](https://github.com/gkjohnson/three-mesh-bvh),
[fflate](https://github.com/101arrowz/fflate), FastAPI and the Anthropic Python SDK (all MIT/BSD).

No 3D models are included: bring your own, and respect their licences when sharing painted files.
