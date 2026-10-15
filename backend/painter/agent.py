"""Hand-painting agent: Claude looks at renders and uses slicer-style tools.

The browser owns the model, so it executes every tool call (rendering views,
raycasting clicks, filling faces) and posts the results back here. This
module only keeps each session's conversation and talks to the API.
"""

import os
import uuid

import anthropic

from .llm import ProviderError, _split_data_url

SYSTEM = """\
You are painting a 3D-printable model for a multi-filament printer, the way a \
person hand-paints in a slicer: you look at the model from different angles, \
click on features with the slicer's tools, then check your work and fix it.

Every image you get is a render of the model in its current paint, with a \
coordinate grid. Points are given as image pixel coordinates (x right, y down) \
on a specific image id. Clicks land on the surface under that pixel.

Tools:
- look: render a view. Pick azimuth/elevation, and optionally zoom in centred on a pixel of an earlier image. Zoom in close for small features: eyes, pupils, teeth, nostrils, claw tips.
- paint: apply one or more strokes on an image. Stroke tools:
  - outline: trace a polygon (points, at least 3, in order around the shape) around a feature as you see it in the image; all surface visible inside the outline gets painted. This is the most precise tool: use it whenever a feature has no sharp edge where it meets the body (e.g. spikes that flare smoothly out of the body). Trace just inside the feature's silhouette. Surfaces hidden in that view are not painted, so do the other side from another view.
  - line: brush along a path (points) with radius_mm, e.g. a stripe.
  - smart_fill: fill the connected surface around (x, y) until an edge sharper than angle_deg (default 30). Good for crease-bounded features.
  - thin_fill: fill from (x, y) through every connected surface thinner than max_thickness_mm (default 5). Catches thin blades (spike tips, claws, teeth, horn tips) but stops where they thicken.
  - thin_fill_similar: thin_fill plus every other thin part of the same shape and size across the model.
  - fill_similar: smart_fill plus every other crease-bounded feature of the same shape and size.
  - brush: paint a disc of radius_mm around (x, y), front-facing surface only.
- segment_rule_fill: THE tool for spikes, fins and plates on segmented \
bodies. Click one body segment; in that segment and every segment with the \
same shape, it paints everything above height_frac of the segment's own \
height OR further sideways than reach_frac of its own half-width, each \
scaled to that segment's size, then smooths the cut lines. Spikes stick up \
and out, so this colours whole spikes consistently on every segment with \
clean edges. Good starting values: height_frac 0.72, reach_frac 0.7. Lower \
height_frac = more of the back coloured; lower reach_frac = more of the \
sides. Check a close-up and adjust (undo, retry). Segments with an unusual \
shape (e.g. where legs attach) are not matched: click one of those and run \
it again with the same values. Never trace spikes with outline on a \
segmented body. stripe_frac adds a stripe down the centre of the spine on the same segments (0 for none); paint it in its own call if it needs a different colour from the spikes.
- smooth_edges: clean stair-stepped colour borders on the whole model.
- range_fill: paint by position, like a slicer's height-range tool. Paint \
everything between heights z_min and z_max (mm above the model's lowest \
point), and/or between r_min and r_max mm horizontally from the centre of \
a clicked part. Give image/x/y to limit it to the part under that pixel \
(r needs this), or null for the whole model. Spikes are the parts that stick \
furthest up and out, so a band like "above 28 mm" or "further than 14 mm \
from this segment's centre" isolates them without tracing. The result \
tells you the clicked part's height span and horizontal reach so you can \
pick bands; check the render and adjust.
- copy_to_matching_parts: click a part of the model; its paint is copied onto every other part with the same shape (scaled to fit), e.g. all the segments of an articulated body, or all four legs. Articulated prints are built from many near-identical separate segments: paint ONE segment completely and carefully (all its spikes, stripes, every side of it, zoomed in), check it, then copy it to the rest instead of painting each segment. The head, legs and tail tip are usually different shapes and need their own treatment (legs can be copied leg to leg).
- fill_all: base coat the whole model in one filament.
- undo: revert the last paint, fill, copy or smoothing call.
- done: finish this request with a short summary for the user.

After each paint call you get a fresh render of the same view showing what \
changed, how much of the model each stroke covered, and how much of the \
model each filament now covers. A stroke that covers far more area than the \
feature you meant has leaked: undo and retry with an outline, a lower angle \
or a brush. Use the coverage numbers as a sanity check: if the brief \
describes a black body with green spikes, black must stay the clear \
majority, and a stroke that covers a whole body segment is not "a spike".

Work like a careful painter:
1. Look at the overview images and identify the parts the brief mentions.
2. Base coat with the main body colour.
3. Paint large features, then small ones, zooming in for small ones.
4. Do both sides: features on the left usually have a twin on the right.
5. Before finishing, look from a few angles (including the other side) and \
fix anything missed or wrong.
Be efficient: batch several strokes in one paint call when you can see them \
all in one image. Use only the filament names given."""

_IMG_REF = {"type": "string", "description": "Image id the point refers to, e.g. img-3"}
_POINT = {"type": "integer", "description": "Pixel coordinate on that image"}

TOOLS = [
    {
        "name": "look",
        "description": "Render the model from a direction. azimuth: 0 = front (-Y side), 90 = right (+X), 180 = back, 270 = left. elevation: degrees above the horizon (-90 underside .. 90 straight down from the top). zoom: 1 shows the whole model; 2-8 magnify around focus.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "azimuth": {"type": "number"},
                "elevation": {"type": "number"},
                "zoom": {"type": "number"},
                "focus": {
                    "type": ["object", "null"],
                    "description": "Centre the view on the surface under this pixel of an earlier image, or null to centre on the whole model.",
                    "properties": {"image": _IMG_REF, "x": _POINT, "y": _POINT},
                    "required": ["image", "x", "y"],
                    "additionalProperties": False,
                },
            },
            "required": ["azimuth", "elevation", "zoom", "focus"],
            "additionalProperties": False,
        },
    },
    {
        "name": "paint",
        "description": "Apply strokes on an image. Point tools use x, y; outline and line use points.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "strokes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "image": _IMG_REF,
                            "tool": {
                                "type": "string",
                                "enum": ["outline", "line", "smart_fill", "thin_fill", "thin_fill_similar", "fill_similar", "brush"],
                            },
                            "color": {"type": "string", "description": "Filament name"},
                            "x": {"type": ["integer", "null"], "description": "Click pixel for point tools, else null"},
                            "y": {"type": ["integer", "null"]},
                            "points": {
                                "type": ["array", "null"],
                                "description": "Pixel points for outline / line, else null",
                                "items": {
                                    "type": "object",
                                    "properties": {"x": _POINT, "y": _POINT},
                                    "required": ["x", "y"],
                                    "additionalProperties": False,
                                },
                            },
                            "angle_deg": {"type": ["number", "null"], "description": "smart_fill / fill_similar edge angle; null for 30"},
                            "radius_mm": {"type": ["number", "null"], "description": "brush / line radius; null for 1.5"},
                            "max_thickness_mm": {"type": ["number", "null"], "description": "thin_fill limit; null for 5"},
                        },
                        "required": ["image", "tool", "color", "x", "y", "points", "angle_deg", "radius_mm", "max_thickness_mm"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["strokes"],
            "additionalProperties": False,
        },
    },
    {
        "name": "segment_rule_fill",
        "description": "Colour spikes on a segmented body: in the clicked segment and all same-shaped segments, paint above height_frac of each segment's height or beyond reach_frac of its half-width.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "image": _IMG_REF,
                "x": _POINT,
                "y": _POINT,
                "color": {"type": "string"},
                "height_frac": {"type": "number", "description": "0-1, e.g. 0.72"},
                "reach_frac": {"type": "number", "description": "0-1, e.g. 0.7"},
                "stripe_frac": {"type": "number", "description": "Width of a stripe down the centre of each segment's back, as a fraction of its half-width (0 = no stripe, ~0.12 = narrow, 0.25 = wide)"},
            },
            "required": ["image", "x", "y", "color", "height_frac", "reach_frac", "stripe_frac"],
            "additionalProperties": False,
        },
    },
    {
        "name": "smooth_edges",
        "description": "Smooth jagged colour borders across the whole model.",
        "strict": True,
        "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
    {
        "name": "range_fill",
        "description": "Paint surface inside a height band and/or horizontal distance band (from a clicked part's centre). Use null for unused bounds.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "color": {"type": "string"},
                "image": {"type": ["string", "null"], "description": "Image id to click a part on, or null for whole model"},
                "x": {"type": ["integer", "null"]},
                "y": {"type": ["integer", "null"]},
                "z_min": {"type": ["number", "null"], "description": "mm above the model's lowest point"},
                "z_max": {"type": ["number", "null"]},
                "r_min": {"type": ["number", "null"], "description": "mm horizontally from the clicked part's centre"},
                "r_max": {"type": ["number", "null"]},
            },
            "required": ["color", "image", "x", "y", "z_min", "z_max", "r_min", "r_max"],
            "additionalProperties": False,
        },
    },
    {
        "name": "copy_to_matching_parts",
        "description": "Copy the paint of the part under this pixel onto every other part of the model with the same shape.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"image": _IMG_REF, "x": _POINT, "y": _POINT},
            "required": ["image", "x", "y"],
            "additionalProperties": False,
        },
    },
    {
        "name": "fill_all",
        "description": "Paint the entire model one filament (base coat).",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"color": {"type": "string"}},
            "required": ["color"],
            "additionalProperties": False,
        },
    },
    {
        "name": "undo",
        "description": "Revert the most recent paint or fill_all call.",
        "strict": True,
        "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
    {
        "name": "done",
        "description": "Finish painting.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        },
    },
]


# USD per million tokens: input, output, cache write, cache read.
PRICES = {
    "claude-opus-5-5": (4.0, 20.0, 5.0, 0.20),
    "claude-sonnet-5-5": (2.0, 10.0, 2.5, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.125, 0.01),
}


def _cost(model: str, usage) -> float:
    inp, out, write, read = PRICES.get(model, PRICES["claude-opus-5-5"])
    return (
        usage.input_tokens * inp
        + usage.output_tokens * out
        + (usage.cache_creation_input_tokens or 0) * write
        + (usage.cache_read_input_tokens or 0) * read
    ) / 1e6


def _image_block(url: str) -> dict:
    mt, data = _split_data_url(url)
    return {"type": "image", "source": {"type": "base64", "media_type": mt, "data": data}}


class AgentSessions:
    def __init__(self):
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5")
        self.effort = os.getenv("ANTHROPIC_AGENT_EFFORT", "medium")
        configured = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
        self.client = anthropic.AsyncAnthropic() if configured else None
        self.sessions: dict[str, list[dict]] = {}

    async def start(self, brief: str, filaments: list[dict], reference: str | None, overview: list[dict], facts: str = "") -> dict:
        if not self.client:
            raise ProviderError("ANTHROPIC_API_KEY is not set (put it in local.env)")
        content: list[dict] = []
        if reference:
            content += [{"type": "text", "text": "Reference image from the user:"}, _image_block(reference)]
        for v in overview:
            content += [{"type": "text", "text": f"{v['id']}: {v['caption']}"}, _image_block(v["image"])]
        lines = ["Filaments loaded (name — hex):"] + [f"- {f['name']} — {f['hex']}" for f in filaments]
        if facts:
            lines += ["", facts]
        lines += ["", "The brief:", brief or "(none — follow the reference image)", "", "Start painting."]
        content.append({"type": "text", "text": "\n".join(lines)})
        sid = uuid.uuid4().hex
        self.sessions[sid] = [{"role": "user", "content": content}]
        return {"session": sid, **await self._turn(sid)}

    async def step(self, sid: str, results: list[dict], message: str | None = None) -> dict:
        messages = self.sessions.get(sid)
        if messages is None:
            raise ProviderError("Unknown or expired agent session")
        blocks: list[dict] = []
        for r in results:
            inner: list[dict] = [{"type": "text", "text": r.get("text") or "ok"}]
            for img in r.get("images") or []:
                inner += [{"type": "text", "text": f"{img['id']}: {img['caption']}"}, _image_block(img["image"])]
            blocks.append({"type": "tool_result", "tool_use_id": r["id"], "content": inner, "is_error": bool(r.get("error"))})
        if message:
            # A chat follow-up from the user rides along with the result of the last done call.
            blocks.append({"type": "text", "text": f"The user says: {message}"})
        messages.append({"role": "user", "content": blocks})
        return await self._turn(sid)

    async def _turn(self, sid: str) -> dict:
        messages = self.sessions[sid]
        try:
            msg = await self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=SYSTEM,
                tools=TOOLS,
                messages=messages,
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                cache_control={"type": "ephemeral"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.AuthenticationError as e:
            raise ProviderError("Anthropic rejected the API key") from e
        except anthropic.RateLimitError as e:
            raise ProviderError("Anthropic rate limit hit, try again shortly") from e
        except anthropic.APIStatusError as e:
            raise ProviderError(f"Anthropic API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise ProviderError("Could not reach the Anthropic API") from e

        if msg.stop_reason == "refusal":
            raise ProviderError("The model declined this request")
        # Keep the full content (thinking blocks included) for the next turn.
        messages.append({"role": "assistant", "content": msg.content})
        calls = [{"id": b.id, "name": b.name, "input": b.input} for b in msg.content if b.type == "tool_use"]
        text = "\n".join(b.text for b in msg.content if b.type == "text")
        usage = msg.usage
        return {
            "calls": calls,
            "text": text,
            "finished": not calls,
            "cost": _cost(self.model, usage),
            "usage": {
                "input_tokens": usage.input_tokens,
                "cache_write_input_tokens": usage.cache_creation_input_tokens or 0,
                "cache_read_input_tokens": usage.cache_read_input_tokens or 0,
                "output_tokens": usage.output_tokens,
            },
        }
