"""Prompt and output schema for the region-colouring requests."""

SYSTEM = """\
You plan colour layouts for multi-filament 3D prints. The slicer prints every \
surface region of the model in exactly one of the user's filaments, so your \
job is to choose a filament for each numbered region so the printed model \
looks the way the user describes.

What you receive:
- The user's description of the look they want. This is the brief: follow \
it literally and with taste. If they say "green spikes", every spike is green; \
if they describe an eye with a pupil, the eye and pupil get those colours.
- Optionally a reference image showing a colour scheme to imitate. The \
creature in it is usually a different sculpt in a different pose, so transfer \
the scheme by anatomy, not by pixel position.
- Several views of the model. Each image has two halves: on the left the \
model itself (plain grey, or its current paint), on the right the same view \
split into regions, each labelled with a red number. Region borders follow \
the sculpt's own creases, so a region is usually one natural feature: a \
spike, a plate, a claw, a horn ring, a tooth. The grey shades on the right \
only separate regions; they carry no colour meaning.
- The filaments available. Use only those names.

How to work:
1. Work out the model first: what it is, which way it faces, where the head, \
eyes, mouth, limbs, spikes, tail and any base are, and which region numbers \
belong to each. A region seen in several views is the same region.
2. Decide the scheme per anatomical feature from the brief, then apply it \
consistently: every spike, every claw, both sides of the body alike.
3. Anything the brief doesn't mention gets the body's main colour. Avoid \
stray single regions in an accent colour unless they are a feature the brief \
asks for.

Return JSON matching the schema."""

INITIAL_TAIL = """\
Assign a filament to every region number you can read. Also list in \
detail_regions the regions that contain small features the brief cares about \
but that are too small or too merged to colour properly at this scale \
(typically the regions covering the eyes, mouth/teeth, nostrils). They will be \
re-cut much finer and shown to you close up in a second pass. Leave it empty \
if nothing needs that."""

REFINE_TAIL = """\
This is a correction pass. The left half of each view now shows the current \
paint. Compare it with the brief, fix anything wrong or inconsistent, and \
return the complete corrected assignment for every visible region (not just \
changes). detail_regions may be empty."""

DETAIL_TAIL = """\
This is the close-up detail pass. The views are zoomed in on one area of the \
model (everything else is cut away or unlabelled), already painted from the \
first pass, which is shown on the left. The area has been re-cut into much \
smaller regions so that small features (eyeball, pupil, teeth, nostrils, \
lips, claw tips) are separate. Assign filaments ONLY to regions that need to \
change or that form these small features; regions you leave out keep their \
current paint. Be precise: a pupil is the small centre of the eye, teeth are \
the individual pointed shapes along the jaw. detail_regions must be empty."""


def user_text(description: str, filaments: list[dict], patch_count: int, mode: str, current: list[dict] | None) -> str:
    lines = ["Filaments available (name — hex):"]
    lines += [f"- {f['name']} — {f['hex']}" for f in filaments]
    lines.append("")
    lines.append(f"Regions are numbered 0 to {patch_count - 1}.")
    if description:
        lines += ["", "The brief:", description]
    else:
        lines += ["", "No written brief: follow the reference image."]
    if current and mode in ("refine", "detail"):
        by_colour: dict[str, list[int]] = {}
        for a in current:
            by_colour.setdefault(a["color"], []).append(a["patch"])
        lines += ["", "Current paint per region:"]
        lines += [f"- {c}: {sorted(ps)}" for c, ps in by_colour.items()]
    lines += ["", {"initial": INITIAL_TAIL, "refine": REFINE_TAIL, "detail": DETAIL_TAIL}[mode]]
    return "\n".join(lines)


def output_schema(filament_names: list[str]) -> dict:
    colour = {"type": "string", "enum": filament_names}
    return {
        "type": "object",
        "properties": {
            "plan": {"type": "string"},
            "default_color": colour,
            "assignments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"patch": {"type": "integer"}, "color": colour},
                    "required": ["patch", "color"],
                    "additionalProperties": False,
                },
            },
            "detail_regions": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["plan", "default_color", "assignments", "detail_regions"],
        "additionalProperties": False,
    }
