"""Local web server: serves the painter UI and proxies vision-model calls."""

import base64
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / "local.env")
load_dotenv(ROOT / ".env")

from . import agent, llm  # noqa: E402  (reads env at import)

app = FastAPI(title="Model Painter")
PROVIDERS = llm.providers()
RUNS = ROOT / "runs"


class Filament(BaseModel):
    name: str
    hex: str


class View(BaseModel):
    name: str
    image: str
    labelled: list[int] = []


class Assignment(BaseModel):
    patch: int
    color: str


class PaintRequest(BaseModel):
    provider: str = "anthropic"
    description: str = ""
    filaments: list[Filament]
    reference_image: str | None = None
    views: list[View]
    patch_count: int
    mode: str = "initial"
    current: list[Assignment] | None = None


@app.get("/api/providers")
def get_providers():
    return [
        {"id": p.id, "label": p.label, "model": p.model or "(no model set)", "configured": p.configured}
        for p in PROVIDERS.values()
    ]


def _save_run(req: dict, result: dict | None, error: str | None):
    """Keep each request's images and answer on disk for prompt tuning."""
    d = RUNS / time.strftime("%Y%m%d-%H%M%S")
    d.mkdir(parents=True, exist_ok=True)
    images = [("reference", req.get("reference_image"))] + [(v["name"], v["image"]) for v in req["views"]]
    for i, (name, url) in enumerate(images):
        if not url:
            continue
        safe = "".join(c if c.isalnum() else "_" for c in name)
        (d / f"{i:02d}-{safe}.jpg").write_bytes(base64.b64decode(url.split(",", 1)[1]))
    meta = {k: v for k, v in req.items() if k not in ("views", "reference_image")}
    (d / "request.json").write_text(json.dumps(meta, indent=2))
    (d / "response.json").write_text(json.dumps(result or {"error": error}, indent=2))


@app.post("/api/paint")
async def paint(body: PaintRequest):
    provider = PROVIDERS.get(body.provider)
    if not provider:
        raise HTTPException(400, f"Unknown provider {body.provider}")
    if not body.filaments:
        raise HTTPException(400, "Add at least one filament")
    if len({f.name.strip().lower() for f in body.filaments}) != len(body.filaments):
        raise HTTPException(400, "Filament names must be unique")
    req = body.model_dump()
    try:
        res = await provider.paint(req)
    except llm.ProviderError as e:
        _save_run(req, None, str(e))
        raise HTTPException(502, str(e)) from e
    out = {
        "plan": res.plan,
        "default_color": res.default_color,
        "assignments": res.assignments,
        "detail_regions": res.detail_regions,
        "model": res.model,
        "usage": res.usage,
    }
    _save_run(req, out, None)
    return out


AGENT = agent.AgentSessions()


class OverviewImage(BaseModel):
    id: str
    caption: str
    image: str


class AgentStart(BaseModel):
    description: str = ""
    facts: str = ""
    filaments: list[Filament]
    reference_image: str | None = None
    overview: list[OverviewImage]


class ToolResult(BaseModel):
    id: str
    text: str = ""
    error: bool = False
    images: list[OverviewImage] = []


class AgentStep(BaseModel):
    session: str
    results: list[ToolResult]
    message: str | None = None


@app.post("/api/agent/start")
async def agent_start(body: AgentStart):
    try:
        return await AGENT.start(
            body.description,
            [f.model_dump() for f in body.filaments],
            body.reference_image,
            [v.model_dump() for v in body.overview],
            body.facts,
        )
    except llm.ProviderError as e:
        raise HTTPException(502, str(e)) from e


@app.post("/api/agent/step")
async def agent_step(body: AgentStep):
    try:
        return await AGENT.step(body.session, [r.model_dump() for r in body.results], body.message)
    except llm.ProviderError as e:
        raise HTTPException(502, str(e)) from e


dist = ROOT / "frontend" / "dist"
if dist.exists():
    app.mount("/", StaticFiles(directory=dist, html=True), name="ui")


def main():
    import uvicorn

    port = int(os.getenv("PAINTER_PORT", "8517"))
    print(f"Model Painter on http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
