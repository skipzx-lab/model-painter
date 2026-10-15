"""Vision model providers: Anthropic (default) and an OpenAI-compatible local server."""

import json
import os
import re
from dataclasses import dataclass

import anthropic
import httpx

from . import prompts


class ProviderError(Exception):
    pass


@dataclass
class PaintResult:
    plan: str
    default_color: str
    assignments: list[dict]
    detail_regions: list[int]
    model: str
    usage: dict


def _split_data_url(url: str) -> tuple[str, str]:
    m = re.match(r"data:(image/[\w.+-]+);base64,(.*)", url, re.S)
    if not m:
        raise ProviderError("Images must be base64 data URLs")
    return m.group(1), m.group(2)


def _clean(data: dict, names: list[str], patch_count: int) -> tuple[str, str, list[dict], list[int]]:
    lookup = {n.lower(): n for n in names}
    default = lookup.get(str(data.get("default_color", "")).strip().lower(), names[0])
    out = []
    for a in data.get("assignments", []):
        try:
            p = int(a["patch"])
        except (KeyError, TypeError, ValueError):
            continue
        c = lookup.get(str(a.get("color", "")).strip().lower())
        if c and 0 <= p < patch_count:
            out.append({"patch": p, "color": c})
    detail = []
    for r in data.get("detail_regions") or []:
        try:
            r = int(r)
        except (TypeError, ValueError):
            continue
        if 0 <= r < patch_count:
            detail.append(r)
    return str(data.get("plan", "")), default, out, detail


class AnthropicProvider:
    id = "anthropic"
    label = "Claude"

    def __init__(self):
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5")
        self.effort = os.getenv("ANTHROPIC_EFFORT", "high")
        self.configured = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
        self.client = anthropic.AsyncAnthropic() if self.configured else None

    async def paint(self, req: dict) -> PaintResult:
        if not self.client:
            raise ProviderError("ANTHROPIC_API_KEY is not set (put it in local.env)")
        names = [f["name"] for f in req["filaments"]]
        content: list[dict] = []
        if req.get("reference_image"):
            mt, data = _split_data_url(req["reference_image"])
            content.append({"type": "text", "text": "Reference image (colour scheme to imitate):"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": mt, "data": data}})
        for v in req["views"]:
            mt, data = _split_data_url(v["image"])
            content.append({"type": "text", "text": f"View: {v['name']}"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": mt, "data": data}})
        content.append({
            "type": "text",
            "text": prompts.user_text(req.get("description", ""), req["filaments"], req["patch_count"], req["mode"], req.get("current")),
        })

        try:
            async with self.client.beta.messages.stream(
                model=self.model,
                max_tokens=32000,
                system=prompts.SYSTEM,
                messages=[{"role": "user", "content": content}],
                thinking={"type": "adaptive"},
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": prompts.output_schema(names)},
                },
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            ) as stream:
                msg = await stream.get_final_message()
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
        if msg.stop_reason == "max_tokens":
            raise ProviderError("Response was cut off; try fewer patches")
        text = next((b.text for b in msg.content if b.type == "text"), "")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ProviderError("Model returned invalid JSON") from e
        plan, default, assignments, detail = _clean(data, names, req["patch_count"])
        usage = {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}
        return PaintResult(plan, default, assignments, detail, msg.model, usage)


class OpenAICompatibleProvider:
    """Any /chat/completions server with vision: Open WebUI, Ollama, llama.cpp, vLLM."""

    id = "local"
    label = "Local"

    def __init__(self):
        self.base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/")
        self.api_key = os.getenv("LOCAL_LLM_API_KEY", "")
        self.model = os.getenv("LOCAL_LLM_MODEL", "")
        self.configured = bool(self.model)

    async def paint(self, req: dict) -> PaintResult:
        if not self.configured:
            raise ProviderError("Set LOCAL_LLM_MODEL (and LOCAL_LLM_API_KEY) in local.env")
        names = [f["name"] for f in req["filaments"]]
        content: list[dict] = []
        if req.get("reference_image"):
            content.append({"type": "text", "text": "Reference image (colour scheme to imitate):"})
            content.append({"type": "image_url", "image_url": {"url": req["reference_image"]}})
        for v in req["views"]:
            content.append({"type": "text", "text": f"View: {v['name']}"})
            content.append({"type": "image_url", "image_url": {"url": v["image"]}})
        content.append({
            "type": "text",
            "text": prompts.user_text(req.get("description", ""), req["filaments"], req["patch_count"], req["mode"], req.get("current"))
            + "\n\nReply with a single JSON object only.",
        })
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": prompts.SYSTEM}, {"role": "user", "content": content}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "paint", "schema": prompts.output_schema(names)},
            },
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            async with httpx.AsyncClient(timeout=600) as client:
                r = await client.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
        except httpx.HTTPError as e:
            raise ProviderError(f"Could not reach local model at {self.base_url}: {e}") from e
        if r.status_code >= 400:
            raise ProviderError(f"Local model error {r.status_code}: {r.text[:300]}")
        resp = r.json()
        text = resp["choices"][0]["message"].get("content") or ""
        m = re.search(r"\{.*\}", text, re.S)  # tolerate prose or code fences around the JSON
        try:
            data = json.loads(m.group(0) if m else text)
        except json.JSONDecodeError as e:
            raise ProviderError("Local model returned invalid JSON") from e
        plan, default, assignments, detail = _clean(data, names, req["patch_count"])
        return PaintResult(plan, default, assignments, detail, resp.get("model", self.model), resp.get("usage") or {})


def providers() -> dict:
    return {p.id: p for p in (AnthropicProvider(), OpenAICompatibleProvider())}
