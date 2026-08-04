import asyncio

import httpx

from .config import Settings
from .models import Source


SYSTEM_PROMPT = """You answer questions using only the supplied research-paper excerpts.
If the excerpts do not contain enough evidence, say so clearly. Do not use outside facts.
Cite factual statements inline using [1], [2], etc. The numbers correspond exactly to the
ordered excerpts. Give a concise synthesis instead of copying long passages."""


def _context(sources: list[Source]) -> str:
    return "\n\n".join(
        f"[{i}] Paper: {source.paper}; page: {source.page}; section: {source.section_name}\n{source.text}"
        for i, source in enumerate(sources, start=1)
    )


async def generate_answer(question: str, sources: list[Source], settings: Settings) -> tuple[str, str]:
    if not sources:
        return "No papers have been indexed yet, or no relevant passage was found.", "none"

    prompt = f"{SYSTEM_PROMPT}\n\nEXCERPTS\n{_context(sources)}\n\nQUESTION\n{question}"
    providers = (
        [("groq", lambda: _groq(prompt, settings))]
        if settings.llm_provider.lower() == "groq"
        else [("gemini", lambda: _gemini(prompt, settings))]
    )
    if settings.gemini_api_key and providers[0][0] != "gemini":
        providers.append(("gemini", lambda: _gemini(prompt, settings)))
    if settings.groq_api_key and providers[0][0] != "groq":
        providers.append(("groq", lambda: _groq(prompt, settings)))

    for provider, call in providers:
        key_exists = settings.gemini_api_key if provider == "gemini" else settings.groq_api_key
        if not key_exists:
            continue
        for attempt in range(3):
            try:
                return await call(), provider
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in {429, 500, 502, 503, 504}:
                    break
            except httpx.RequestError:
                pass
            if attempt < 2:
                await asyncio.sleep(0.5 * (2**attempt))

    fallback = "\n\n".join(
        f"[{i}] {source.text}" for i, source in enumerate(sources[:3], start=1)
    )
    return (
        "The answer-generation service is unavailable. Here are the most relevant retrieved passages:\n\n" + fallback,
        "extractive",
    )


async def _gemini(prompt: str, settings: Settings) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.gemini_model}:generateContent"
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            url,
            params={"key": settings.gemini_api_key},
            json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.1}},
        )
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]


async def _groq(prompt: str, settings: Settings) -> str:
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            json={
                "model": settings.groq_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
