from __future__ import annotations

from dataclasses import dataclass

from google import genai
from google.genai import types

from ..corpus.models import Chunk

_SYSTEM = """\
You are a helpful assistant that answers questions about FastAPI using only the provided documentation excerpts.

Rules:
- Answer concisely and accurately based strictly on the provided context.
- Cite sources inline using [1], [2], etc. corresponding to the numbered excerpts.
- If the context does not contain enough information, say so clearly.
- Include code examples from the context when relevant.
- At the end, list the cited sources with their URLs."""

_CONTEXT_TEMPLATE = "[{i}] {breadcrumb}\nURL: {url}\n\n{content}"

_USER_TEMPLATE = """\
Context excerpts:
{context}

Question: {question}"""


@dataclass
class GenerationResult:
    answer: str
    sources: list[Chunk]


def generate_answer(
    question: str,
    chunks: list[Chunk],
    model: str = "gemini-2.5-flash",
    api_key: str = "",
) -> GenerationResult:
    client = genai.Client(api_key=api_key)

    context = "\n\n---\n\n".join(
        _CONTEXT_TEMPLATE.format(i=i + 1, breadcrumb=c.breadcrumb, url=c.url, content=c.content)
        for i, c in enumerate(chunks)
    )

    response = client.models.generate_content(
        model=model,
        config=types.GenerateContentConfig(system_instruction=_SYSTEM),
        contents=_USER_TEMPLATE.format(context=context, question=question),
    )

    return GenerationResult(answer=response.text, sources=chunks)
