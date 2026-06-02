import hashlib
import re

import tiktoken

from .models import Chunk

_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _make_id(source: str, doc_path: str, key: str) -> str:
    raw = f"{source}:{doc_path}:{key}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _parse_sections(markdown: str) -> list[tuple[int, str, str]]:
    """Return list of (heading_level, heading_title, body_text)."""
    lines = markdown.splitlines()
    sections: list[tuple[int, str, list[str]]] = []
    current_level = 0
    current_title = ""
    current_body: list[str] = []

    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            if current_title or current_body:
                sections.append((current_level, current_title, current_body))
            current_level = len(m.group(1))
            # Strip MkDocs anchor syntax: "Title { #anchor-id }"
            current_title = re.sub(r"\s*\{[^}]*\}\s*$", "", m.group(2)).strip()
            current_body = []
        else:
            current_body.append(line)

    if current_title or current_body:
        sections.append((current_level, current_title, current_body))

    return [(lvl, title, "\n".join(body).strip()) for lvl, title, body in sections]


def _split_paragraphs(
    text: str,
    source: str,
    doc_path: str,
    url: str,
    title: str,
    breadcrumb: str,
    max_tokens: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    paragraphs = re.split(r"\n{2,}", text)
    buffer: list[str] = []
    buffer_tokens = 0
    part = 0

    def _flush():
        nonlocal buffer, buffer_tokens, part
        body = "\n\n".join(buffer).strip()
        if not body:
            return
        part_title = title if part == 0 else f"{title} (part {part + 1})"
        chunks.append(
            Chunk(
                source=source,
                doc_path=doc_path,
                url=url,
                title=part_title,
                breadcrumb=breadcrumb,
                content=body,
                token_count=_count_tokens(body),
                chunk_id=_make_id(source, doc_path, f"{title}_{part}"),
            )
        )
        buffer, buffer_tokens, part = [], 0, part + 1

    for para in paragraphs:
        para_tokens = _count_tokens(para)
        if buffer_tokens + para_tokens > max_tokens and buffer:
            _flush()
        buffer.append(para)
        buffer_tokens += para_tokens

    _flush()
    return chunks


def chunk_document(
    content: str,
    source: str,
    doc_path: str,
    url: str,
    max_tokens: int = 512,
) -> list[Chunk]:
    sections = _parse_sections(content)
    chunks: list[Chunk] = []
    breadcrumb_map: dict[int, str] = {}

    for level, title, body in sections:
        breadcrumb_map[level] = title
        for lvl in list(breadcrumb_map):
            if lvl > level:
                del breadcrumb_map[lvl]

        if not body:
            continue

        breadcrumb = " > ".join(breadcrumb_map[l] for l in sorted(breadcrumb_map))
        token_count = _count_tokens(body)

        if token_count <= max_tokens:
            chunks.append(
                Chunk(
                    source=source,
                    doc_path=doc_path,
                    url=url,
                    title=title,
                    breadcrumb=breadcrumb,
                    content=body,
                    token_count=token_count,
                    chunk_id=_make_id(source, doc_path, title),
                )
            )
        else:
            chunks.extend(
                _split_paragraphs(body, source, doc_path, url, title, breadcrumb, max_tokens)
            )

    return chunks
