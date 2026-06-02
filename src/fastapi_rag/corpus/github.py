"""Fetch issues from tiangolo/fastapi and convert them to Chunks."""

from __future__ import annotations

import httpx

from ..config import settings
from .models import Chunk
from .chunker import _make_id, _count_tokens

_REPO = "tiangolo/fastapi"
_API = "https://api.github.com"
_ISSUE_URL = "https://github.com/tiangolo/fastapi/issues/{number}"
_LABELS = ["question", "bug"]
_MIN_BODY_LEN = 100   # skip trivial issues
_MAX_COMMENT_LEN = 800
_MAX_COMMENTS = 3


def _headers() -> dict:
    h = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if settings.github_token:
        h["Authorization"] = f"Bearer {settings.github_token}"
    return h


def _fetch_issues(client: httpx.Client, label: str, max_pages: int = 10) -> list[dict]:
    issues = []
    for page in range(1, max_pages + 1):
        resp = client.get(
            f"{_API}/repos/{_REPO}/issues",
            headers=_headers(),
            params={"state": "closed", "labels": label, "per_page": 100, "page": page},
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        # skip pull requests
        issues.extend(i for i in batch if "pull_request" not in i)
    return issues


def _fetch_comments(client: httpx.Client, issue_number: int) -> list[str]:
    resp = client.get(
        f"{_API}/repos/{_REPO}/issues/{issue_number}/comments",
        headers=_headers(),
        params={"per_page": _MAX_COMMENTS},
    )
    resp.raise_for_status()
    return [c["body"] for c in resp.json() if c.get("body")]


def _issue_to_chunk(issue: dict, comments: list[str]) -> Chunk:
    number = issue["number"]
    title = issue["title"].strip()
    body = (issue.get("body") or "").strip()
    url = _ISSUE_URL.format(number=number)

    parts = [f"## {title}", "", body]
    for i, comment in enumerate(comments[:_MAX_COMMENTS], 1):
        truncated = comment[:_MAX_COMMENT_LEN]
        if len(comment) > _MAX_COMMENT_LEN:
            truncated += "..."
        parts.append(f"\n### Answer {i}\n{truncated}")

    content = "\n".join(parts).strip()
    return Chunk(
        source="github",
        doc_path=f"issues/{number}",
        url=url,
        title=title,
        breadcrumb=f"GitHub Issues > {title}",
        content=content,
        token_count=_count_tokens(content),
        chunk_id=_make_id("github", f"issues/{number}", title),
    )


def iter_github_issues(max_pages_per_label: int = 5) -> list[Chunk]:
    """Fetch closed issues (question + bug labels) and return as Chunks."""
    seen: set[int] = set()
    chunks: list[Chunk] = []

    with httpx.Client(timeout=30, follow_redirects=True) as client:
        for label in _LABELS:
            issues = _fetch_issues(client, label, max_pages=max_pages_per_label)
            for issue in issues:
                number = issue["number"]
                if number in seen:
                    continue
                seen.add(number)

                body = (issue.get("body") or "")
                if len(body) < _MIN_BODY_LEN:
                    continue

                comments = _fetch_comments(client, number)
                chunks.append(_issue_to_chunk(issue, comments))

    return chunks
