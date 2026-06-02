import base64
from pathlib import PurePosixPath

import httpx

from ..config import settings

_API_BASE = "https://api.github.com/repos/tiangolo/fastapi/contents/docs/en/docs"
_DOCS_BASE = "https://fastapi.tiangolo.com"
_DOCS_PREFIX = "docs/en/docs/"


def _headers() -> dict:
    h = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if settings.github_token:
        h["Authorization"] = f"Bearer {settings.github_token}"
    return h


def _doc_path_to_url(doc_path: str) -> str:
    p = PurePosixPath(doc_path)
    if p.name == "index.md":
        slug = "" if str(p.parent) == "." else str(p.parent)
    else:
        slug = str(p.with_suffix(""))
    return f"{_DOCS_BASE}/{slug}/" if slug else f"{_DOCS_BASE}/"


def _list_files(client: httpx.Client, rel_path: str = "") -> list[dict]:
    url = f"{_API_BASE}/{rel_path}".rstrip("/")
    resp = client.get(url, headers=_headers())
    resp.raise_for_status()

    files: list[dict] = []
    for item in resp.json():
        if item["type"] == "file" and item["name"].endswith(".md"):
            files.append(item)
        elif item["type"] == "dir":
            sub_path = item["path"].removeprefix(_DOCS_PREFIX)
            files.extend(_list_files(client, sub_path))
    return files


def _fetch_content(client: httpx.Client, api_url: str) -> str:
    resp = client.get(api_url, headers=_headers())
    resp.raise_for_status()
    return base64.b64decode(resp.json()["content"]).decode("utf-8")


def iter_fastapi_docs() -> list[tuple[str, str, str]]:
    """Return list of (doc_path, canonical_url, markdown_content)."""
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        files = _list_files(client)
        results: list[tuple[str, str, str]] = []
        for f in files:
            doc_path = f["path"].removeprefix(_DOCS_PREFIX)
            url = _doc_path_to_url(doc_path)
            content = _fetch_content(client, f["url"])
            results.append((doc_path, url, content))
    return results
