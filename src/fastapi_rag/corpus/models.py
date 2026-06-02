from dataclasses import dataclass, field


@dataclass
class Chunk:
    source: str       # "docs" | "github" | "stackoverflow"
    doc_path: str     # e.g. "tutorial/path-params.md"
    url: str          # canonical URL
    title: str        # section heading
    breadcrumb: str   # "Tutorial > Path Parameters > Using Path Params"
    content: str      # plain text content
    token_count: int = 0
    chunk_id: str = ""
