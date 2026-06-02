from fastapi_rag.corpus.chunker import chunk_document

_SAMPLE_MD = """\
# FastAPI

## Path Parameters

You can declare path "parameters" with type annotations.

```python
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

## Query Parameters

When you declare function parameters that are not part of the path, they become query parameters.

```python
@app.get("/items/")
async def read_items(skip: int = 0, limit: int = 10):
    return fake_items_db[skip : skip + limit]
```
"""


def test_splits_by_h2():
    chunks = chunk_document(_SAMPLE_MD, "docs", "tutorial/index.md", "https://fastapi.tiangolo.com/tutorial/")
    titles = [c.title for c in chunks]
    assert "Path Parameters" in titles
    assert "Query Parameters" in titles


def test_chunk_ids_are_unique():
    chunks = chunk_document(_SAMPLE_MD, "docs", "tutorial/index.md", "https://fastapi.tiangolo.com/tutorial/")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_token_counts_are_positive():
    chunks = chunk_document(_SAMPLE_MD, "docs", "tutorial/index.md", "https://fastapi.tiangolo.com/tutorial/")
    for c in chunks:
        assert c.token_count > 0


def test_no_chunk_exceeds_max_tokens():
    chunks = chunk_document(_SAMPLE_MD, "docs", "tutorial/index.md", "https://fastapi.tiangolo.com/tutorial/", max_tokens=512)
    for c in chunks:
        assert c.token_count <= 512


def test_breadcrumb_contains_title():
    chunks = chunk_document(_SAMPLE_MD, "docs", "tutorial/index.md", "https://fastapi.tiangolo.com/tutorial/")
    for c in chunks:
        assert c.title in c.breadcrumb
