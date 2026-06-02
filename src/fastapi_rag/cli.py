from pathlib import Path

import typer
from dotenv import load_dotenv

load_dotenv()  # puts HTTPS_PROXY (and others) into os.environ for httpx to pick up
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.table import Table

app = typer.Typer(name="fastapi-rag", add_completion=False, no_args_is_help=True)
console = Console()

_DEFAULT_DB = Path("data/corpus.db")
_DEFAULT_FAISS = Path("data/indexes/docs.faiss")
_DEFAULT_BM25 = Path("data/indexes/docs.bm25")
_DEFAULT_EMBED = "BAAI/bge-small-en-v1.5"
_DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"
_DEFAULT_LLM = "gemini-2.5-flash"


@app.command()
def ingest(
    output: Path = typer.Option(_DEFAULT_DB, "--output", "-o", help="SQLite output path"),
    max_tokens: int = typer.Option(512, "--max-tokens", help="Max tokens per chunk"),
):
    """Fetch FastAPI docs from GitHub and store chunks in SQLite."""
    from .corpus.chunker import chunk_document
    from .corpus.loader import iter_fastapi_docs
    from .corpus.store import clear_source, count_chunks, init_db, save_chunks

    output.parent.mkdir(parents=True, exist_ok=True)
    init_db(output)
    clear_source(output, "docs")

    console.print(f"[bold cyan]Ingesting FastAPI docs[/bold cyan] → {output}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=True,
    ) as progress:
        fetch_task = progress.add_task("Fetching file list from GitHub...", total=None)
        docs = iter_fastapi_docs()
        progress.update(fetch_task, completed=1, total=1, description="File list fetched")

        chunk_task = progress.add_task("Chunking...", total=len(docs))
        for doc_path, url, content in docs:
            chunks = chunk_document(content, "docs", doc_path, url, max_tokens=max_tokens)
            save_chunks(chunks, output)
            progress.advance(chunk_task, 1)

    stats = count_chunks(output)
    console.print(f"[green]Done![/green] Stored chunks: {stats}")


@app.command("ingest-github")
def ingest_github(
    output: Path = typer.Option(_DEFAULT_DB, "--output", "-o", help="SQLite output path"),
    max_pages: int = typer.Option(5, "--max-pages", help="Pages per label (100 issues/page)"),
):
    """Fetch closed GitHub issues from tiangolo/fastapi and store as chunks."""
    from .corpus.github import iter_github_issues
    from .corpus.store import clear_source, count_chunks, init_db, save_chunks

    output.parent.mkdir(parents=True, exist_ok=True)
    init_db(output)
    clear_source(output, "github")

    console.print(f"[bold cyan]Fetching GitHub issues[/bold cyan] (labels: question, bug)")

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console, transient=True) as p:
        task = p.add_task("Fetching issues...", total=None)
        chunks = iter_github_issues(max_pages_per_label=max_pages)
        p.update(task, description=f"Fetched {len(chunks)} issues")

    save_chunks(chunks, output)
    stats = count_chunks(output)
    console.print(f"[green]Done![/green] Stored chunks: {stats}")


@app.command("build-index")
def build_index_cmd(
    db: Path = typer.Option(_DEFAULT_DB, "--db", help="Corpus SQLite path"),
    faiss_out: Path = typer.Option(_DEFAULT_FAISS, "--faiss-out", help="FAISS index output path"),
    bm25_out: Path = typer.Option(_DEFAULT_BM25, "--bm25-out", help="BM25 index output path"),
    model: str = typer.Option(_DEFAULT_EMBED, "--model", help="Sentence-transformers model"),
    batch_size: int = typer.Option(64, "--batch-size", help="Encoding batch size"),
):
    """Build both FAISS (dense) and BM25 (sparse) indexes from the corpus."""
    from .corpus.store import load_chunks
    from .retrieval.bm25 import build_bm25
    from .retrieval.dense import build_index

    console.print("[bold cyan]Loading chunks from DB...[/bold cyan]")
    chunks = load_chunks(db)
    if not chunks:
        console.print("[red]No chunks found. Run 'ingest' first.[/red]")
        raise typer.Exit(1)

    ids = [c.chunk_id for c in chunks]
    texts = [c.content for c in chunks]

    console.print(f"Building FAISS index ({model})...")
    build_index(ids, texts, model, faiss_out, batch_size=batch_size)
    console.print(f"[green]FAISS index → {faiss_out}[/green]")

    console.print("Building BM25 index...")
    build_bm25(ids, texts, bm25_out)
    console.print(f"[green]BM25 index → {bm25_out}[/green]")


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    db: Path = typer.Option(_DEFAULT_DB, "--db", help="Corpus SQLite path"),
    faiss_index: Path = typer.Option(_DEFAULT_FAISS, "--faiss-index"),
    bm25_index: Path = typer.Option(_DEFAULT_BM25, "--bm25-index"),
    embed_model: str = typer.Option(_DEFAULT_EMBED, "--embed-model"),
    reranker_model: str = typer.Option(_DEFAULT_RERANKER, "--reranker-model"),
    llm_model: str = typer.Option(_DEFAULT_LLM, "--llm-model"),
    retrieval_k: int = typer.Option(50, "--retrieval-k", help="Candidates per retriever"),
    reranker_k: int = typer.Option(10, "--reranker-k", help="Chunks sent to LLM"),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Skip reranker"),
    no_generate: bool = typer.Option(False, "--no-generate", help="Show retrieval only"),
):
    """Ask a question — full RAG pipeline: retrieve → rerank → generate."""
    from .config import settings
    from .corpus.store import load_chunks
    from .retrieval.bm25 import BM25Retriever
    from .retrieval.dense import DenseRetriever
    from .retrieval.hybrid import reciprocal_rank_fusion

    if not faiss_index.exists() or not bm25_index.exists():
        console.print("[red]Indexes not found. Run 'build-index' first.[/red]")
        raise typer.Exit(1)

    # ── Retrieval ────────────────────────────────────────────────────────────
    console.print("[dim]Loading indexes...[/dim]")
    dense = DenseRetriever.load(faiss_index, embed_model)
    bm25 = BM25Retriever.load(bm25_index)
    all_chunks = {c.chunk_id: c for c in load_chunks(db)}

    dense_results = dense.search(question, k=retrieval_k)
    bm25_results = bm25.search(question, k=retrieval_k)
    hybrid_results = reciprocal_rank_fusion([dense_results, bm25_results], top_n=retrieval_k)

    # ── Rerank ───────────────────────────────────────────────────────────────
    if no_rerank:
        final_results = hybrid_results[:reranker_k]
    else:
        from .retrieval.reranker import Reranker
        console.print("[dim]Reranking...[/dim]")
        reranker = Reranker(reranker_model)
        chunk_texts = {cid: c.content for cid, c in all_chunks.items()}
        final_results = reranker.rerank(question, hybrid_results, chunk_texts, top_k=reranker_k)

    top_chunks = [all_chunks[r.chunk_id] for r in final_results if r.chunk_id in all_chunks]

    if no_generate:
        table = Table(title=f'"{question}"', show_lines=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Score", width=7)
        table.add_column("Breadcrumb", style="cyan", max_width=40)
        table.add_column("Snippet", max_width=60)
        for i, (r, c) in enumerate(zip(final_results, top_chunks), 1):
            snippet = c.content[:200].replace("\n", " ") + "..."
            table.add_row(str(i), f"{r.score:.4f}", c.breadcrumb, snippet)
        console.print(table)
        return

    # ── Generate ─────────────────────────────────────────────────────────────
    if not settings.google_api_key:
        console.print("[red]GOOGLE_API_KEY not set in .env[/red]")
        raise typer.Exit(1)

    console.print("[dim]Generating answer...[/dim]")
    from .generation.generator import generate_answer
    result = generate_answer(question, top_chunks, model=llm_model, api_key=settings.google_api_key)

    console.print(Rule(f'[bold cyan]{question}[/bold cyan]'))
    console.print(Markdown(result.answer))


@app.command()
def eval(
    db: Path = typer.Option(_DEFAULT_DB, "--db"),
    faiss_index: Path = typer.Option(_DEFAULT_FAISS, "--faiss-index"),
    bm25_index: Path = typer.Option(_DEFAULT_BM25, "--bm25-index"),
    embed_model: str = typer.Option(_DEFAULT_EMBED, "--embed-model"),
    reranker_model: str = typer.Option(_DEFAULT_RERANKER, "--reranker-model"),
    gt: Path = typer.Option(Path("eval_data/ground_truth.jsonl"), "--gt", help="Ground truth file"),
    k: int = typer.Option(10, "--k", help="Eval cutoff (MRR@k, Recall@k)"),
    retrieval_k: int = typer.Option(20, "--retrieval-k", help="Candidates per retriever before reranking"),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Skip reranker (faster)"),
):
    """Run retrieval evaluation: MRR, Recall@k, Hit Rate, latency."""
    from .eval.harness import run_eval

    console.print(f"[bold cyan]Running eval[/bold cyan] on {gt.name} ({k=})")

    report = run_eval(
        db_path=db,
        faiss_path=faiss_index,
        bm25_path=bm25_index,
        embed_model=embed_model,
        gt_path=gt,
        retrieval_k=retrieval_k,
        eval_k=k,
        reranker_model=None if no_rerank else reranker_model,
    )

    table = Table(title=f"Retrieval Evaluation  (n={len(report.results)}, k={k})", show_lines=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right", style="cyan")

    table.add_row("MRR@k",          f"{report.mrr():.4f}")
    table.add_row("Recall@k",       f"{report.mean_recall(k):.4f}")
    table.add_row(f"Hit Rate@{k}",  f"{report.hit_rate(k):.4f}")
    table.add_row("Avg latency",    f"{report.mean_latency_ms():.1f} ms")

    console.print(table)

    # Per-query breakdown
    miss_table = Table(title="Misses (RR=0)", show_lines=True)
    miss_table.add_column("Query", max_width=55)
    miss_table.add_column("Expected URL", max_width=55, style="dim")
    for r in report.results:
        if r.reciprocal_rank == 0:
            miss_table.add_row(r.query, r.relevant_urls[0])

    if miss_table.row_count:
        console.print(miss_table)
    else:
        console.print("[green]No misses — all queries hit in top-k![/green]")


if __name__ == "__main__":
    app()
