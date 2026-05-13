"""
Download the karma benchmark corpus: legal, finance, health.

All sources are public and freely accessible. We download a small, curated
set per domain — enough to demonstrate the framework, small enough to fit
on a laptop and to iterate quickly during development.

Usage:
    python scripts/download_corpus.py
    python scripts/download_corpus.py --domain legal
    python scripts/download_corpus.py --max-per-domain 5
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(add_completion=False)
console = Console()

CORPUS_ROOT = Path(__file__).resolve().parent.parent / "benchmarks" / "corpus"
USER_AGENT = "ParetoCorpusDownloader/0.1 (https://github.com/Engineer-coding/pareto)"


@dataclass
class CorpusItem:
    """One downloadable item in the corpus."""

    domain: str  # legal | finance | health
    filename: str
    url: str
    description: str


# A small, curated, public set. URLs are stable as of 2026.
CORPUS: list[CorpusItem] = [
    # ── LEGAL ──────────────────────────────────────────────────────────────
    CorpusItem(
        domain="legal",
        filename="gdpr_full_text.html",
        url="https://gdpr-info.eu/",
        description="GDPR — General Data Protection Regulation (EU) full text",
    ),
    CorpusItem(
        domain="legal",
        filename="ccpa_full_text.html",
        url="https://oag.ca.gov/privacy/ccpa",
        description="CCPA — California Consumer Privacy Act overview",
    ),
    CorpusItem(
        domain="legal",
        filename="mit_license.txt",
        url="https://opensource.org/license/mit",
        description="MIT License canonical text",
    ),
    CorpusItem(
        domain="legal",
        filename="apache_2_license.txt",
        url="https://www.apache.org/licenses/LICENSE-2.0.txt",
        description="Apache License 2.0 canonical text",
    ),
    # ── FINANCE ────────────────────────────────────────────────────────────
    CorpusItem(
        domain="finance",
        filename="apple_10k_2024.html",
        url="https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193&type=10-K&dateb=&owner=include&count=40",
        description="Apple Inc. — SEC 10-K filing index (annual report)",
    ),
    CorpusItem(
        domain="finance",
        filename="microsoft_10k_2024.html",
        url="https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000789019&type=10-K&dateb=&owner=include&count=40",
        description="Microsoft Corp. — SEC 10-K filing index",
    ),
    CorpusItem(
        domain="finance",
        filename="basel3_summary.html",
        url="https://www.bis.org/bcbs/basel3.htm",
        description="Basel III — banking capital regulation summary",
    ),
    # ── HEALTH ─────────────────────────────────────────────────────────────
    CorpusItem(
        domain="health",
        filename="who_covid_overview.html",
        url="https://www.who.int/health-topics/coronavirus",
        description="WHO — COVID-19 health topic overview",
    ),
    CorpusItem(
        domain="health",
        filename="who_diabetes.html",
        url="https://www.who.int/news-room/fact-sheets/detail/diabetes",
        description="WHO — Diabetes fact sheet",
    ),
    CorpusItem(
        domain="health",
        filename="cdc_hypertension.html",
        url="https://www.cdc.gov/high-blood-pressure/about/index.html",
        description="CDC — High blood pressure (hypertension) fact sheet",
    ),
]


def _download_one(item: CorpusItem, dest_dir: Path, timeout: int = 30) -> tuple[bool, str]:
    """Download a single item. Returns (success, message)."""
    dest = dest_dir / item.filename
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        return True, f"already exists ({dest.stat().st_size:,} bytes)"

    req = Request(item.url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if not data:
            return False, "empty response"
        dest.write_bytes(data)
        return True, f"downloaded ({len(data):,} bytes)"
    except URLError as e:
        return False, f"URL error: {e.reason}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


@app.command()
def main(
    domain: str | None = typer.Option(
        None, "--domain", "-d", help="Only download a specific domain (legal/finance/health)."
    ),
    max_per_domain: int | None = typer.Option(
        None, "--max-per-domain", "-n", help="Cap the number of items per domain."
    ),
    delay: float = typer.Option(
        0.5, "--delay", help="Seconds to wait between downloads (polite)."
    ),
) -> None:
    """Download the karma corpus into benchmarks/corpus/{domain}/."""
    selected = [c for c in CORPUS if domain is None or c.domain == domain]

    if max_per_domain is not None:
        # Group by domain, take first N of each
        per_domain: dict[str, list[CorpusItem]] = {}
        for item in selected:
            per_domain.setdefault(item.domain, []).append(item)
        selected = [it for items in per_domain.values() for it in items[:max_per_domain]]

    if not selected:
        console.print("[yellow]Nothing to download (check --domain).[/yellow]")
        raise typer.Exit(1)

    console.print(f"[bold]Downloading {len(selected)} items to {CORPUS_ROOT}[/bold]")

    success = 0
    failed: list[tuple[CorpusItem, str]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        for item in selected:
            task = progress.add_task(f"  [{item.domain}] {item.filename}", total=None)
            ok, msg = _download_one(item, CORPUS_ROOT / item.domain)
            progress.remove_task(task)
            tag = "[green]OK   [/green]" if ok else "[red]FAIL [/red]"
            console.print(f"  {tag} [{item.domain:7}] {item.filename:<35} — {msg}")
            if ok:
                success += 1
            else:
                failed.append((item, msg))
            time.sleep(delay)

    console.print()
    console.print(f"[bold green]{success}[/bold green] downloaded, "
                  f"[bold red]{len(failed)}[/bold red] failed.")

    if failed:
        console.print("\n[yellow]Failures (you may need to download these manually):[/yellow]")
        for item, msg in failed:
            console.print(f"  • {item.filename}: {msg}")
            console.print(f"    URL: {item.url}")


if __name__ == "__main__":
    app()