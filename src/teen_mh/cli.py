"""Typer CLI: ``teen-mh <subcommand>``."""

from __future__ import annotations

import logging

import typer
from rich.logging import RichHandler

from teen_mh.config import load_config
from teen_mh.data import download_dataset, load_splits
from teen_mh.eda import run_eda
from teen_mh.evaluate import run_evaluation
from teen_mh.train import run_training
from teen_mh.tune import run_tuning

app = typer.Typer(help="Teen Mental Health ML pipeline", add_completion=False)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_time=True, show_path=False)],
    )


@app.callback()
def _main() -> None:
    _configure_logging()


@app.command()
def ingest(force: bool = typer.Option(False, help="Re-download even if raw file exists.")) -> None:
    """Download raw dataset and materialise train/test parquet splits."""
    cfg = load_config()
    cfg.ensure_dirs()
    download_dataset(cfg, force=force)
    load_splits(cfg)  # persists splits on first call


@app.command()
def eda() -> None:
    """Run deep-dive EDA and write PNGs + summary JSON."""
    cfg = load_config()
    cfg.ensure_dirs()
    run_eda(cfg)


@app.command()
def train() -> None:
    """Benchmark candidate models with CV and persist the winning baseline."""
    cfg = load_config()
    cfg.ensure_dirs()
    run_training(cfg)


@app.command()
def tune() -> None:
    """Tune the benchmark winner with Optuna and persist the final model."""
    cfg = load_config()
    cfg.ensure_dirs()
    run_tuning(cfg)


@app.command()
def evaluate() -> None:
    """Evaluate the final model on the held-out test set."""
    cfg = load_config()
    cfg.ensure_dirs()
    run_evaluation(cfg)


@app.command("all")
def run_all() -> None:
    """Run the entire pipeline end-to-end."""
    cfg = load_config()
    cfg.ensure_dirs()
    download_dataset(cfg)
    load_splits(cfg)
    run_eda(cfg)
    run_training(cfg)
    run_tuning(cfg)
    run_evaluation(cfg)


if __name__ == "__main__":
    app()
