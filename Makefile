SHELL=/bin/bash

venv:
	uv venv

install:
	uv run maturin develop

install-release:
	uv run maturin develop --release

pre-commit:
	cargo +nightly fmt --all && cargo clippy --all-features
	uv run python -m ruff check . --fix --exit-non-zero-on-fix
	uv run python -m ruff format polars_textproc tests
	uv run mypy polars_textproc tests

test:
	uv run python -m pytest tests
	.venv/bin/python -m pytest tests

run: install
	uv run run.py

run-release: install-release
	uv run run.py

