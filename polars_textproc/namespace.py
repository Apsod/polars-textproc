from __future__ import annotations

from typing import List

import polars as pl
from tokenizers import Tokenizer


@pl.api.register_expr_namespace("textproc")
class TextprocNamespace:
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def compressed_size(self, *, level: int = 6) -> pl.Expr:
        from . import compressed_size

        return compressed_size(self._expr, level=level)

    def compression_ratio(self, *, level: int = 6) -> pl.Expr:
        from . import compression_ratio

        return compression_ratio(self._expr, level=level)

    def samplebyte(self) -> pl.Expr:
        from . import samplebyte

        return samplebyte(self._expr)

    def uuid4(self) -> pl.Expr:
        from . import uuid4

        return uuid4(self._expr)

    def tokenize(self, *, tokenizer: Tokenizer | str) -> pl.Expr:
        from . import tokenize

        return tokenize(self._expr, tokenizer=tokenizer)

    def minhash(
        self,
        *,
        tokenizer_pattern: str = r"\w+",
        seed=None,
        buckets=14,
        bsize=8,
        window=5,
    ) -> pl.Expr:
        from . import SEED, minhash

        if seed is None:
            seed = SEED
        return minhash(
            self._expr,
            tokenizer_pattern=tokenizer_pattern,
            seed=seed,
            buckets=buckets,
            bsize=bsize,
            window=window,
        )

    def repetition_signals(
        self,
        *,
        tokenizer_pattern: str = r"\w+",
        num_top=4,
        num_dup=10,
    ) -> pl.Expr:
        from . import repetition_signals

        return repetition_signals(
            self._expr,
            tokenizer_pattern=tokenizer_pattern,
            num_top=num_top,
            num_dup=num_dup,
        )

    def scrub(
        self,
        *,
        patterns: List[str],
        replacement: str = "REDACTED",
    ) -> pl.Expr:
        from . import scrub

        return scrub(self._expr, patterns=patterns, replacement=replacement)

    def fasttext(
        self,
        *,
        path: str,
        labels: List[str],
        output_aggregate: bool = True,
        output_scores: bool = False,
    ) -> pl.Expr:
        from . import fasttext

        return fasttext(
            self._expr,
            path=path,
            labels=labels,
            output_aggregate=output_aggregate,
            output_scores=output_scores,
        )
