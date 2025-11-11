from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List

import polars as pl
from polars.plugins import register_plugin_function

from polars_textproc._internal import __version__ as __version__

if TYPE_CHECKING:
    from polars_textproc.typing import IntoExprColumn

LIB = Path(__file__).parent

SEED = [
    179,
    28,
    18,
    84,
    75,
    144,
    79,
    252,
    138,
    70,
    118,
    68,
    23,
    234,
    55,
    243,
    220,
    195,
    42,
    178,
    73,
    192,
    91,
    161,
    228,
    176,
    67,
    210,
    33,
    75,
    126,
    56,
]


def samplebyte(expr: IntoExprColumn) -> pl.Expr:
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="samplebyte",
        is_elementwise=True,
    )


def uuid4(expr: IntoExprColumn) -> pl.Expr:
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="uuid4",
        is_elementwise=True,
    )


def minhash(
    expr: IntoExprColumn,
    *,
    tokenizer_pattern: str = r"\w+",
    seed=SEED,
    buckets=14,
    bsize=8,
    window=5,
) -> pl.Expr:
    """
    construct a hex representation of the minhash hash of the given text column.
    `tokenizer_pattern`: tokenizer pattern for the word-shingling.
    `seed`: The seed for the hash-permutations.
    `buckets`: Number of minhash buckets.
    `bsize`: Size (in hashes) of each minhash bucket.
    `window`: Shingle window size.

    By default, it creates `buckets * bsize` 64-bit hashes, and then hash each bucket into
    a 128-bit hash, which is then hex encoded as a string, resulting in a `buckets*32` byte
    long minhash signature (With every chunk of 32 bytes being a separate bucket signature).

    Alternatively, if bsize is set to one, it skips the 128-bit hashing, resulting in a
    `buckets*16` byte long minhash signature.

    Suppling buckets > 1 saves space in the case where you've commited to a specific bucketing
    strategy, and the 128-bit hash should be enough to avoid collisions.

    Supplying buckets == 1 could be useful for varying bucket sizes.
    """
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="minhash",
        is_elementwise=True,
        kwargs={
            "tokenizer_pattern": tokenizer_pattern,
            "buckets": buckets,
            "bsize": bsize,
            "seed": seed,
            "window": window,
        },
    )


def repetition_signals(
    expr: IntoExprColumn, *, tokenizer_pattern: str = r"\w+", num_top=4, num_dup=10
) -> pl.Expr:
    """
    Runs gopher repetition signals on the given text column.
    Words are extracted using the supplied tokenizer pattern.
    Computes "top_n_gram_char_ratio"-signals for top_1, top_2, .. top_{num_top}.
    Computes "dup_n_gram_char_ratio"-signals for dup_{num_top+1}, .. dup_{num_dup}.
    """
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="repetition_signals",
        is_elementwise=True,
        kwargs={
            "tokenizer_pattern": tokenizer_pattern,
            "num_top": num_top,
            "num_dup": num_dup,
        },
    )


def scrub(
    expr: IntoExprColumn, *, patterns: List[str], replacement: str = "REDACTED"
) -> pl.Expr:
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="scrub",
        is_elementwise=True,
        kwargs={"patterns": patterns, "replacement": replacement},
    )


def fasttext(
    expr: IntoExprColumn,
    *,
    path: str,
    labels: List[str],
    output_aggregate: bool = True,
    output_scores: bool = False,
) -> pl.Expr:
    """
    Runs a fasttext model against the given text column.
    `path` is the path to the fasttext model bin path.
    `labels` are the labels that should be included in the output.
    output_aggregate=True =>
        output columns:
        `top_label`   : String = top scoring label
        `top_score`   : Float  = score of the top scoring label
        `total_score` : Float  = total score of all given labels
    output_scores=True =>
        output columns:
        `$label` : Float = score of the `$label` (only including given `labels`)
    """
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="fasttext",
        is_elementwise=True,
        kwargs={
            "path": path,
            "labels": labels,
            "output_aggregate": output_aggregate,
            "output_scores": output_scores,
        },
    )
