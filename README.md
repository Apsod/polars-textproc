# TextProc

This is a polars plugin to enable various standard text processing algorithms in polars, including gopher repetition signals, minhash calculation,
fasttext classifiers, and tokenization.

`polars_textproc.repetition_signals(expr)` applies the gopher repetetition signals to each text in the given `expr` (e.g. a dataframe column).
Returns a struct containing `top_1_gram_char_ratio`, ... `top_4_gram_char_ratio`, `dup_5_gram_char_ratio` ... `dup_10_gram_char_ratio`.
The underlying tokenization can be controlled using the `tokenizer_pattern` kwargs, a regexp which by default is `r"\w+"`.
Note that the pattern is compiled by the rust regex crate, which doesn't match pythons `re` module.

`polars_textproc.fasttext(expr, path, labels)` applies the fasttext model at path to each text in the given `expr` (e.g. a column). By default
it returns a struct with the fields `top_label`, `top_score`, and `total_score`. 
The returned values can be controlled with `output_aggregate` (default: `True`), and `output_scores` (default: `False`). 
With `output_scores=True`, the score for all supplied labels will be returned (with the label as the struct field name). 
With `output_aggregate=False`, `top_label`, `top_score`, and `total_score` will not be returned.

`polars_textproc.minhash(expr, tokenizer_pattern=r"\w+", seed=SEED, buckets=14, bsize=8, window=5)` constructs a hex minhash signature of each text 
given by expr. It produces `window`-shingles of the extracted tokens, as specified by `tokenizer_pattern`, and hashes each shingle into `buckets * bsize`
hashes.
If `bsize>1`, the final minhashes are themselves hashed into 128-bit bucket hashes and returned as a hex encoded `buckets*32`-byte string. 
If `bsize=1`, it returns the raw 64-bit minhashes hex encoded as a `buckets*16`-byte string. 

`polars_textproc.scrub(expr, patterns, replacement="REDACTED")` replaces all matches of the given regex patterns with the replacement string.
Overlapping matches are merged. Regexes use the Rust `regex` crate.

`polars_textproc.compression_ratio(expr, level=6)` returns `original_size / compressed_size` using deflate compression at the given level.

`polars_textproc.compressed_size(expr, level=6)` returns the compressed size in bytes (deflate, excluding the 2-byte zlib header).

`polars_textproc.samplebyte(expr)` returns a random `UInt8` per row (derived from a random 64-bit sample), such that the probability of that byte being exactly x is 2^(-x) for x > 0.

`polars_textproc.uuid4(expr)` returns a random UUID v4 string per row.

`polars_textproc.tokenize(expr, tokenizer)` returns the tokenization of the text in expr, using the supplied tokenizer. 
The tokenizer can be supplied either as a path to a json dump of a `tokenizers.Tokenizer`, or as a `tokenizers.Tokenizer`.

The plugin can also be registered as a namespace using `polars_textproc.register_namespace(name='textproc')`,  
which registers the polars expression namespace `textproc`, and enables calling the function that way,
e.g. `lf.select(pl.col('text').str.to_lowercase().textproc.minhash())`.
