# ccstuff

Polars plugins to apply gopher repetetition penalties and fasttext classifiers to text data.

`polars_textproc.repetition_signals(expr)` applies the gopher repetetition signals to each text in the given `expr` (e.g. a dataframe column).
Returns a struct containing `top_1_gram_char_ratio`, ... `top_4_gram_char_ratio`, `dup_5_gram_char_ratio` ... `dup_10_gram_char_ratio`.
The underlying tokenization can be controlled using the `tokenizer_pattern` kwargs, a regexp which by default is `r"\w+"`.
Note that the pattern is compiled by the rust regex crate, which doesn't match pythons `re` module.

`polars_textproc.fasttext(expr, path, labels)` applies the fasttext model at path to each text in the given `expr` (e.g. a column). By default
it returns a struct with the fields `top_label`, `top_score`, and `total_score`. 
The returned values can be controlled with `output_aggregate` (default: `True`), and `output_scores` (default: `False`). 
With `output_scores=True`, the score for all supplied labels will be returned (with the label as the struct field name). 
With `output_aggregate=False`, `top_label`, `top_score`, and `total_score` will not be returned.
