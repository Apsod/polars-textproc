use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use regex::{Regex, RegexSet};
use serde::Deserialize;

fn fuse_bounds(
    bounds: impl Iterator<Item = (usize, usize)>,
) -> impl Iterator<Item = (usize, usize)> {
    let mut bounds: Vec<(usize, usize)> = bounds.collect();
    if bounds.is_empty() {
        Vec::new().into_iter()
    } else {
        bounds.sort_unstable_by_key(|k| k.0);

        let mut merged = Vec::with_capacity(bounds.len());
        let mut current_merge = bounds[0];

        for &(next_start, next_stop) in &bounds[1..] {
            if next_start <= current_merge.1 {
                current_merge.1 = current_merge.1.max(next_stop);
            } else {
                merged.push(current_merge);
                current_merge = (next_start, next_stop);
            }
        }
        merged.push(current_merge);
        merged.into_iter()
    }
}

#[derive(Deserialize)]
struct ScrubKwargs {
    patterns: Vec<String>,
    replacement: String,
}

#[polars_expr(output_type = String)]
fn scrub(inputs: &[Series], kwargs: ScrubKwargs) -> PolarsResult<Series> {
    let ca: &StringChunked = inputs[0].str()?;
    let replacement = kwargs.replacement;
    let pattern_set = RegexSet::new(kwargs.patterns)?;
    let patterns: Vec<Regex> = pattern_set
        .patterns()
        .iter()
        .map(|pat| Regex::new(pat).unwrap())
        .collect();

    let out = ca.apply_into_string_amortized(|txt: &str, res: &mut String| {
        let bounds = pattern_set
            .matches(txt)
            .into_iter()
            .map(|index| &patterns[index])
            .flat_map(|pattern| pattern.find_iter(txt).map(move |m| (m.start(), m.end())));

        let mut last_stop = 0;
        for (start, stop) in fuse_bounds(bounds) {
            res.push_str(&txt[last_stop..start]);
            res.push_str(&replacement);
            last_stop = stop;
        }
        res.push_str(&txt[last_stop..]);
    });

    Ok(out.into_series())
}
