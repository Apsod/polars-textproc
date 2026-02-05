use itertools::izip;
use polars::prelude::*;
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use pyo3_polars::derive::polars_expr;
use regex::Regex;
use serde::Deserialize;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{BuildHasher, Hasher};
use xxhash_rust::xxh3::Xxh3Builder;

fn ratio(num: usize, den: usize) -> f32 {
    ((num as f64) / (den as f64)) as f32
}

fn dup_ngrams_hash<'a>(
    hash_builder: &Xxh3Builder,
    num_top: usize,
    num_dup: usize,
    vals: impl Iterator<Item = &'a str>,
) -> Vec<f32> {
    // Counts duplicate and top ngrams, avoiding overlap for duplicate ngrams.
    let mut seen: HashSet<u128> = HashSet::new();
    let mut counts: HashMap<u128, usize> = HashMap::new();
    //sbuf tracks the last N seen tokens
    //lbuf tracks the cumulative length of the last N seen tokens.
    let mut sbuf: VecDeque<&str> = VecDeque::with_capacity(num_dup + 1);
    let mut lbuf: VecDeque<usize> = VecDeque::with_capacity(num_dup + 1);
    // last[n] is the leftmost position of the last duplicate "n"-gram.
    // It is used to avoid double counting overlapping duplicates.
    // dups[n] counts the number of characters covered by duplicate "n"-grams.
    // tot is the total number of characters seen.
    let last: &mut [usize] = &mut vec![0; num_dup];
    let dups: &mut [usize] = &mut vec![0; num_dup];
    let mut tot: usize = 0;

    for (pos, v) in vals.enumerate() {
        let vlen = v.chars().count();
        lbuf.push_front(0);
        sbuf.push_front(v);
        lbuf.truncate(num_dup);
        sbuf.truncate(num_dup);
        tot += vlen;
        let mut hasher = hash_builder.build_hasher();
        // s : string buffer where we put the n-gram parts.
        // The ngram is built up in reverse, iterating over the deques:
        // Say we've seen [the, cat, sat, on, the], and the current word is "mat", for N=4, L=2.
        // pos = 5
        // i = 1
        // sbuf = [mat, the, on, sat]
        for (n, gram, dup) in izip!(0..sbuf.len(), &sbuf, &mut *dups) {
            lbuf[n] += vlen;
            hasher.update(gram.as_bytes());
            hasher.write_u8(0xff);
            let ngram = hasher.digest128();
            if n < num_top {
                let v = counts.entry(ngram).or_insert(0);
                *v += lbuf[n];
                *dup = std::cmp::max(*dup, *v);
            } else if !seen.insert(ngram) {
                // unaccounted is the number of n-gram parts (-1) that should be accounted for
                // when updating the number of characters covered by duplicate "n"-grams.
                // For example:
                // pos = 12
                // n = 3
                // last[3] = 10, i.e. we observed a repeated 4(!)-gram at position 10.
                // unaccouned = min(3, 12 - 10 - 1): 1
                // lbuf[unaccounted] = lbuf[1], i.e. the length of the rightmost
                // two-gram (corresponding to positions 11, 12)
                let unaccounted: usize = std::cmp::min(n, pos - last[n] - 1);
                *dup += lbuf[unaccounted];
                last[n] = pos;
            }
        }
    }

    // Hack to deal with division by zero.
    // tot = 0 => all dups = 0.
    let tot = std::cmp::max(1, tot);
    dups.iter().map(|dup| ratio(*dup, tot)).collect()
}

fn fieldname(num_top: usize, num_dup: usize, i: usize) -> String {
    if i < num_top {
        format!("top_{}_gram_char_ratio", i + 1)
    } else if i < num_dup {
        format!("dup_{}_gram_char_ratio", i + 1)
    } else {
        panic!("field {} larger than {}", i, num_dup)
    }
}

fn repetition_output(input_fields: &[Field], kwargs: RepetitionKwargs) -> PolarsResult<Field> {
    let field = &input_fields[0];

    if kwargs.num_top > kwargs.num_dup {
        polars_bail!(InvalidOperation: "num top must be not be greater than num dup, got {} > {}", kwargs.num_top, kwargs.num_dup)
    }

    match field.dtype() {
        DataType::String => {
            let mut fields: Vec<Field> = Vec::with_capacity(kwargs.num_dup);
            for i in 0..kwargs.num_dup {
                fields.push(Field::new(
                    fieldname(kwargs.num_top, kwargs.num_dup, i).into(),
                    DataType::Float32,
                ));
            }
            Ok(Field::new("repetition".into(), DataType::Struct(fields)))
        },
        dtype => polars_bail!(InvalidOperation: "expected string dtype, got {}", dtype),
    }
}

#[derive(Deserialize)]
struct RepetitionKwargs {
    tokenizer_pattern: String,
    num_top: usize,
    num_dup: usize,
}

#[polars_expr(output_type_func_with_kwargs = repetition_output)]
fn repetition_signals(inputs: &[Series], kwargs: RepetitionKwargs) -> PolarsResult<Series> {
    let tokenizer: Regex = Regex::new(&kwargs.tokenizer_pattern)?;
    let hash_builder = Xxh3Builder::new().with_seed(0x5eed);
    let ca: &StringChunked = inputs[0].str()?;

    let mut res: Vec<Vec<f32>> = vec![Vec::with_capacity(ca.len()); kwargs.num_dup];
    let mut validities = MutableBitmap::with_capacity(ca.len());
    validities.extend_constant(ca.len(), true);

    ca.iter().enumerate().for_each(|(row, v)| {
        match v.map(|txt| {
            dup_ngrams_hash(
                &hash_builder,
                kwargs.num_top,
                kwargs.num_dup,
                tokenizer.find_iter(txt).map(|x| x.as_str()),
            )
        }) {
            Some(signals) => {
                res.iter_mut().zip(signals).for_each(|(r, s)| r.push(s));
            },
            None => {
                validities.set(row, false);
                res.iter_mut().for_each(|r| r.push(0.0));
            },
        }
    });

    let validities: Bitmap = validities.into();
    let res: Vec<Series> = res
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            ChunkedArray::<Float32Type>::from_vec_validity(
                fieldname(kwargs.num_top, kwargs.num_dup, i).into(),
                v,
                Some(validities.clone()),
            )
            .into_series()
        })
        .collect();

    StructChunked::from_series(inputs[0].name().clone(), ca.len(), res.iter())
        .map(|x| x.into_series())
}
