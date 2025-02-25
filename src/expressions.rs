#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use std::collections::{HashSet, HashMap};
use regex::Regex;
use fasttext::{FastText};
use cached::proc_macro::cached;
use serde::Deserialize;
use std::sync::Arc;

use std::io::{Error, ErrorKind};

// #########################
// GOPHER repetition signals
// #########################

const L: usize = 4;
const N: usize = 10;

fn ratio(num: usize, den:usize) -> f32 {
    ((num as f64) / (den as f64)) as f32
}

fn dup_ngrams_str<'a>(vals: impl Iterator<Item = &'a str>) -> [f32; N] 
{
    // Counts duplicate and top ngrams, avoiding overlap for duplicate ngrams. 
    let mut seen : HashSet<String> = HashSet::new();
    let mut counts : HashMap<String, usize> = HashMap::new();
    //sbuf and lbuf are circular buffers
    //sbuf tracks the last N seen tokens
    //lbuf tracks the cumulative length of the last N seen tokens.
    let mut sbuf : [&str; N] = [""; N];
    let mut lbuf : [usize; N] = [0; N];
    // last[n] is the leftmost position of the last duplicate "n"-gram.
    // It is used to avoid double counting overlapping duplicates.
    // dups[n] counts the number of characters covered by duplicate "n"-grams.
    // tot is the total number of characters seen.
    let mut last : [usize; N] = [0; N];
    let mut dups : [usize; N] = [0; N];
    let mut tot: usize = 0;

    for (pos, v) in vals.enumerate() {
        let vlen = v.chars().count();
        let filled = std::cmp::min(N, pos+1);
        let i = pos % N;

        tot += vlen;
        lbuf[i] = 0;
        sbuf[i] = v;
        
        let mut s = String::with_capacity(vlen + filled + lbuf[(i + filled - 1) % N]);
        // s : string buffer where we put the n-gram parts.
        // n : zero-indexed n-gram "n", so n=0 ~ unigram, n=1 ~ bigrams, et.c.
        // j : index corresponding to the current n-gram in the circular buffers lbuf, sbuf.
        // The ngram is built up in reverse, iterating over the circular buffer:
        // Say we've seen [the, cat, sat, on, the], and the current word is "mat", for N=4, L=2.
        // pos = 5
        // i = 1
        // sbuf = [the, mat, sat, on]
        // n = 0: j = 1, lbuf[1] =  3, ngram = "mat"
        // n = 1: j = 0, lbuf[0] =  6, ngram = "mat the"
        // n = 2: j = 3, lbuf[3] =  8, ngram = "mat the on"
        // n = 3: j = 2, lbuf[2] = 11, ngram = "mat the on sat"
        for n in 0..filled {
            // this corresponds to j = (i - n) % N.
            // the formulation below is to avoid underflow.
            let j = (i + n*(N-1)) % N;

            lbuf[j] += vlen;
            s.push_str(sbuf[j]);
            let ngram = s.clone();
            s.push(' ');

            if n < L {
                // top-ngram
                // Increment counts[ngram] by the ngram length, 
                // And set dups[n] to counts[ngram] if it's larger
                // than the current dups[n].
                let v = counts.entry(ngram).or_insert(0);
                *v += lbuf[j];
                dups[n] = std::cmp::max(dups[n], *v);
            } else if ! seen.insert(ngram) {
                // dup-ngram
                // last[n] is the position where a duplicate "n"-gram was last observed.
                // unaccounted is the number of n-gram parts (-1) that should be accounted for
                // when updating the number of characters covered by duplicate "n"-grams.
                // lbuf[(i - unaccounted) % N] corresponds the the cumulative length of
                // the (unaccounted + 1) most recent tokens.
                let unaccounted : usize = std::cmp::min(n, pos - last[n] - 1);
                dups[n] += lbuf[(i + unaccounted*(N-1)) % N];
                last[n] = pos;
            }
        }
    }
    
    // Hack to deal with division by zero.
    // tot = 0 => all dups = 0.
    let tot = std::cmp::max(1, tot); 
    dups.map(|dup| ratio(dup, tot))
}

fn fieldname(i: usize) -> String {
    format!("{}_{}_gram_char_ratio", if i < L {"top"} else {"dup"}, i+1)
}

fn ngrammer_output(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];

    match field.dtype() {
        DataType::String => {
            let fields : [Field; N] = core::array::from_fn(|i| {
                Field::new(
                    fieldname(i).into(),
                    DataType::Float32,
                    ) 
            });
            Ok(Field::new(
                    "repetition".into(), 
                    DataType::Struct(fields.into())
                    ))
        }
        dtype => polars_bail!(InvalidOperation: "expected string dtype, got {}", dtype)
    }
}

#[polars_expr(output_type_func=ngrammer_output)]
fn repetition_signals(inputs: &[Series]) -> PolarsResult<Series> {
    let wordsplit: Regex = Regex::new(r"\s+")?;
    let ca: &StringChunked = inputs[0].str()?;

    let mut res : [Vec<f32>; N] = core::array::from_fn(|_| Vec::with_capacity(ca.len()));
    let mut validities = MutableBitmap::with_capacity(ca.len());
    validities.extend_constant(ca.len(), true);
    
    ca.iter().enumerate().for_each(|(row, v)| {
        match v.map(|txt| dup_ngrams_str(wordsplit.split(txt))) {
            Some(signals) => {
                res.iter_mut().zip(signals).for_each(|(r, s)| r.push(s));
            }
            None => {
                validities.set(row, false); 
                res.iter_mut().for_each(|r| r.push(0.0));
            }
        }
    });

    let validities : Bitmap = validities.into();
    let res : Vec<Series> = res.into_iter().enumerate().map(|(i, v)| {
        ChunkedArray::<Float32Type>::from_vec_validity(fieldname(i).into(), v, Some(validities.clone())).into_series()
    }).collect();

    StructChunked::from_series(
        inputs[0].name().clone(),
        ca.len(),
        res.iter(),
        ).map(|x| x.into_series())
}

// #################
// Fasttext labeling
// #################

#[cached(time=60, time_refresh=true, sync_writes = true)]
fn load_model(path: String) -> Result<Arc<FastText>, String> {
    let mut model = FastText::new();
    model.load_model(&path)?;
    Ok(Arc::new(model))
}

struct FasttextModel {
    model : Arc<FastText>,
    labelmap : HashMap<String, usize>, 
}

struct FasttextOutput {
    top_label: u32,
    top_score: f32,
    total_score: f32,
    scores: Vec<f32>
}

impl FasttextModel {
    fn new(path: &str, labels: &[String]) -> Result<Self, String> {
        let m = load_model(path.into())?;
        Ok(
            Self {
                model: m,
                labelmap: HashMap::from_iter(labels.iter().enumerate().map(|(i,s)| (s.clone(), i))),
            }
        )
    }

    fn len(&self) -> usize {
        self.labelmap.len()
    }

    fn predict(&self, txt: &str) -> Result<FasttextOutput, String> {
        let preds = self.model.predict(txt, -1, 0.0)?;
        let mut scores : Vec<f32> = vec![0.0; self.len()];
        let mut top_label = 0;
        let mut top_score = 0.0;
        let mut total_score = 0.0;
        
        preds.into_iter().for_each(|p| {
            if let Some(i) = self.labelmap.get(&p.label) {
                let i = *i;
                scores[i] = p.prob;
                total_score += p.prob;
                if p.prob > top_score {
                    top_label = i as u32;
                    top_score = p.prob;
                }
            }
        });
        Ok(FasttextOutput { top_label: top_label, top_score: top_score, total_score: total_score, scores: scores })
    }
}

fn fasttext_output(input_fields: &[Field], kwargs: FasttextKwargs) -> PolarsResult<Field> {
    let field = &input_fields[0];
    let mut fields = vec![
        Field::new("top_label".into(), DataType::String),
        Field::new("top_score".into(), DataType::Float32),
        Field::new("total_score".into(), DataType::Float32),
    ];
    
    for label in kwargs.labels {
        fields.push(Field::new(label.into(), DataType::Float32));
    }

    match field.dtype() {
        DataType::String => {
            Ok(
                Field::new(
                    "langid".into(),
                    DataType::Struct(
                        fields
                    )
                )
            )
        }
        dtype => polars_bail!(InvalidOperation: "expected string dtype, got {}", dtype)
    }
}

#[derive(Deserialize)]
struct FasttextKwargs{
    path: String,
    labels: Vec<String>,
}

impl FasttextKwargs {
    fn load(&self) -> Result<FasttextModel, Error> {
        FasttextModel::new(&self.path, &self.labels).map_err(|e| std::io::Error::new(ErrorKind::Other, e))
    }
}

#[polars_expr(output_type_func_with_kwargs=fasttext_output)]
fn fasttext(inputs: &[Series], kwargs: FasttextKwargs) -> PolarsResult<Series> {
    let ca: &StringChunked = inputs[0].str()?;
    let model = kwargs.load()?;
    let l = ca.len();
    let n = model.len();
    
    let mut validities = MutableBitmap::with_capacity(l);
    validities.extend_constant(ca.len(), true);

    let mut top_labels : Vec<u32> = Vec::with_capacity(l); 
    let mut top_scores : Vec<f32> = Vec::with_capacity(l); 
    let mut total_scores : Vec<f32> = Vec::with_capacity(l); 
    let mut label_scores : Vec<Vec<f32>> = vec![Vec::with_capacity(l); n];

    let space_pattern = Regex::new(r"\s+").unwrap();

    ca.iter().enumerate().for_each(|(row, v)| {
        match v.and_then(|txt| model.predict(&space_pattern.replace_all(txt, " ")).ok()) {
            Some(output) => {
                top_labels.push(output.top_label);
                top_scores.push(output.top_score);
                total_scores.push(output.total_score);
                label_scores.iter_mut().zip(output.scores).for_each(|(r, s)| {
                    r.push(s); 
                });
            },
            None => {
                validities.set(row, false);
                top_labels.push(0);
                top_scores.push(0.0);
                total_scores.push(0.0);
                label_scores.iter_mut().for_each(|r| {
                    r.push(0.0);
                });
            }
        }
    });

    let validities : Bitmap = validities.into();
    let mut res : Vec<Series> = Vec::new();
    res.push(
        ChunkedArray::<UInt32Type>::from_vec_validity("top_label".into(), top_labels, Some(validities.clone())).apply_into_string_amortized(
            | index: u32, output: &mut String | {
                output.push_str(&kwargs.labels[index as usize]);
            }
        ).into_series()
    );
    res.push(
        ChunkedArray::<Float32Type>::from_vec_validity("top_score".into(), top_scores, Some(validities.clone())).into_series()
    );
    res.push(
        ChunkedArray::<Float32Type>::from_vec_validity("total_score".into(), total_scores, Some(validities.clone())).into_series()
    );
    for (i, label_score) in label_scores.into_iter().enumerate() {
        res.push(
            ChunkedArray::<Float32Type>::from_vec_validity(kwargs.labels[i].clone().into(), label_score, Some(validities.clone())).into_series()
        )
    }

    StructChunked::from_series(
        inputs[0].name().clone(),
        ca.len(),
        res.iter(),
    ).map(|x| x.into_series())
}
