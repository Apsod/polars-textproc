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
    // Counts duplicate and top ngrams using hashing. 
    // Hash collisions will lead to overestimates, but
    // the probability of this is small for sequences of 
    // less than 2**32 tokens. 
    let mut seen : HashSet<String> = HashSet::new();
    let mut counts : [HashMap<String, usize>; L] = core::array::from_fn(|_| HashMap::new());
    // hashers and lens are "circular" buffers.
    let mut sbuf : [&str; N] = [""; N];
    let mut lbuf : [usize; N] = [0; N];
    // last[i] is the leftmost position of the last duplicate i-gram. Used to not over
    // lbs, dups and tot count the total number of characters. 
    let mut last : [usize; N] = [0; N];
    let mut dups : [usize; N] = [0; N];
    let mut tot: usize = 0;

    for (pos, v) in vals.enumerate() {
        let vlen = v.chars().count();
        // n: saturated length of buffer.
        // ix: current 0-index in buffer.
        // Corresponds to the new unigram.
        let n = std::cmp::min(N, pos+1);
        let ix = pos % N;

        tot += vlen;
        lbuf[ix] = 0;
        sbuf[ix] = v;

        let mut s = String::with_capacity(vlen + n + lbuf[(ix + n - 1) % N]);
        for i in 0..n {
            // j : index corresponding to the i-gram.
            // j = (ix - 1) % N
            // the stuff below is due to underflow.
            let j = (ix + i*(N-1)) % N;

            lbuf[j] += vlen;
            s.push_str(sbuf[j]);
            s.push(' ');

            if i < L {
                *counts[i].entry(s.clone()).or_insert(0) += lbuf[j]; 
            } else {
                if ! seen.insert(s.clone()) {
                    let unaccounted : usize = std::cmp::min(i, pos - last[i] - 1);
                    dups[i] += lbuf[(ix + unaccounted*(N-1)) % N];
                    last[i] = pos;
                }
            }
        }
    }
    
    let counts = counts.map(|c| {c.into_values().max().unwrap_or(0)});

    for i in 0..L {
        dups[i] = counts[i];
    }

    dups.map(|dup| ratio(dup, tot))
}

fn fieldname(i: usize) -> String {
    format!("{}_{}_gram_char_ratio", if i < L {"top"} else {"dup"}, i+1)
}

fn ngrammer_output(input_fields: &[Field]) -> PolarsResult<Field> {
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

#[polars_expr(output_type_func=ngrammer_output)]
fn repetition_signals(inputs: &[Series]) -> PolarsResult<Series> {
    let wordsplit: Regex = Regex::new(r"\s+")?;
    let ca: &StringChunked = inputs[0].str()?;

    let mut res : [Vec<f32>; N] = core::array::from_fn(|_| Vec::with_capacity(ca.len()));
    let mut validities = MutableBitmap::with_capacity(ca.len());
    validities.extend_constant(ca.len(), true);
    
    ca.iter().enumerate().for_each(|(row, v)| {
        match v.map(|txt| dup_ngrams_str(wordsplit.split(txt))) {
            Some(r) => {
                for (i, s) in r.into_iter().enumerate() {
                    res[i].push(s);
                }
            }
            None => {
                for i in 0..N {
                    validities.set(row, false);
                    res[i].push(0.0);
                }
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

// ######################################
// Language identification using fasttext
// ######################################

#[cached(time=60, time_refresh=true, sync_writes = true)]
fn load_model(path: String) -> Result<Arc<FastText>, String> {
    let mut model = FastText::new();
    model.load_model(&path)?;
    Ok(Arc::new(model))
}

struct FasttextModel {
    model : Arc<FastText>,
    labels : HashMap<String, usize>, 
}

impl FasttextModel {
    fn new(path: &str, labels: &Vec<String>) -> Result<Self, String> {
        let m = load_model(path.into())?;
        Ok(
            Self {
                model: m,
                labels: HashMap::from_iter(labels.iter().enumerate().map(|(i,s)| (s.clone(), i)))
            }
        )
    }

    fn len(self: &Self) -> usize {
        self.labels.len()
    }

    fn predict(self: &Self, txt: &str) -> Result<Vec<f32>, String> {
        let preds = self.model.predict(txt, -1, 0.0)?;
        let mut ret : Vec<f32> = vec![0.0; self.labels.len()];

        for p in preds.into_iter() {
            match self.labels.get(&p.label) {
                Some(i) => ret[*i] = p.prob,
                None => (),
            }
        }
        Ok(ret)
    }
}


fn fasttext_output(input_fields: &[Field], kwargs: FasttextKwargs) -> PolarsResult<Field> {
    Ok(
        Field::new(
            "langid".into(),
            DataType::Struct(
                kwargs.labels.iter().map(|l| Field::new(l.into(), DataType::Float32)).collect::<Vec<_>>()
            )
        )
    )
}


#[derive(Deserialize)]
struct FasttextKwargs{
    path: String,
    labels: Vec<String>,
}

impl FasttextKwargs {
    fn load(self: &Self) -> Result<FasttextModel, Error> {
        FasttextModel::new(&self.path, &self.labels).map_err(|e| std::io::Error::new(ErrorKind::Other, e))
    }
}

#[polars_expr(output_type_func_with_kwargs=fasttext_output)]
fn fasttext(inputs: &[Series], kwargs: FasttextKwargs) -> PolarsResult<Series> {
    let ca: &StringChunked = inputs[0].str()?;
    let model = kwargs.load()?;
    
    let mut validities = MutableBitmap::with_capacity(ca.len());
    validities.extend_constant(ca.len(), true);

    let n = model.len();
    let mut ret : Vec<Vec<f32>> = Vec::new();
    for _ in 0..n {
        ret.push(Vec::with_capacity(ca.len()));
    }

    let space_pattern = Regex::new(r"\s+").unwrap();

    ca.iter().enumerate().for_each(|(row, v)| {
        match v.and_then(|txt| model.predict(&space_pattern.replace_all(txt, " ")).ok()) {
            Some(scores) => {
                for i in 0..n {
                    ret[i].push(scores[i]);
                }
            },
            None => {
                validities.set(row, false);
                for i in 0..n {
                    ret[i].push(0.0);
                }
            }
        }
    });

    let validities : Bitmap = validities.into();
    let res : Vec<Series> = ret.into_iter().enumerate().map(|(i, v)| {
        ChunkedArray::<Float32Type>::from_vec_validity(kwargs.labels[i].clone().into(), v, Some(validities.clone())).into_series()
    }).collect();

    StructChunked::from_series(
        inputs[0].name().clone(),
        ca.len(),
        res.iter(),
    ).map(|x| x.into_series())
}
