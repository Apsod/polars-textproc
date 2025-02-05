#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use std::collections::{HashSet, HashMap, VecDeque};
use regex::Regex;
use fasttext::{FastText};
use cached::proc_macro::cached;
use serde::Deserialize;
use std::sync::Arc;
use std::rc::Rc;
use std::cell::RefCell;

use std::io::{Error, ErrorKind};
use std::collections::hash_map::Entry::{Occupied, Vacant};

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
    let mut counts : HashMap<String, usize> = HashMap::new();
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
        let filled = std::cmp::min(N, pos+1);
        let i = pos % N;

        tot += vlen;
        lbuf[i] = 0;
        sbuf[i] = v;
        
        let mut s = String::with_capacity(vlen + filled + lbuf[(i + filled - 1) % N]);
        // s : string buffer where we put the n-gram parts.
        // n : zero-indexed n-gram, so n=0 ~ unigram, n=1 ~ bigrams, et.c.
        // j : index corresponding to the current n-gram.
        // j = (i - n) % N
        for n in 0..filled {
            // the stuff below is due to underflow.
            let j = (i + n*(N-1)) % N;

            lbuf[j] += vlen;
            s.push_str(sbuf[j]);
            let ngram = s.clone();
            s.push(' ');

            if n < L {
                let v = counts.entry(ngram).or_insert(0);
                *v += lbuf[j];
                dups[n] = std::cmp::max(dups[n], *v);
            } else if ! seen.insert(ngram) {
                let unaccounted : usize = std::cmp::min(n, pos - last[n] - 1);
                dups[n] += lbuf[(i + unaccounted*(N-1)) % N];
                last[n] = pos;
            }
        }
    }
    
    let tot = std::cmp::max(1, tot); 
    //let counts = counts.map(|c| {c.into_values().max().unwrap_or(0)});
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
        let mut ret : Vec<f32> = vec![0.0; self.len()];
        
        preds.into_iter().for_each(|p| {
            self.labels.get(&p.label).map(|i| ret[*i] = p.prob);
        });

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
    let mut ret : Vec<Vec<f32>> = vec![Vec::with_capacity(ca.len()); n];

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
