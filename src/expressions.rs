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
use xxhash_rust::xxh3::{xxh3_128, Xxh3Builder};
use rand::prelude::{StdRng, RngCore, SeedableRng};
use itertools::izip;

use std::hash::{BuildHasher, Hasher};
use std::io::{Error, ErrorKind};

// #######
// Minhash
// #######

const MP_61: u64 = (1<<61) - 1;
const MP_61_128: u128 = MP_61 as u128;

struct MinHash {
    a: Vec<u128>,
    b: Vec<u128>,
    buckets: usize,
    bsize: usize,
    window: usize,
    hash_builder: Xxh3Builder,
}

macro_rules! into_bytes {
    ($x:expr) => {
        $x.into_iter().flat_map(|v| v.to_be_bytes()).collect::<Vec<u8>>()
    }
}


impl MinHash {
    fn from_rng(rng: &mut StdRng, buckets: usize, bsize: usize, window: usize) -> Self {
        let hashes = buckets*bsize;
        let mut a = Vec::with_capacity(hashes);
        let mut b = Vec::with_capacity(hashes);
        for _ in 0..hashes {
            a.push(1 + (rng.next_u64() % (MP_61-1)) as u128);
            b.push((rng.next_u64() % MP_61) as u128);
        }
        MinHash {
            a: a,
            b: b,
            buckets: buckets,
            bsize: bsize,
            window: window,
            hash_builder: Xxh3Builder::new().with_seed(rng.next_u64()),
        }
    }

    fn hashes(&self) -> usize {
        self.buckets * self.bsize
    }

    fn from_seed(seed: [u8; 32], buckets: usize, bsize: usize, window: usize) -> Self{
        Self::from_rng(&mut StdRng::from_seed(seed), buckets, bsize, window)
    }

    fn permute<'a>(&'a self, shingle: u64) -> impl 'a + Iterator<Item=u64> {
        izip!(self.a.iter(), self.b.iter()).map(move |(ai, bi)| {
            ((ai * (shingle as u128) + bi) % MP_61_128) as u64
        })
    }

    fn mk_minhash<'a>(&self, vals: impl Iterator<Item=&'a str>) -> Vec<u64> {
        let mut builder: VecDeque<&str> = VecDeque::with_capacity(self.window+1);
        let mut minhash: Vec<u64> = vec![u64::MAX; self.hashes()];

        for v in vals {
            builder.push_front(v);
            builder.truncate(self.window);
            if builder.len() == self.window {
                let mut hasher = self.hash_builder.build_hasher();
                for v in &builder {
                    hasher.update(v.as_bytes());
                    hasher.write_u8(0xff);
                }
                let shingle_hash = hasher.digest() % MP_61;
                izip!(&mut minhash, self.permute(shingle_hash)).for_each(|(mh, sh)| {
                    *mh = std::cmp::min(*mh, sh)
                });
            }
        }
        minhash
    }

    fn mk_buckets<'a>(&self, vals: impl Iterator<Item=&'a str>) -> Vec<u128> {
        // Take a `bucket * bsize` vector of minhashes, buckets them into
        // `buckets` chunks of size `bsize`, and hash each bucket into a u128 hash.
        // (Should be fine, unless we expect 2^64 different values, which we don't,
        // and saves space for all scenarios where bsize > 1)
        self.mk_minhash(vals).chunks(self.bsize).map(|bucket| xxh3_128(&into_bytes!(bucket))).collect()
    }

    fn apply_str<'a>(&self, vals: impl Iterator<Item=&'a str>) -> String {
        // Construct a hex string representation of the bucket hashes.
        if self.bsize > 1 {
            hex::encode(into_bytes!(self.mk_buckets(vals)))
        } else {
            hex::encode(into_bytes!(self.mk_minhash(vals)))
        }
    }
}

#[derive(Deserialize)]
struct MinHashKwargs{
    tokenizer_pattern: String,
    seed: [u8; 32], 
    buckets: usize,
    bsize: usize,
    window: usize,
}

#[polars_expr(output_type=String)]
fn minhash(inputs: &[Series], kwargs: MinHashKwargs) -> PolarsResult<Series> {
    let tokenizer: Regex = Regex::new(&kwargs.tokenizer_pattern)?;
    let ca: &StringChunked = inputs[0].str()?;

    let hasher = MinHash::from_seed(kwargs.seed, kwargs.buckets, kwargs.bsize, kwargs.window);
    let out = ca.apply_into_string_amortized(|txt: &str, res: &mut String| {
        res.push_str(&hasher.apply_str(tokenizer.find_iter(txt).map(|x| x.as_str())));
    });

    Ok(out.into_series())
}

// #########################
// GOPHER repetition signals
// #########################

fn ratio(num: usize, den:usize) -> f32 {
    ((num as f64) / (den as f64)) as f32
}

fn dup_ngrams_hash<'a>(hash_builder: &Xxh3Builder, num_top: usize, num_dup: usize, vals: impl Iterator<Item = &'a str>) -> Vec<f32>
{
    // Counts duplicate and top ngrams, avoiding overlap for duplicate ngrams. 
    let mut seen : HashSet<u128> = HashSet::new();
    let mut counts : HashMap<u128, usize> = HashMap::new();
    //sbuf tracks the last N seen tokens
    //lbuf tracks the cumulative length of the last N seen tokens.
    let mut sbuf: VecDeque<&str> = VecDeque::with_capacity(num_dup+1);
    let mut lbuf: VecDeque<usize> = VecDeque::with_capacity(num_dup+1);
    // last[n] is the leftmost position of the last duplicate "n"-gram.
    // It is used to avoid double counting overlapping duplicates.
    // dups[n] counts the number of characters covered by duplicate "n"-grams.
    // tot is the total number of characters seen.
    let mut last : Vec<usize> = vec![0; num_dup];
    let mut dups : Vec<usize> = vec![0; num_dup];
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
        
        for (n, (gram, dup)) in izip!(&sbuf, &mut dups).enumerate() {
            lbuf[n] += vlen;
            hasher.update(gram.as_bytes());
            hasher.write_u8(0xff);
            let ngram = hasher.digest128();
            if n < num_top {
                let v = counts.entry(ngram).or_insert(0);
                *v += lbuf[n];
                *dup = std::cmp::max(*dup, *v);
            } else if ! seen.insert(ngram) {
                // unaccounted is the number of n-gram parts (-1) that should be accounted for
                // when updating the number of characters covered by duplicate "n"-grams.
                let unaccounted: usize = std::cmp::min(n, pos - last[n] - 1);
                *dup += lbuf[unaccounted];
                last[n] = pos;
            }
        }
    }
    
    // Hack to deal with division by zero.
    // tot = 0 => all dups = 0.
    let tot = std::cmp::max(1, tot); 
    dups.into_iter().map(|dup| ratio(dup, tot)).collect()
}

fn fieldname(num_top: usize, num_dup: usize, i: usize) -> String {
    if i < num_top {
        format!("top_{}_gram_char_ratio", i+1)
    } else if i < num_dup {
        format!("dup_{}_gram_char_ratio", i+1)
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
            let mut fields : Vec<Field> = Vec::with_capacity(kwargs.num_dup);
            for i in 0..kwargs.num_dup {
                fields.push(
                    Field::new(
                        fieldname(kwargs.num_top, kwargs.num_dup, i).into(),
                        DataType::Float32,
                    )
                );
            }
            Ok(Field::new(
                    "repetition".into(),
                    DataType::Struct(fields.into())
                    )
                )
        }
        dtype => polars_bail!(InvalidOperation: "expected string dtype, got {}", dtype)
    }
}

#[derive(Deserialize)]
struct RepetitionKwargs {
    tokenizer_pattern: String,
    num_top: usize,
    num_dup: usize,
}

#[polars_expr(output_type_func_with_kwargs=repetition_output)]
fn repetition_signals(inputs: &[Series], kwargs: RepetitionKwargs) -> PolarsResult<Series> {
    let tokenizer: Regex = Regex::new(&kwargs.tokenizer_pattern)?;
    let hash_builder = Xxh3Builder::new().with_seed(0x5eed);
    let ca: &StringChunked = inputs[0].str()?;
    
    let mut res: Vec<Vec<f32>> = vec![Vec::with_capacity(ca.len()); kwargs.num_dup];
    let mut validities = MutableBitmap::with_capacity(ca.len());
    validities.extend_constant(ca.len(), true);
    
    ca.iter().enumerate().for_each(|(row, v)| {
        match v.map(|txt| dup_ngrams_hash(&hash_builder, kwargs.num_top, kwargs.num_dup, tokenizer.find_iter(txt).map(|x| x.as_str()))) {
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
        ChunkedArray::<Float32Type>::from_vec_validity(fieldname(kwargs.num_top, kwargs.num_dup, i).into(), v, Some(validities.clone())).into_series()
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
        Ok(FasttextOutput { top_label, top_score, total_score, scores })
    }
}

fn fasttext_output(input_fields: &[Field], kwargs: FasttextKwargs) -> PolarsResult<Field> {
    let field = &input_fields[0];

    let mut fields = Vec::new();

    if kwargs.output_aggregate {
        fields.push(Field::new("top_label".into(), DataType::String));
        fields.push(Field::new("top_score".into(), DataType::Float32));
        fields.push(Field::new("total_score".into(), DataType::Float32));
    }
    if kwargs.output_scores {
        for label in kwargs.labels {
            fields.push(Field::new(label.into(), DataType::Float32));
        }
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
    output_aggregate: bool,
    output_scores: bool,
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

    let mut top_label : Vec<u32> = Vec::new(); 
    let mut top_score : Vec<f32> = Vec::new();
    let mut total_score : Vec<f32> = Vec::new();
    let mut label_scores : Vec<Vec<f32>> = Vec::new();
    
    if kwargs.output_aggregate {
        top_label.reserve_exact(l);
        top_score.reserve_exact(l);
        total_score.reserve_exact(l);
    }

    if kwargs.output_scores {
        for _ in 0..n {
            label_scores.push(Vec::with_capacity(l));
        }
    }

    let space_pattern = Regex::new(r"\s+").unwrap();

    ca.iter().enumerate().for_each(|(row, v)| {
        match v.and_then(|txt| model.predict(&space_pattern.replace_all(txt, " ")).ok()) {
            Some(output) => {
                if kwargs.output_aggregate {
                    top_label.push(output.top_label);
                    top_score.push(output.top_score);
                    total_score.push(output.total_score);
                }
                if kwargs.output_scores {
                    label_scores.iter_mut().zip(output.scores).for_each(|(r, s)| {
                        r.push(s); 
                    });
                }
            },
            None => {
                validities.set(row, false);
                if kwargs.output_aggregate {
                    top_label.push(0);
                    top_score.push(0.0);
                    total_score.push(0.0);
                }
                if kwargs.output_scores {
                    label_scores.iter_mut().for_each(|r| {
                        r.push(0.0);
                    });
                }
            }
        }
    });

    let validities : Bitmap = validities.into();
    let mut res : Vec<Series> = Vec::new();

    if kwargs.output_aggregate {
        res.push(
            ChunkedArray::<UInt32Type>::from_vec_validity("top_label".into(), top_label, Some(validities.clone())).apply_into_string_amortized(
                | index: u32, output: &mut String | {
                    output.push_str(&kwargs.labels[index as usize]);
                }
            ).into_series()
        );
        res.push(
            ChunkedArray::<Float32Type>::from_vec_validity("top_score".into(), top_score, Some(validities.clone())).into_series()
        );
        res.push(
            ChunkedArray::<Float32Type>::from_vec_validity("total_score".into(), total_score, Some(validities.clone())).into_series()
        );
    }
    if kwargs.output_scores {
        for (i, label_score) in label_scores.into_iter().enumerate() {
            res.push(
                ChunkedArray::<Float32Type>::from_vec_validity(kwargs.labels[i].clone().into(), label_score, Some(validities.clone())).into_series()
            )
        }
    }

    StructChunked::from_series(
        inputs[0].name().clone(),
        ca.len(),
        res.iter(),
    ).map(|x| x.into_series())
}
