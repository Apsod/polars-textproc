#![allow(clippy::unused_unit)]
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{BuildHasher, Hasher};
use std::io::Error;
use std::sync::Arc;

use cached::proc_macro::cached;
use fasttext::FastText;
use itertools::izip;
use polars::prelude::*;
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use pyo3_polars::derive::polars_expr;
use rand::distr::uniform::Uniform;
use rand::distr::StandardUniform;
use rand::prelude::{RngCore, SeedableRng, StdRng};
use rand::Rng;
use regex::{Regex, RegexSet};
use serde::Deserialize;
use uuid::Uuid;
use xxhash_rust::xxh3::{xxh3_128, Xxh3Builder};

// ##########
// SampleByte
// ##########

#[polars_expr(output_type=UInt8)]
fn samplebyte(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    let n = s.len();
    let mut rng = rand::rng();
    let mut builder = PrimitiveChunkedBuilder::<UInt8Type>::new(s.name().clone(), n);
    for _ in 0..n {
        let val: u64 = rng.sample(StandardUniform);
        if val.is_power_of_two() {
            builder.append_value((val.leading_zeros() + 1) as u8);
        } else {
            builder.append_value(val.leading_zeros() as u8);
        }
    }
    Ok(builder.finish().into_series())
}

// ####
// UUID
// ####

#[polars_expr(output_type=String)]
fn uuid4(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    let n = s.len();
    let mut builder = StringChunkedBuilder::new(s.name().clone(), n);
    for _ in 0..n {
        builder.append_value(Uuid::new_v4().simple().to_string());
    }
    Ok(builder.finish().into_series())
}

// #######
// Minhash
// #######

//const HM: u64 = (1 << 32) - 1;
const MP: u64 = (1 << 61) - 1;
const MP_128: u128 = MP as u128;

fn mod61(x: u64) -> u64 {
    let y = (x & MP) + (x >> 61);
    if y < MP {
        y
    } else {
        y - MP
    }
}

fn mod61_128(x: u128) -> u64 {
    let y = ((x & MP_128) + (x >> 61)) as u64;
    if y < MP {
        y
    } else {
        y - MP
    }
}

fn affine61(a: u64, b: u64, x: u64) -> u64 {
    let y = (a as u128) * (x as u128) + (b as u128);
    mod61_128(y)
}

struct MinHash {
    a: Vec<u64>,
    b: Vec<u64>,
    buckets: usize,
    bsize: usize,
    window: usize,
    hash_builder: Xxh3Builder,
}

macro_rules! into_bytes {
    ($x:expr) => {
        $x.into_iter()
            .flat_map(|v| v.to_be_bytes())
            .collect::<Vec<u8>>()
    };
}

impl MinHash {
    fn from_rng(rng: &mut StdRng, buckets: usize, bsize: usize, window: usize) -> Self {
        let hashes = buckets * bsize;
        let mut a = Vec::with_capacity(hashes);
        let mut b = Vec::with_capacity(hashes);
        let a_dist: Uniform<u64> = Uniform::new(1, MP).unwrap();
        let b_dist: Uniform<u64> = Uniform::new(0, MP).unwrap();
        for _ in 0..hashes {
            a.push(rng.sample(a_dist));
            b.push(rng.sample(b_dist));
        }
        let hash_builder = Xxh3Builder::new().with_seed(rng.next_u64());
        MinHash {
            a,
            b,
            buckets,
            bsize,
            window,
            hash_builder,
        }
    }

    fn hashes(&self) -> usize {
        self.buckets * self.bsize
    }

    fn from_seed(seed: [u8; 32], buckets: usize, bsize: usize, window: usize) -> Self {
        Self::from_rng(&mut StdRng::from_seed(seed), buckets, bsize, window)
    }

    fn mk_minhash<'a>(&self, vals: impl Iterator<Item = &'a str>) -> Vec<u64> {
        let mut builder: VecDeque<&str> = VecDeque::with_capacity(self.window + 1);
        let minhash: &mut [u64] = &mut vec![u64::MAX; self.hashes()][..];
        //let mut minhash: Vec<u64> = vec![u64::MAX; self.hashes()];
        vals.filter_map(|w| {
            builder.push_front(w);
            builder.truncate(self.window);
            if builder.len() == self.window {
                let mut hasher = self.hash_builder.build_hasher();
                for v in &builder {
                    hasher.update(v.as_bytes());
                    hasher.write_u8(0xff);
                }
                Some(mod61(hasher.digest()))
            } else {
                None
            }
        })
        .for_each(|shingle| {
            izip!(minhash.iter_mut(), &self.a, &self.b)
                .for_each(|(mh, a, b)| *mh = std::cmp::min(*mh, affine61(*a, *b, shingle)));
        });
        minhash.to_vec()
    }

    fn mk_buckets<'a>(&self, vals: impl Iterator<Item = &'a str>) -> Vec<u128> {
        // Take a `bucket * bsize` vector of minhashes, buckets them into
        // `buckets` chunks of size `bsize`, and hash each bucket into a u128 hash.
        // (Should be fine, unless we expect 2^64 different values, which we don't,
        // and saves space for all scenarios where bsize > 1)
        self.mk_minhash(vals)
            .chunks(self.bsize)
            .map(|bucket| xxh3_128(&into_bytes!(bucket)))
            .collect()
    }

    fn apply_str<'a>(&self, vals: impl Iterator<Item = &'a str>) -> String {
        // Construct a hex string representation of the bucket hashes.
        if self.bsize > 1 {
            hex::encode(into_bytes!(self.mk_buckets(vals)))
        } else {
            hex::encode(into_bytes!(self.mk_minhash(vals)))
        }
    }
}

#[derive(Deserialize)]
struct MinHashKwargs {
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

#[polars_expr(output_type_func_with_kwargs=repetition_output)]
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

// ###############
// Regexp scrubber
// ###############

//fn fuse_bounds(bounds: impl Iterator<Item=(usize, usize)>) -> impl Iterator<Item=(usize, usize)> {
//    bounds.fold(BTreeMap::new(), |mut acc, (start, stop)| {
//        let mut middle = acc.split_off(&start);
//        let mut tail = middle.split_off(&(stop+1));
//        // acc:    Tree with keys in (..., start)
//        // middle: Tree with keys in [start, stop]
//        // tail:   Tree with (start, ...)
//        // acc always contain non-overlapping spans.
//
//        let mut start = start;
//        let mut stop = stop;
//
//        acc.last_entry().map(|entry| {
//            if *entry.get() >= start {
//                let (other_start, other_stop) = entry.remove_entry();
//                start = other_start;
//                stop = stop.max(other_stop);
//            }
//        });
//
//        if let Some(entry) = middle.last_entry() {
//            stop = stop.max(*entry.get());
//        }
//
//        acc.insert(start, stop);
//        acc.append(&mut tail);
//        acc
//    }).into_iter()
//}

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

#[polars_expr(output_type=String)]
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

// #################
// Fasttext labeling
// #################

#[cached(time = 60, time_refresh = true, sync_writes = true)]
fn load_model(path: String) -> Result<Arc<FastText>, String> {
    let mut model = FastText::new();
    model.load_model(&path)?;
    Ok(Arc::new(model))
}

struct FasttextModel {
    model: Arc<FastText>,
    labelmap: HashMap<String, usize>,
}

struct FasttextOutput {
    top_label: u32,
    top_score: f32,
    total_score: f32,
    scores: Vec<f32>,
}

impl FasttextModel {
    fn new(path: &str, labels: &[String]) -> Result<Self, String> {
        let m = load_model(path.into())?;
        Ok(Self {
            model: m,
            labelmap: HashMap::from_iter(labels.iter().enumerate().map(|(i, s)| (s.clone(), i))),
        })
    }

    fn len(&self) -> usize {
        self.labelmap.len()
    }

    fn predict(&self, txt: &str) -> Result<FasttextOutput, String> {
        let preds = self.model.predict(txt, -1, 0.0)?;
        let mut scores: Vec<f32> = vec![0.0; self.len()];
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
        Ok(FasttextOutput {
            top_label,
            top_score,
            total_score,
            scores,
        })
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
        DataType::String => Ok(Field::new("langid".into(), DataType::Struct(fields))),
        dtype => polars_bail!(InvalidOperation: "expected string dtype, got {}", dtype),
    }
}

#[derive(Deserialize)]
struct FasttextKwargs {
    path: String,
    labels: Vec<String>,
    output_aggregate: bool,
    output_scores: bool,
}

impl FasttextKwargs {
    fn load(&self) -> Result<FasttextModel, Error> {
        FasttextModel::new(&self.path, &self.labels).map_err(std::io::Error::other)
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

    let mut top_label: Vec<u32> = Vec::new();
    let mut top_score: Vec<f32> = Vec::new();
    let mut total_score: Vec<f32> = Vec::new();
    let mut label_scores: Vec<Vec<f32>> = Vec::new();

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
                    label_scores
                        .iter_mut()
                        .zip(output.scores)
                        .for_each(|(r, s)| {
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
            },
        }
    });

    let validities: Bitmap = validities.into();
    let mut res: Vec<Series> = Vec::new();

    if kwargs.output_aggregate {
        res.push(
            ChunkedArray::<UInt32Type>::from_vec_validity(
                "top_label".into(),
                top_label,
                Some(validities.clone()),
            )
            .apply_into_string_amortized(|index: u32, output: &mut String| {
                output.push_str(&kwargs.labels[index as usize]);
            })
            .into_series(),
        );
        res.push(
            ChunkedArray::<Float32Type>::from_vec_validity(
                "top_score".into(),
                top_score,
                Some(validities.clone()),
            )
            .into_series(),
        );
        res.push(
            ChunkedArray::<Float32Type>::from_vec_validity(
                "total_score".into(),
                total_score,
                Some(validities.clone()),
            )
            .into_series(),
        );
    }
    if kwargs.output_scores {
        for (i, label_score) in label_scores.into_iter().enumerate() {
            res.push(
                ChunkedArray::<Float32Type>::from_vec_validity(
                    kwargs.labels[i].clone().into(),
                    label_score,
                    Some(validities.clone()),
                )
                .into_series(),
            )
        }
    }

    StructChunked::from_series(inputs[0].name().clone(), ca.len(), res.iter())
        .map(|x| x.into_series())
}
