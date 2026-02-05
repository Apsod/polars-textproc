use std::collections::VecDeque;
use std::hash::{BuildHasher, Hasher};

use itertools::izip;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::uniform::Uniform;
use rand::prelude::{RngCore, SeedableRng, StdRng};
use rand::Rng;
use regex::Regex;
use serde::Deserialize;
use xxhash_rust::xxh3::{xxh3_128, Xxh3Builder};

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

#[polars_expr(output_type = String)]
fn minhash(inputs: &[Series], kwargs: MinHashKwargs) -> PolarsResult<Series> {
    let tokenizer: Regex = Regex::new(&kwargs.tokenizer_pattern)?;
    let ca: &StringChunked = inputs[0].str()?;

    let hasher = MinHash::from_seed(kwargs.seed, kwargs.buckets, kwargs.bsize, kwargs.window);
    let out = ca.apply_into_string_amortized(|txt: &str, res: &mut String| {
        res.push_str(&hasher.apply_str(tokenizer.find_iter(txt).map(|x| x.as_str())));
    });

    Ok(out.into_series())
}
