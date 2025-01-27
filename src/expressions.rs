#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use std::fmt::Write;
use std::collections::{HashSet, HashMap, VecDeque};
use std::hash::{DefaultHasher, BuildHasher, RandomState, Hasher, Hash};
use std::cmp::Eq;
use regex::Regex;

const L: usize = 4;
const N: usize = 10;

fn ratio(num: usize, den:usize) -> f32 {
    ((num as f64) / (den as f64)) as f32
}


fn fieldname(i: usize) -> String {
    format!("{}_{}_gram_char_ratio", if i < L {"top"} else {"dup"}, i+1)
}

fn dup_ngrams<'a, BH: BuildHasher>(builder: &BH, vals: impl Iterator<Item = &'a str>) -> [f32; N] 
{
    // Counts duplicate and top ngrams using hashing. 
    // Hash collisions will lead to overestimates, but
    // the probability of this is small for sequences of 
    // less than 2**32 tokens. 
    let mut seen : HashSet<(usize, u64)> = HashSet::new();
    let mut counts : [HashMap<u64, usize>; L] = core::array::from_fn(|_| HashMap::new());
    // hashers and lens are "circular" buffers.
    let mut hashers : [BH::Hasher; N] = core::array::from_fn(|_| builder.build_hasher());
    let mut lens : [usize; N] = [0; N];
    // lbs, dups and tot count the total number of characters. 
    let mut lbs : [usize; N] = core::array::from_fn(|i| i);
    let mut dups : [usize; N] = [0; N];
    let mut tot: usize = 0;

    for (ix, v) in vals.enumerate() {
        let vlen = v.chars().count();
        // n: saturated length of buffer.
        // ix: current 0-index in buffer.
        // Corresponds to the new unigram.
        let n = std::cmp::min(N, ix+1);
        let ix = ix % N;

        tot += vlen;
        // Overwrite the length and hasher corresponding to the 1-length sequence.
        lens[ix] = 0;
        hashers[ix] = builder.build_hasher();
        hashers[ix].write(b"\x8e\xd6\xaf\xbd"); //salt
        
        //update all lens (adding the new token length to all cumulative lens)
        for l in &mut lens { 
            *l += vlen;
        }

        //update all hashes (adding the new token to all hashers)
        for h in &mut hashers {
            h.write(v.as_bytes());
            h.write_u8(0xff);
        }

        for i in 0..n {
            // j : index corresponding to the i-gram.
            // j = (ix - 1) % N
            // the stuff below is due to underflow.
            let j = (ix + i*(N-1)) % N;
            let hash = hashers[j].finish();
            if i < L {
                *counts[i].entry(hash).or_insert(0) += lens[j]; 
                //.and_modify(|c| {
                //    *c += lens[j]
                //});
            } else {
                if ! seen.insert((i, hash)) {
                    dups[i] += lens[(ix + lbs[i]*(N-1)) % N];
                    lbs[i] = 0;
                } else {
                    lbs[i] = std::cmp::min(i, lbs[i] + 1); 
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

fn ngrammer<BH: BuildHasher>(builder: &BH, wordsplit: &Regex, value: Option<&str>, res: &mut[Vec<Option<f32>>; N]) -> ()
{
    match value {
        Some(txt) => {
            for (i, s) in dup_ngrams::<BH>(builder, wordsplit.split(txt)).into_iter().enumerate() {
                res[i].push(Some(s));
            }
        }
        None => {
            for i in 0..N {
                res[i].push(None);
            }
        }
    }
}

#[polars_expr(output_type_func=ngrammer_output)]
fn repetition_signals(inputs: &[Series]) -> PolarsResult<Series> {
    let wordsplit: Regex = Regex::new(r"\s+")?;
    let ca: &StringChunked = inputs[0].str()?;
    let builder = RandomState::new();

    let mut res : [Vec<Option<f32>>; N] = core::array::from_fn(|i| Vec::with_capacity(ca.len()));

    ca.iter().for_each(|v| ngrammer::<RandomState>(&builder, &wordsplit, v, &mut res));

    let ss : Vec<Series> = res.iter().enumerate().map(|(i, v)| Series::new(fieldname(i).into(), v)).collect();

    StructChunked::from_series(
        inputs[0].name().clone(),
        ca.len(),
        ss.iter(),
        ).map(|x| x.into_series())

    //let out : StructChunked = ca.iter().map(|v| ngrammer::<RandomState>(&builder, &wordsplit, v)).collect_ca(PlSmallStr::EMPTY);
    //Ok(out.into_series())
}


