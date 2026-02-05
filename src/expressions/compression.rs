use std::io::Write;

use flate2::write::DeflateEncoder;
use flate2::Compression;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

struct CountingSink {
    count: usize,
}

impl CountingSink {
    fn new() -> Self {
        Self { count: 0 }
    }
}

impl Write for CountingSink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let l = buf.len();
        self.count += l;
        Ok(l)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[derive(Deserialize)]
struct CompressionRatioKwargs {
    level: u32,
}

#[polars_expr(output_type = Float32)]
fn compression_ratio(inputs: &[Series], kwargs: CompressionRatioKwargs) -> PolarsResult<Series> {
    let ca: &StringChunked = inputs[0].str()?;

    let mut encoder = DeflateEncoder::new(CountingSink::new(), Compression::new(kwargs.level));

    let out = ca.iter().map(|opt_text| {
        opt_text.map(|text| {
            let bytes = text.as_bytes();
            let _ = encoder.write_all(bytes);
            let compressed_size = encoder.reset(CountingSink::new()).unwrap().count;
            let original_size = bytes.len();
            original_size as f32 / (compressed_size - 2) as f32
        })
    });
    Ok(Float32Chunked::from_iter(out).into_series())
}

#[polars_expr(output_type = UInt64)]
fn compressed_size(inputs: &[Series], kwargs: CompressionRatioKwargs) -> PolarsResult<Series> {
    let ca: &StringChunked = inputs[0].str()?;

    let mut encoder = DeflateEncoder::new(CountingSink::new(), Compression::new(kwargs.level));

    let out = ca.iter().map(|opt_text| {
        opt_text.map(|text| {
            let bytes = text.as_bytes();
            let _ = encoder.write_all(bytes);
            (encoder.reset(CountingSink::new()).unwrap().count - 2) as u64
        })
    });
    Ok(UInt64Chunked::from_iter(out).into_series())
}
