use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::Rng;
use uuid::Uuid;

#[polars_expr(output_type = UInt8)]
fn samplebyte(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    let n = s.len();
    let mut rng = rand::rng();
    let mut builder = PrimitiveChunkedBuilder::<UInt8Type>::new(s.name().clone(), n);
    for _ in 0..n {
        let v = loop {
            let v = rng.next_u64();
            if v != 0 {
                break (v.leading_zeros() + 1) as u8;
            }
        };
        builder.append_value(v);
    }
    Ok(builder.finish().into_series())
}

#[polars_expr(output_type = String)]
fn uuid4(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    let n = s.len();
    let mut builder = StringChunkedBuilder::new(s.name().clone(), n);
    for _ in 0..n {
        builder.append_value(Uuid::new_v4().simple().to_string());
    }
    Ok(builder.finish().into_series())
}
