use std::io::Error;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use cached::proc_macro::cached;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use tokenizers::tokenizer::Tokenizer;

#[cached(time = 60, time_refresh = true, sync_writes = "by_key")]
fn tok_from_file(path: String) -> Result<Arc<Tokenizer>, String> {
    let tok = Tokenizer::from_file(&path)
        .map_err(|_| format!("Error loading tokenizer from path: {}", &path))?;
    Ok(Arc::new(tok))
}

#[cached(time = 60, time_refresh = true, sync_writes = "by_key")]
fn tok_from_str(payload: String) -> Result<Arc<Tokenizer>, String> {
    let tok = Tokenizer::from_str(&payload)
        .map_err(|_| format!("Error loading tokenizer from string: {}", &payload))?;
    Ok(Arc::new(tok))
}

#[derive(Deserialize)]
struct Kwargs {
    payload: String,
    is_path: bool,
}

impl Kwargs {
    fn load(self) -> Result<Arc<Tokenizer>, Error> {
        let res = if self.is_path {
            tok_from_file(self.payload)
        } else {
            tok_from_str(self.payload)
        };
        res.map_err(Error::other)
    }
}

fn outtype(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = Field::new(
        input_fields[0].name.clone(),
        DataType::List(Box::new(DataType::UInt32)),
    );
    Ok(field.clone())
}

#[polars_expr(output_type_func = outtype)]
fn tokenize(inputs: &[Series], kwargs: Kwargs) -> PolarsResult<Series> {
    let ca: &StringChunked = inputs[0].str()?;

    let tokenizer = kwargs.load()?;

    let mut builder =
        ListPrimitiveChunkedBuilder::<UInt32Type>::new(ca.name().clone(), 0, 0, DataType::UInt32);

    for opt in ca {
        match opt {
            Some(text) => {
                builder.append_slice(tokenizer.encode_fast(text, false).unwrap().get_ids())
            },
            None => builder.append_null(),
        }
    }
    Ok(builder.finish().into_series())
    //Ok(UInt32Chunked::from_iter(out).into_series())
}
