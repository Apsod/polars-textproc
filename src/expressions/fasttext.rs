use std::collections::HashMap;
use std::io::Error;
use std::sync::Arc;
use std::time::Duration;

use cached::proc_macro::cached;
use fasttext::FastText;
use polars::prelude::*;
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use pyo3_polars::derive::polars_expr;
use regex::Regex;
use serde::Deserialize;

#[cached(time = 60, time_refresh = true, sync_writes = "by_key")]
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

#[polars_expr(output_type_func_with_kwargs = fasttext_output)]
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
