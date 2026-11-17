//! The phoneme CTC acoustic model, run with ONNX Runtime.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Mutex;

use ort::environment::Environment;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

use crate::frames::Frames;

pub struct PhonemeModel {
    session: Mutex<Session>,
    // Must outlive the session.
    _environment: Environment,
    phoneme_to_id: HashMap<String, usize>,
    pub id_to_phoneme: HashMap<usize, String>,
    /// The CTC blank token.
    pub blank: usize,
}

impl PhonemeModel {
    pub fn load(dir: &Path, threads: usize) -> Result<Self, String> {
        let read = |name: &str| {
            fs::read_to_string(dir.join(name)).map_err(|e| {
                format!(
                    "cannot read {}: {e} (run tools/export_onnx.py to create the model files)",
                    dir.join(name).display()
                )
            })
        };
        let phoneme_to_id: HashMap<String, usize> =
            serde_json::from_str(&read("vocab.json")?).map_err(|e| format!("bad vocab.json: {e}"))?;
        let config: serde_json::Value =
            serde_json::from_str(&read("config.json")?).map_err(|e| format!("bad config.json: {e}"))?;
        let blank = config["pad_token_id"]
            .as_u64()
            .ok_or("config.json has no pad_token_id")? as usize;
        let id_to_phoneme = phoneme_to_id.iter().map(|(p, &i)| (i, p.clone())).collect();

        let model_path = dir.join("model.onnx");
        if !model_path.is_file() {
            return Err(format!(
                "{} not found (run tools/export_onnx.py to create it)",
                model_path.display()
            ));
        }
        let environment = ort::init().build().map_err(|e| format!("onnxruntime: {e}"))?;
        let session = Session::builder(&environment)
            .and_then(|b| Ok(b.with_optimization_level(GraphOptimizationLevel::Level3)?))
            .and_then(|b| Ok(b.with_intra_threads(threads)?))
            .and_then(|mut b| b.commit_from_file(&model_path))
            .map_err(|e| format!("cannot load {}: {e}", model_path.display()))?;

        Ok(Self {
            session: Mutex::new(session),
            _environment: environment,
            phoneme_to_id,
            id_to_phoneme,
            blank,
        })
    }

    /// The ONNX Runtime environment, shared with the speech voices.
    pub fn environment(&self) -> &Environment {
        &self._environment
    }

    /// (frames, vocab) phoneme probabilities for 16 kHz mono audio.
    pub fn posteriors(&self, samples: &[f32]) -> Result<Frames, String> {
        let input = Tensor::from_array(([1, samples.len()], normalize(samples)))
            .map_err(|e| e.to_string())?;

        let mut session = self.session.lock().unwrap_or_else(|e| e.into_inner());
        let outputs = session
            .run(ort::inputs!["input_values" => input])
            .map_err(|e| format!("inference failed: {e}"))?;
        let (shape, logits) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        let vocab = *shape.last().ok_or("model returned a scalar")? as usize;

        let mut probs = logits.to_vec();
        probs.chunks_exact_mut(vocab).for_each(softmax);
        Ok(Frames::new(vocab, probs))
    }

    /// Token ids for the sentence, and the phonemes actually kept per word.
    ///
    /// A phoneme the model does not know is split into known characters where
    /// possible (e.g. a long vowel into vowel + length mark), else dropped.
    pub fn encode(&self, word_phonemes: &[Vec<String>]) -> (Vec<usize>, Vec<Vec<String>>) {
        let mut ids = Vec::new();
        let mut kept = Vec::with_capacity(word_phonemes.len());
        for phonemes in word_phonemes {
            let mut word_kept = Vec::new();
            for phoneme in phonemes {
                if let Some(&id) = self.phoneme_to_id.get(phoneme) {
                    ids.push(id);
                    word_kept.push(phoneme.clone());
                    continue;
                }
                for c in phoneme.chars() {
                    let part = c.to_string();
                    if let Some(&id) = self.phoneme_to_id.get(&part) {
                        ids.push(id);
                        word_kept.push(part);
                    }
                }
            }
            kept.push(word_kept);
        }
        (ids, kept)
    }
}

/// Zero mean and unit variance: the same normalisation as the model's feature extractor.
fn normalize(samples: &[f32]) -> Vec<f32> {
    let n = samples.len().max(1) as f64;
    let mean = samples.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
    let var = samples.iter().map(|&x| (f64::from(x) - mean).powi(2)).sum::<f64>() / n;
    let scale = 1.0 / (var + 1e-7).sqrt();
    samples.iter().map(|&x| ((f64::from(x) - mean) * scale) as f32).collect()
}

fn softmax(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for x in row.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    for x in row.iter_mut() {
        *x /= sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_gives_zero_mean_and_unit_variance() {
        let out = normalize(&[1.0, 2.0, 3.0, 4.0]);
        let mean: f32 = out.iter().sum::<f32>() / 4.0;
        let var: f32 = out.iter().map(|x| x * x).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-6);
        assert!((var - 1.0).abs() < 1e-4);
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let mut row = [1.0, 2.0, 3.0, 1000.0];
        softmax(&mut row);
        assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(row[3] > 0.999);
    }
}
