//! Speech synthesis with Piper voices (VITS models exported to ONNX).
//!
//! A voice is `<name>.onnx` plus `<name>.onnx.json`. The text is turned into
//! espeak IPA, each character mapped to ids through the voice's
//! `phoneme_id_map`, and the model returns the waveform.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Mutex;

use ort::environment::Environment;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use unicode_normalization::UnicodeNormalization;

use crate::config::{Language, LANGUAGES};
use crate::g2p::Phonemizer;

/// Silence between sentences, in seconds.
const SENTENCE_GAP: f32 = 0.25;

struct Voice {
    session: Mutex<Session>,
    sample_rate: u32,
    espeak_voice: String,
    ids: HashMap<char, Vec<i64>>,
    /// noise_scale, length_scale, noise_w
    scales: [f32; 3],
    multi_speaker: bool,
}

/// The voices found in the TTS directory, by language code.
pub struct Tts {
    voices: HashMap<&'static str, Voice>,
}

pub enum TtsError {
    /// No voice for the language.
    Unavailable(String),
    /// The text has nothing to pronounce.
    NothingToSay,
    Internal(String),
}

impl Tts {
    /// Loads every voice that is present. A missing voice only turns speech
    /// off for that language.
    pub fn load(dir: &Path, environment: &Environment, threads: usize) -> Self {
        let mut voices = HashMap::new();
        for language in LANGUAGES {
            match Voice::load(dir, language, environment, threads) {
                Ok(Some(voice)) => {
                    voices.insert(language.code, voice);
                }
                Ok(None) => eprintln!(
                    "note: {}/{}.onnx not found; speech is off for {}",
                    dir.display(),
                    language.piper_voice,
                    language.name
                ),
                Err(e) => eprintln!("warning: {} voice not loaded: {e}", language.name),
            }
        }
        Self { voices }
    }

    pub fn languages(&self) -> Vec<&'static str> {
        let mut codes: Vec<_> = self.voices.keys().copied().collect();
        codes.sort_unstable();
        codes
    }

    /// A 16-bit mono WAV of the text.
    pub fn synthesize(
        &self,
        phonemizer: &Phonemizer,
        language: &str,
        text: &str,
    ) -> Result<Vec<u8>, TtsError> {
        let voice = self
            .voices
            .get(language)
            .ok_or_else(|| TtsError::Unavailable(format!("no voice for language {language:?}")))?;

        let mut samples: Vec<f32> = Vec::new();
        for sentence in split_sentences(text) {
            let ids = voice.phoneme_ids(phonemizer, &sentence).map_err(TtsError::Internal)?;
            if ids.is_empty() {
                continue;
            }
            if !samples.is_empty() {
                let gap = (SENTENCE_GAP * voice.sample_rate as f32) as usize;
                samples.resize(samples.len() + gap, 0.0);
            }
            samples.extend(voice.run(&ids).map_err(TtsError::Internal)?);
        }
        if samples.is_empty() {
            return Err(TtsError::NothingToSay);
        }
        Ok(wav(&samples, voice.sample_rate))
    }
}

impl Voice {
    fn load(
        dir: &Path,
        language: &Language,
        environment: &Environment,
        threads: usize,
    ) -> Result<Option<Self>, String> {
        let model_path = dir.join(format!("{}.onnx", language.piper_voice));
        let config_path = dir.join(format!("{}.onnx.json", language.piper_voice));
        if !model_path.is_file() || !config_path.is_file() {
            return Ok(None);
        }
        let config: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(&config_path)
                .map_err(|e| format!("cannot read {}: {e}", config_path.display()))?,
        )
        .map_err(|e| format!("bad {}: {e}", config_path.display()))?;

        let sample_rate = config["audio"]["sample_rate"]
            .as_u64()
            .ok_or("voice config has no audio.sample_rate")? as u32;
        let espeak_voice = config["espeak"]["voice"]
            .as_str()
            .unwrap_or(language.espeak_voice)
            .to_string();
        let mut ids = HashMap::new();
        for (phoneme, value) in config["phoneme_id_map"]
            .as_object()
            .ok_or("voice config has no phoneme_id_map")?
        {
            let mut chars = phoneme.chars();
            let (Some(c), None) = (chars.next(), chars.next()) else { continue };
            let list: Vec<i64> = value
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_i64()).collect())
                .unwrap_or_default();
            ids.insert(c, list);
        }
        let inference = &config["inference"];
        let scale = |name: &str, default: f64| inference[name].as_f64().unwrap_or(default) as f32;
        let scales =
            [scale("noise_scale", 0.667), scale("length_scale", 1.0), scale("noise_w", 0.8)];
        let multi_speaker = config["num_speakers"].as_u64().unwrap_or(1) > 1;

        let session = Session::builder(environment)
            .and_then(|b| Ok(b.with_optimization_level(GraphOptimizationLevel::Level3)?))
            .and_then(|b| Ok(b.with_intra_threads(threads)?))
            .and_then(|mut b| b.commit_from_file(&model_path))
            .map_err(|e| format!("cannot load {}: {e}", model_path.display()))?;

        Ok(Some(Self {
            session: Mutex::new(session),
            sample_rate,
            espeak_voice,
            ids,
            scales,
            multi_speaker,
        }))
    }

    /// Phoneme ids for one sentence: `^ _ p _ p _ ... $`. The sentence's
    /// punctuation is kept so the voice pauses and rises where it should.
    /// Empty when there is nothing to pronounce.
    fn phoneme_ids(&self, phonemizer: &Phonemizer, sentence: &str) -> Result<Vec<i64>, String> {
        let mut ipa = String::new();
        for (clause, mark) in split_clauses(sentence) {
            if clause.chars().any(|c| c.is_alphanumeric()) {
                let part = phonemizer.phonemize_ipa(&self.espeak_voice, &clause)?;
                if !ipa.is_empty() {
                    ipa.push(' ');
                }
                ipa.push_str(&part.split_whitespace().collect::<Vec<_>>().join(" "));
            }
            if let Some(mark) = mark {
                ipa.push(mark);
            }
        }
        let ipa: String = ipa.nfd().collect();

        let special = |c: char, default: i64| self.ids.get(&c).cloned().unwrap_or_else(|| vec![default]);
        let pad = special('_', 0);
        let mut out = special('^', 1);
        out.extend(&pad);
        let mut spoken = false;
        for c in ipa.chars() {
            // Symbols the voice does not know carry no sound.
            let Some(ids) = self.ids.get(&c) else { continue };
            spoken |= c.is_alphabetic();
            out.extend(ids);
            out.extend(&pad);
        }
        out.extend(special('$', 2));
        Ok(if spoken { out } else { Vec::new() })
    }

    fn run(&self, ids: &[i64]) -> Result<Vec<f32>, String> {
        let n = ids.len();
        let input = Tensor::from_array(([1, n], ids.to_vec())).map_err(|e| e.to_string())?;
        let lengths = Tensor::from_array(([1], vec![n as i64])).map_err(|e| e.to_string())?;
        let scales = Tensor::from_array(([3], self.scales.to_vec())).map_err(|e| e.to_string())?;

        let mut session = self.session.lock().unwrap_or_else(|e| e.into_inner());
        let outputs = if self.multi_speaker {
            let sid = Tensor::from_array(([1], vec![0i64])).map_err(|e| e.to_string())?;
            session.run(ort::inputs![
                "input" => input,
                "input_lengths" => lengths,
                "scales" => scales,
                "sid" => sid
            ])
        } else {
            session.run(ort::inputs![
                "input" => input,
                "input_lengths" => lengths,
                "scales" => scales
            ])
        }
        .map_err(|e| format!("speech synthesis failed: {e}"))?;
        let (_, audio) = outputs[0].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
        Ok(audio.to_vec())
    }
}

/// Splits text into sentences at `.`, `!`, `?` and line breaks.
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    for c in text.chars() {
        if c == '\n' {
            sentences.push(std::mem::take(&mut current));
            continue;
        }
        current.push(c);
        if matches!(c, '.' | '!' | '?') {
            sentences.push(std::mem::take(&mut current));
        }
    }
    sentences.push(current);
    sentences.into_iter().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()
}

/// Splits a sentence at `, ; : . ! ?` into (clause, punctuation mark).
fn split_clauses(sentence: &str) -> Vec<(String, Option<char>)> {
    let mut clauses = Vec::new();
    let mut current = String::new();
    for c in sentence.chars() {
        match c {
            ',' | ';' | ':' | '.' | '!' | '?' => {
                clauses.push((std::mem::take(&mut current), Some(c)));
            }
            // Spanish opening marks add nothing to the sound.
            '¿' | '¡' => {}
            _ => current.push(c),
        }
    }
    if !current.trim().is_empty() {
        clauses.push((current, None));
    }
    clauses
}

/// Peak-normalised 16-bit mono WAV.
fn wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let peak = samples.iter().fold(0.01_f32, |m, s| m.max(s.abs()));
    let scale = 32767.0 / peak;
    let data_len = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        let v = (s * scale).round().clamp(-32768.0, 32767.0) as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_sentences() {
        assert_eq!(
            split_sentences("Hello there. How are you? Fine"),
            ["Hello there.", "How are you?", "Fine"]
        );
        assert_eq!(split_sentences("¿Cómo estás?\nBien."), ["¿Cómo estás?", "Bien."]);
    }

    #[test]
    fn splits_clauses_keeping_marks() {
        assert_eq!(
            split_clauses("Well, I think so."),
            [("Well".to_string(), Some(',')), (" I think so".to_string(), Some('.'))]
        );
        assert_eq!(split_clauses("¿Qué tal?"), [("Qué tal".to_string(), Some('?'))]);
    }

    #[test]
    fn wav_has_a_valid_header() {
        let bytes = wav(&[0.0, 0.5, -0.5], 22050);
        assert_eq!(&bytes[..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(bytes.len(), 44 + 6);
    }
}
