//! Turn an alignment into phoneme, word and fluency scores.

use std::collections::HashMap;

use serde::Serialize;

use crate::align::Segment;
use crate::config::{Config, FRAME_SECONDS};
use crate::frames::Frames;

/// Below this much non-blank probability mass in a segment, the model heard
/// close to nothing there, so the score is scaled down instead of being a ratio
/// of two tiny numbers.
const MIN_SPEECH_MASS: f64 = 0.3;

#[derive(Debug, Serialize)]
pub struct PhonemeResult {
    pub ipa: String,
    pub score: i64,
    pub level: &'static str,
    pub start: f64,
    pub end: f64,
    /// The phoneme the model heard instead, when the score is not good.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub heard: Option<String>,
}

/// `score`, `level`, `start` and `end` are null for a word with no scorable
/// phonemes (not in the model's phoneme set).
#[derive(Debug, Serialize)]
pub struct WordResult {
    pub word: String,
    pub score: Option<i64>,
    pub level: Option<&'static str>,
    pub start: Option<f64>,
    pub end: Option<f64>,
    pub phonemes: Vec<PhonemeResult>,
}

#[derive(Debug, Serialize)]
pub struct Pause {
    pub after: String,
    pub seconds: f64,
}

#[derive(Debug, Serialize)]
pub struct Fluency {
    pub speaking_seconds: f64,
    pub phonemes_per_second: Option<f64>,
    pub pause_count: usize,
    pub pauses: Vec<Pause>,
}

#[derive(Debug, Serialize)]
pub struct Assessment {
    pub accuracy: i64,
    pub level: &'static str,
    pub fluency: Fluency,
    pub audio_seconds: f64,
    pub words: Vec<WordResult>,
    pub text: String,
    pub processing_seconds: f64,
}

/// Rounds to `digits` decimals the way the reference implementation does
/// (correctly rounded on the decimal value, ties to even).
fn round_to(x: f64, digits: usize) -> f64 {
    format!("{x:.digits$}").parse().unwrap_or(x)
}

fn percent(score: f64) -> i64 {
    (score * 100.0).round_ties_even() as i64
}

pub fn level(score: f64, config: &Config) -> &'static str {
    if score >= config.good_threshold {
        "good"
    } else if score >= config.fair_threshold {
        "fair"
    } else {
        "poor"
    }
}

/// Goodness of pronunciation for one aligned phoneme.
///
/// Returns (score in 0..1, id of the phoneme the model heard most).
/// The score is the share of non-blank probability that went to the expected
/// phoneme over the frames aligned to it.
pub fn phoneme_score(probs: &Frames, seg: &Segment, blank: usize) -> (f64, usize) {
    let mut mass = vec![0f32; probs.vocab()];
    for t in seg.start..seg.end {
        for (m, p) in mass.iter_mut().zip(probs.row(t)) {
            *m += p;
        }
    }
    let speech = f64::from((seg.end - seg.start) as f32 - mass[blank]);
    let score = f64::from(mass[seg.token]) / speech.max(MIN_SPEECH_MASS);

    mass[blank] = -1.0;
    let mut heard = 0;
    for (id, &m) in mass.iter().enumerate() {
        if m > mass[heard] {
            heard = id;
        }
    }
    (score.min(1.0), heard)
}

#[allow(clippy::too_many_arguments)]
pub fn build_result(
    probs: &Frames,
    segments: &[Segment],
    words: &[String],
    word_phonemes: &[Vec<String>],
    id_to_phoneme: &HashMap<usize, String>,
    blank: usize,
    audio_seconds: f64,
    config: &Config,
) -> Assessment {
    let dt = FRAME_SECONDS;
    let mut out_words = Vec::with_capacity(words.len());
    let mut all_scores = Vec::with_capacity(segments.len());
    let mut segments = segments.iter();

    for (word, phonemes) in words.iter().zip(word_phonemes) {
        let mut out_phonemes = Vec::with_capacity(phonemes.len());
        for (ipa, seg) in phonemes.iter().zip(&mut segments) {
            let (score, heard_id) = phoneme_score(probs, seg, blank);
            all_scores.push(score);
            let lvl = level(score, config);
            let heard = (heard_id != seg.token && lvl != "good")
                .then(|| id_to_phoneme.get(&heard_id).cloned().unwrap_or_else(|| "?".into()));
            out_phonemes.push(PhonemeResult {
                ipa: ipa.clone(),
                score: percent(score),
                level: lvl,
                start: round_to(seg.start as f64 * dt, 2),
                end: round_to(seg.end as f64 * dt, 2),
                heard,
            });
        }

        // The word score is the mean of the rounded phoneme scores.
        let word_score = (!out_phonemes.is_empty()).then(|| {
            let sum: i64 = out_phonemes.iter().map(|p| p.score).sum();
            sum as f64 / out_phonemes.len() as f64 / 100.0
        });
        out_words.push(WordResult {
            word: word.clone(),
            score: word_score.map(percent),
            level: word_score.map(|s| level(s, config)),
            start: out_phonemes.first().map(|p| p.start),
            end: out_phonemes.last().map(|p| p.end),
            phonemes: out_phonemes,
        });
    }

    let accuracy = all_scores.iter().sum::<f64>() / all_scores.len().max(1) as f64;
    Assessment {
        accuracy: percent(accuracy),
        level: level(accuracy, config),
        fluency: fluency(&out_words, all_scores.len(), config),
        audio_seconds: round_to(audio_seconds, 2),
        words: out_words,
        text: String::new(),
        processing_seconds: 0.0,
    }
}

fn fluency(words: &[WordResult], n_phonemes: usize, config: &Config) -> Fluency {
    // (word, start, end) of every word that has a time span.
    let timed: Vec<(&str, f64, f64)> = words
        .iter()
        .filter_map(|w| Some((w.word.as_str(), w.start?, w.end?)))
        .collect();

    let mut pauses = Vec::new();
    for pair in timed.windows(2) {
        let gap = pair[1].1 - pair[0].2;
        if gap >= config.pause_seconds {
            pauses.push(Pause { after: pair[0].0.to_string(), seconds: round_to(gap, 2) });
        }
    }

    let speaking = match (timed.first(), timed.last()) {
        (Some(first), Some(last)) => last.2 - first.1,
        _ => 0.0,
    };
    Fluency {
        speaking_seconds: round_to(speaking, 2),
        phonemes_per_second: (speaking > 0.0).then(|| round_to(n_phonemes as f64 / speaking, 1)),
        pause_count: pauses.len(),
        pauses,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::align::tests::{frames_to_probs, BLANK};

    fn seg(token: usize, start: usize, end: usize) -> Segment {
        Segment { token, start, end }
    }

    #[test]
    fn score_is_high_for_expected_phoneme_and_low_for_another() {
        let probs = frames_to_probs(&[2, 2, 2], 0.9);
        let (good, _) = phoneme_score(&probs, &seg(2, 0, 3), BLANK);
        let (bad, heard) = phoneme_score(&probs, &seg(1, 0, 3), BLANK);
        assert!(good > 0.85);
        assert!(bad < 0.1);
        assert_eq!(heard, 2);
    }

    #[test]
    fn score_is_low_when_nothing_was_spoken() {
        let probs = frames_to_probs(&[0, 0], 0.98);
        let (score, _) = phoneme_score(&probs, &seg(1, 0, 2), BLANK);
        assert!(score < 0.1);
    }

    #[test]
    fn rounding_matches_the_reference() {
        // Ties go to even, as in Python's round().
        assert_eq!(percent(0.005), 0);
        assert_eq!(percent(0.015), 2);
        assert_eq!(percent(0.025), 2);
        assert_eq!(round_to(0.125, 2), 0.12);
        assert_eq!(round_to(2.0, 2), 2.0);
        assert_eq!(round_to(12.96, 1), 13.0);
    }

    #[test]
    fn builds_words_pauses_and_heard_phonemes() {
        let config = Config::for_tests();
        // Word "a" = token 1 said well; long silence; word "b" = token 2 expected, 3 said.
        let mut frame_tokens = vec![1, 1];
        frame_tokens.extend([0; 30]);
        frame_tokens.extend([3, 3]);
        let probs = frames_to_probs(&frame_tokens, 0.9);
        let segments = [seg(1, 0, 2), seg(2, 32, 34)];
        let words = ["a".to_string(), "x".to_string(), "b".to_string()];
        let word_phonemes = [vec!["A".to_string()], vec![], vec!["B".to_string()]];
        let id_to_phoneme = HashMap::from([(3, "C".to_string())]);

        let r = build_result(
            &probs, &segments, &words, &word_phonemes, &id_to_phoneme, BLANK, 0.68, &config,
        );

        assert_eq!(r.words[0].level, Some("good"));
        assert_eq!(r.words[0].phonemes[0].heard, None);
        // A word with nothing to score keeps its place, with nulls.
        assert_eq!(r.words[1].score, None);
        assert_eq!(r.words[1].start, None);
        assert_eq!(r.words[2].level, Some("poor"));
        assert_eq!(r.words[2].phonemes[0].heard.as_deref(), Some("C"));
        assert_eq!((r.words[2].start, r.words[2].end), (Some(0.64), Some(0.68)));
        assert_eq!(r.fluency.pause_count, 1);
        assert_eq!(r.fluency.pauses[0].after, "a");
        assert_eq!(r.fluency.pauses[0].seconds, 0.6);
        assert_eq!(r.fluency.speaking_seconds, 0.68);
        assert_eq!(r.fluency.phonemes_per_second, Some(2.9));
    }
}
