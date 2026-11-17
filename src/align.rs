//! CTC forced alignment (Viterbi).

use std::fmt;

use crate::frames::Frames;

#[derive(Debug, PartialEq)]
pub struct AlignmentError(pub &'static str);

impl fmt::Display for AlignmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment {
    pub token: usize,
    /// First frame, inclusive.
    pub start: usize,
    /// Last frame, exclusive.
    pub end: usize,
}

/// Best frame span for every token, in order.
///
/// `log_probs`: (frames, vocab) log posteriors. `tokens`: the reference sequence.
pub fn forced_align(
    log_probs: &Frames,
    tokens: &[usize],
    blank: usize,
) -> Result<Vec<Segment>, AlignmentError> {
    let n_frames = log_probs.len();
    if tokens.is_empty() {
        return Err(AlignmentError("no phonemes to align"));
    }
    if n_frames == 0 {
        return Err(AlignmentError("audio is too short for this sentence"));
    }

    // States: blank, tok0, blank, tok1, ..., blank.
    let n_states = 2 * tokens.len() + 1;
    let states: Vec<usize> = (0..n_states)
        .map(|s| if s % 2 == 1 { tokens[s / 2] } else { blank })
        .collect();

    // A token state may be entered straight from the previous token state
    // (skipping the blank between them) unless both are the same token.
    let can_skip: Vec<bool> = (0..n_states)
        .map(|s| s % 2 == 1 && s >= 3 && states[s] != states[s - 2])
        .collect();

    let emit = |t: usize, s: usize| f64::from(log_probs.row(t)[states[s]]);

    let mut score = vec![f64::NEG_INFINITY; n_states];
    score[0] = emit(0, 0);
    score[1] = emit(0, 1);
    // back[t * n_states + s]: how many states back the best path into (t, s) came from.
    let mut back = vec![0u8; n_frames * n_states];
    let mut next = vec![f64::NEG_INFINITY; n_states];

    for t in 1..n_frames {
        for s in 0..n_states {
            // On a tie the smaller step wins: stay, then advance, then skip.
            let mut best = score[s];
            let mut step = 0u8;
            if s >= 1 && score[s - 1] > best {
                best = score[s - 1];
                step = 1;
            }
            if can_skip[s] && score[s - 2] > best {
                best = score[s - 2];
                step = 2;
            }
            next[s] = best + emit(t, s);
            back[t * n_states + s] = step;
        }
        std::mem::swap(&mut score, &mut next);
    }

    let mut state = if score[n_states - 1] >= score[n_states - 2] {
        n_states - 1
    } else {
        n_states - 2
    };
    if !score[state].is_finite() {
        return Err(AlignmentError("audio is too short for this sentence"));
    }

    // Walk the best path backwards, widening each token's span as it is seen.
    let mut segments: Vec<Segment> = tokens
        .iter()
        .map(|&token| Segment { token, start: usize::MAX, end: 0 })
        .collect();
    for t in (0..n_frames).rev() {
        if state % 2 == 1 {
            let seg = &mut segments[state / 2];
            seg.start = t;
            seg.end = seg.end.max(t + 1);
        }
        state -= usize::from(back[t * n_states + state]);
    }
    Ok(segments)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub const BLANK: usize = 0;

    /// `peak` on each frame's token, the rest of the probability spread evenly.
    pub fn frames_to_probs(frame_tokens: &[usize], peak: f32) -> Frames {
        let vocab = 5;
        let mut data = vec![(1.0 - peak) / (vocab as f32 - 1.0); frame_tokens.len() * vocab];
        for (t, &token) in frame_tokens.iter().enumerate() {
            data[t * vocab + token] = peak;
        }
        Frames::new(vocab, data)
    }

    fn log_probs(frame_tokens: &[usize]) -> Frames {
        frames_to_probs(frame_tokens, 0.9).ln()
    }

    fn seg(token: usize, start: usize, end: usize) -> Segment {
        Segment { token, start, end }
    }

    #[test]
    fn aligns_tokens_to_their_frames() {
        let lp = log_probs(&[0, 1, 1, 0, 2, 0, 0, 3, 3, 3, 0]);
        assert_eq!(
            forced_align(&lp, &[1, 2, 3], BLANK).unwrap(),
            vec![seg(1, 1, 3), seg(2, 4, 5), seg(3, 7, 10)]
        );
    }

    #[test]
    fn repeated_token_needs_a_blank_between() {
        let lp = log_probs(&[1, 1, 0, 1, 1]);
        assert_eq!(
            forced_align(&lp, &[1, 1], BLANK).unwrap(),
            vec![seg(1, 0, 2), seg(1, 3, 5)]
        );
    }

    #[test]
    fn adjacent_different_tokens_need_no_blank() {
        let lp = log_probs(&[1, 2]);
        assert_eq!(
            forced_align(&lp, &[1, 2], BLANK).unwrap(),
            vec![seg(1, 0, 1), seg(2, 1, 2)]
        );
    }

    #[test]
    fn too_few_frames_is_an_error() {
        assert!(forced_align(&log_probs(&[1, 2]), &[1, 2, 3], BLANK).is_err());
        // A repeated token needs three frames: token, blank, token.
        assert!(forced_align(&log_probs(&[1, 1]), &[1, 1], BLANK).is_err());
    }

    #[test]
    fn empty_reference_is_an_error() {
        assert_eq!(
            forced_align(&log_probs(&[0, 0]), &[], BLANK),
            Err(AlignmentError("no phonemes to align"))
        );
    }
}
