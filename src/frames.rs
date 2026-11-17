//! A (frames, vocab) matrix of per-frame phoneme values.

#[derive(Debug, Clone)]
pub struct Frames {
    vocab: usize,
    data: Vec<f32>,
}

impl Frames {
    pub fn new(vocab: usize, data: Vec<f32>) -> Self {
        assert!(vocab > 0 && data.len().is_multiple_of(vocab), "data does not fit the vocabulary size");
        Self { vocab, data }
    }

    /// Number of frames.
    pub fn len(&self) -> usize {
        self.data.len() / self.vocab
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }

    pub fn row(&self, frame: usize) -> &[f32] {
        &self.data[frame * self.vocab..(frame + 1) * self.vocab]
    }

    /// Element-wise natural log, with a floor so that zero stays finite.
    pub fn ln(&self) -> Frames {
        Frames {
            vocab: self.vocab,
            data: self.data.iter().map(|p| (p + 1e-10).ln()).collect(),
        }
    }
}
