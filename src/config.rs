//! Tunable settings. Override any of them with an environment variable.

use std::env;
use std::path::PathBuf;
use std::str::FromStr;

pub const SAMPLE_RATE: usize = 16_000;
/// wav2vec2 downsamples by 320 samples, so one output frame covers 20 ms.
pub const FRAME_SECONDS: f64 = 320.0 / SAMPLE_RATE as f64;

/// A language the server can score and speak.
pub struct Language {
    /// Code used in requests.
    pub code: &'static str,
    /// Name given to the chat model.
    pub name: &'static str,
    /// espeak voice used to turn the sentence into reference phonemes.
    pub espeak_voice: &'static str,
    /// Piper voice file (without extension) in the TTS directory.
    pub piper_voice: &'static str,
}

pub const LANGUAGES: &[Language] = &[
    Language {
        code: "en",
        name: "English",
        espeak_voice: "en-us",
        piper_voice: "en_US-lessac-medium",
    },
    Language {
        code: "es",
        name: "Spanish",
        espeak_voice: "es",
        piper_voice: "es_ES-davefx-medium",
    },
];

/// Language used when a request does not name one.
pub const DEFAULT_LANGUAGE: &str = "en";

pub fn language(code: &str) -> Option<&'static Language> {
    LANGUAGES.iter().find(|l| l.code == code)
}

#[derive(Debug, Clone)]
pub struct Config {
    /// Name reported by `/health`.
    pub model_id: String,
    /// Directory with model.onnx, vocab.json and config.json
    /// (written by tools/export_onnx.py).
    pub model_dir: PathBuf,
    /// Directory with the Piper voices (`<name>.onnx` and `<name>.onnx.json`).
    pub tts_dir: PathBuf,
    /// The espeak-ng shared library, and its data directory if it is not in
    /// the library's default location.
    pub espeak_lib: PathBuf,
    pub espeak_data: Option<PathBuf>,
    /// CPU threads for inference.
    pub threads: usize,

    pub host: String,
    pub port: u16,

    pub max_audio_seconds: f64,
    pub max_upload_bytes: usize,
    pub max_text_chars: usize,
    /// Longest text accepted for speech synthesis.
    pub max_tts_chars: usize,

    /// Chat model behind any OpenAI-compatible `/chat/completions` endpoint.
    /// `None` when chat is not configured.
    pub ai_base_url: Option<String>,
    pub ai_api_key: Option<String>,
    pub ai_model: String,

    /// Score bands on the 0..1 phoneme score. These are starting values, not
    /// calibrated against human ratings; tune them with real learner recordings.
    pub good_threshold: f64,
    pub fair_threshold: f64,

    /// A gap between two words longer than this counts as a pause.
    pub pause_seconds: f64,
}

fn var<T: FromStr>(name: &str, default: T) -> Result<T, String> {
    match env::var(name) {
        Ok(v) => v.parse().map_err(|_| format!("invalid value for {name}: {v:?}")),
        Err(_) => Ok(default),
    }
}

impl Config {
    pub fn from_env() -> Result<Self, String> {
        let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
        let default_lib = if cfg!(windows) {
            "espeak/espeak-ng.dll"
        } else if cfg!(target_os = "macos") {
            "libespeak-ng.dylib"
        } else {
            "libespeak-ng.so.1"
        };
        let espeak_data = match env::var("SPEAKMIND_ESPEAK_DATA") {
            Ok(v) => Some(PathBuf::from(v)),
            Err(_) => Some(PathBuf::from("espeak/espeak-ng-data")).filter(|p| p.is_dir()),
        };
        // Chat is on when a key is set (OpenAI by default) or an endpoint is
        // named (a local server such as Ollama needs no key).
        let non_empty = |name: &str| env::var(name).ok().filter(|v| !v.trim().is_empty());
        let ai_api_key = non_empty("SPEAKMIND_AI_API_KEY");
        let ai_base_url = non_empty("SPEAKMIND_AI_BASE_URL")
            .or_else(|| ai_api_key.as_ref().map(|_| "https://api.openai.com/v1".to_string()))
            .map(|url| url.trim().trim_end_matches('/').to_string());
        Ok(Self {
            model_id: var("SPEAKMIND_MODEL", "facebook/wav2vec2-lv-60-espeak-cv-ft".to_string())?,
            model_dir: var("SPEAKMIND_MODEL_DIR", PathBuf::from("models"))?,
            tts_dir: var("SPEAKMIND_TTS_DIR", PathBuf::from("tts"))?,
            espeak_lib: var("SPEAKMIND_ESPEAK_LIB", PathBuf::from(default_lib))?,
            espeak_data,
            threads: var("SPEAKMIND_THREADS", cpus.min(8))?,
            host: var("SPEAKMIND_HOST", "0.0.0.0".to_string())?,
            port: var("SPEAKMIND_PORT", 8000)?,
            max_audio_seconds: var("SPEAKMIND_MAX_AUDIO_SECONDS", 30.0)?,
            max_upload_bytes: var("SPEAKMIND_MAX_UPLOAD_BYTES", 10 * 1024 * 1024)?,
            max_text_chars: var("SPEAKMIND_MAX_TEXT_CHARS", 300)?,
            max_tts_chars: var("SPEAKMIND_MAX_TTS_CHARS", 600)?,
            ai_base_url,
            ai_api_key,
            ai_model: var("SPEAKMIND_AI_MODEL", "gpt-4o-mini".to_string())?,
            good_threshold: var("SPEAKMIND_GOOD_THRESHOLD", 0.7)?,
            fair_threshold: var("SPEAKMIND_FAIR_THRESHOLD", 0.4)?,
            pause_seconds: var("SPEAKMIND_PAUSE_SECONDS", 0.5)?,
        })
    }
}

#[cfg(test)]
impl Config {
    pub fn for_tests() -> Self {
        Self {
            model_id: String::new(),
            model_dir: PathBuf::new(),
            tts_dir: PathBuf::new(),
            espeak_lib: PathBuf::new(),
            espeak_data: None,
            threads: 1,
            host: String::new(),
            port: 0,
            max_audio_seconds: 30.0,
            max_upload_bytes: 10 * 1024 * 1024,
            max_text_chars: 300,
            max_tts_chars: 600,
            ai_base_url: None,
            ai_api_key: None,
            ai_model: String::new(),
            good_threshold: 0.7,
            fair_threshold: 0.4,
            pause_seconds: 0.5,
        }
    }
}
