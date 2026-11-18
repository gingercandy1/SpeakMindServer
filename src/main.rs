//! HTTP API.
//!
//! Run:  cargo run --release
//! Listens on SPEAKMIND_HOST:SPEAKMIND_PORT (default 0.0.0.0:8000).
//!
//! - `POST /assess`  score a recording against a sentence (English or Spanish)
//! - `POST /tts`     speak a text with a Piper voice
//! - `POST /chat`    one turn of conversation practice with an AI partner
//! - `POST /history` back up play-history records; `GET /history` fetch them

mod align;
mod audio;
mod chat;
mod config;
mod frames;
mod g2p;
mod history;
mod model;
mod scoring;
mod tts;

use std::sync::Arc;
use std::time::Instant;

use axum::body::Bytes;
use axum::extract::multipart::{Multipart, MultipartError, MultipartRejection};
use axum::extract::{DefaultBodyLimit, Query, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;

use crate::audio::AudioError;
use crate::chat::{Chat, ChatError};
use crate::config::{Config, Language, DEFAULT_LANGUAGE, LANGUAGES, SAMPLE_RATE};
use crate::g2p::Phonemizer;
use crate::history::{check_body_size, validate_upload, HistoryStore, MAX_BODY_BYTES};
use crate::model::PhonemeModel;
use crate::scoring::Assessment;
use crate::tts::{Tts, TtsError};

struct AppState {
    config: Config,
    model: PhonemeModel,
    phonemizer: Phonemizer,
    tts: Tts,
    chat: Chat,
    history: HistoryStore,
}

/// An error response: `{"detail": "..."}` with the given status.
struct ApiError(StatusCode, String);

impl ApiError {
    fn unprocessable(detail: impl Into<String>) -> Self {
        Self(StatusCode::UNPROCESSABLE_ENTITY, detail.into())
    }

    fn internal(detail: impl Into<String>) -> Self {
        Self(StatusCode::INTERNAL_SERVER_ERROR, detail.into())
    }
}

/// The language a request asks for, `en` when it names none.
fn requested_language(code: Option<&str>) -> Result<&'static Language, ApiError> {
    let code = code.map(str::trim).filter(|c| !c.is_empty()).unwrap_or(DEFAULT_LANGUAGE);
    config::language(code).ok_or_else(|| {
        let supported: Vec<_> = LANGUAGES.iter().map(|l| l.code).collect();
        ApiError::unprocessable(format!("unsupported language {code:?} (use one of {supported:?})"))
    })
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        if self.0.is_server_error() {
            eprintln!("error: {}", self.1);
        }
        (self.0, Json(json!({ "detail": self.1 }))).into_response()
    }
}

const TOO_LARGE: &str = "audio file is too large";

impl From<MultipartError> for ApiError {
    fn from(e: MultipartError) -> Self {
        if e.status() == StatusCode::PAYLOAD_TOO_LARGE {
            Self(StatusCode::PAYLOAD_TOO_LARGE, TOO_LARGE.into())
        } else {
            Self(e.status(), e.body_text())
        }
    }
}

async fn health(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(json!({
        "status": "ok",
        "model": state.config.model_id,
        "languages": LANGUAGES.iter().map(|l| l.code).collect::<Vec<_>>(),
        "tts": state.tts.languages(),
        "chat": state.config.ai_base_url.is_some(),
    }))
}

/// Multipart form: `audio` is a recording in any format ffmpeg can read,
/// `text` is the sentence the user was asked to read, `language` is `en` (the
/// default) or `es`.
async fn assess(
    State(state): State<Arc<AppState>>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Json<Assessment>, ApiError> {
    let started = Instant::now();
    let config = &state.config;
    let mut multipart =
        multipart.map_err(|_| ApiError::unprocessable("expected a multipart form"))?;

    let mut audio: Option<Vec<u8>> = None;
    let mut text: Option<String> = None;
    let mut language: Option<String> = None;
    while let Some(mut field) = multipart.next_field().await? {
        match field.name() {
            Some("audio") => {
                let mut data = Vec::new();
                while let Some(chunk) = field.chunk().await? {
                    if data.len() + chunk.len() > config.max_upload_bytes {
                        return Err(ApiError(StatusCode::PAYLOAD_TOO_LARGE, TOO_LARGE.into()));
                    }
                    data.extend_from_slice(&chunk);
                }
                audio = Some(data);
            }
            Some("text") => text = Some(field.text().await?),
            Some("language") => language = Some(field.text().await?),
            _ => {}
        }
    }
    let audio = audio.ok_or_else(|| ApiError::unprocessable("audio is required"))?;
    let text = text.ok_or_else(|| ApiError::unprocessable("text is required"))?;
    let language = requested_language(language.as_deref())?;

    if text.chars().count() > config.max_text_chars {
        return Err(ApiError::unprocessable(format!(
            "text is longer than {} characters",
            config.max_text_chars
        )));
    }
    let words = g2p::split_words(&text);
    if words.is_empty() {
        return Err(ApiError::unprocessable("text has no words"));
    }

    let samples = audio::decode(audio, config).await.map_err(|e| match e {
        AudioError::Invalid(detail) => ApiError::unprocessable(detail),
        AudioError::Internal(detail) => ApiError::internal(detail),
    })?;

    // Inference is CPU-bound: keep it off the async worker threads.
    let worker_state = state.clone();
    let mut result = tokio::task::spawn_blocking(move || score(&worker_state, language, &words, &samples))
        .await
        .map_err(|e| ApiError::internal(e.to_string()))??;

    result.text = text;
    result.processing_seconds = (started.elapsed().as_secs_f64() * 100.0).round() / 100.0;
    Ok(Json(result))
}

fn score(
    state: &AppState,
    language: &Language,
    words: &[String],
    samples: &[f32],
) -> Result<Assessment, ApiError> {
    let model = &state.model;
    let phonemes = state
        .phonemizer
        .phonemize_words(language.espeak_voice, words)
        .map_err(ApiError::internal)?;
    let (token_ids, word_phonemes) = model.encode(&phonemes);
    let probs = model.posteriors(samples).map_err(ApiError::internal)?;
    let segments = align::forced_align(&probs.ln(), &token_ids, model.blank)
        .map_err(|e| ApiError::unprocessable(e.to_string()))?;
    Ok(scoring::build_result(
        &probs,
        &segments,
        words,
        &word_phonemes,
        &model.id_to_phoneme,
        model.blank,
        samples.len() as f64 / SAMPLE_RATE as f64,
        &state.config,
    ))
}

#[derive(Deserialize)]
struct TtsRequest {
    text: String,
    language: Option<String>,
}

/// JSON `{"text": "...", "language": "en"}`, answered with a WAV file.
async fn speak(
    State(state): State<Arc<AppState>>,
    Json(request): Json<TtsRequest>,
) -> Result<Response, ApiError> {
    let language = requested_language(request.language.as_deref())?;
    if request.text.chars().count() > state.config.max_tts_chars {
        return Err(ApiError::unprocessable(format!(
            "text is longer than {} characters",
            state.config.max_tts_chars
        )));
    }
    let worker_state = state.clone();
    let wav = tokio::task::spawn_blocking(move || {
        worker_state.tts.synthesize(&worker_state.phonemizer, language.code, &request.text)
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))?
    .map_err(|e| match e {
        TtsError::Unavailable(detail) => ApiError(StatusCode::SERVICE_UNAVAILABLE, detail),
        TtsError::NothingToSay => ApiError::unprocessable("text has nothing to pronounce"),
        TtsError::Internal(detail) => ApiError::internal(detail),
    })?;
    Ok(([(header::CONTENT_TYPE, "audio/wav")], wav).into_response())
}

#[derive(Deserialize)]
struct ChatRequest {
    messages: Vec<chat::Message>,
    language: Option<String>,
}

/// JSON `{"messages": [{"role": "user", "content": "..."}], "language": "en"}`,
/// answered with `{"reply": "...", "tip": "..." | null}`.
async fn converse(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatRequest>,
) -> Result<Json<chat::Reply>, ApiError> {
    let language = requested_language(request.language.as_deref())?;
    state
        .chat
        .reply(&state.config, language, &request.messages)
        .await
        .map(Json)
        .map_err(|e| match e {
            ChatError::Invalid(detail) => ApiError::unprocessable(detail),
            ChatError::Disabled => ApiError(
                StatusCode::SERVICE_UNAVAILABLE,
                "AI chat is not set up on the server (set SPEAKMIND_AI_API_KEY)".into(),
            ),
            ChatError::Upstream(detail) => ApiError(StatusCode::BAD_GATEWAY, detail),
        })
}

#[derive(Deserialize)]
struct HistoryQuery {
    device_id: Option<String>,
    limit: Option<usize>,
}

/// JSON `{"device_id": "...", "records": [...]}`, answered with
/// `{"accepted": n}`. Records that fail validation are skipped, never fail
/// the batch; the whole upload fails only when the device_id is missing or
/// `records` is not an array. File I/O runs off the async worker threads.
async fn upload_history(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> Result<Json<serde_json::Value>, ApiError> {
    // Backstop for the route-level body limit below: the limit is enforced
    // while collecting the body, and re-checked here on the bytes received.
    check_body_size(body.len()).map_err(|e| ApiError(StatusCode::PAYLOAD_TOO_LARGE, e))?;
    let value: serde_json::Value = serde_json::from_slice(&body)
        .map_err(|e| ApiError(StatusCode::BAD_REQUEST, format!("invalid JSON: {e}")))?;
    let device_id = value.get("device_id").and_then(|v| v.as_str());
    let records = value.get("records").cloned().unwrap_or(serde_json::Value::Null);
    let valid = validate_upload(device_id, &records)
        .map_err(|e| ApiError(StatusCode::BAD_REQUEST, e))?;
    let store = state.history.clone();
    let accepted = tokio::task::spawn_blocking(move || store.append(&valid))
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?
        .map_err(|e| ApiError::internal(e.to_string()))?;
    Ok(Json(json!({ "accepted": accepted })))
}

/// `GET /history?device_id=...&limit=100`, answered with
/// `{"device_id": "...", "records": [...]}` (newest first).
async fn list_history(
    State(state): State<Arc<AppState>>,
    Query(query): Query<HistoryQuery>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let device_id = query
        .device_id
        .map(|d| d.trim().to_string())
        .filter(|d| !d.is_empty())
        .ok_or_else(|| ApiError(StatusCode::BAD_REQUEST, "device_id is required".to_string()))?;
    let limit = query.limit.unwrap_or(100).clamp(1, 500);
    let store = state.history.clone();
    let device_id_for_read = device_id.clone();
    let records = tokio::task::spawn_blocking(move || store.read(&device_id_for_read, limit))
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?
        .map_err(|e| ApiError::internal(e.to_string()))?;
    Ok(Json(json!({ "device_id": device_id, "records": records })))
}

fn load() -> Result<AppState, String> {
    let config = Config::from_env()?;
    let phonemizer = Phonemizer::new(&config.espeak_lib, config.espeak_data.as_deref())?;
    for language in LANGUAGES {
        if !phonemizer.has_voice(language.espeak_voice) {
            return Err(format!("espeak-ng has no voice {:?}", language.espeak_voice));
        }
    }
    let model = PhonemeModel::load(&config.model_dir, config.threads)?;
    let tts = Tts::load(&config.tts_dir, model.environment(), config.threads);
    let history_dir = std::env::var("SPEAKMIND_HISTORY_DIR")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "data".to_string());
    Ok(AppState {
        config,
        model,
        phonemizer,
        tts,
        chat: Chat::new()?,
        history: HistoryStore::new(history_dir),
    })
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

#[tokio::main]
async fn main() {
    let started = Instant::now();
    let state = match load() {
        Ok(state) => Arc::new(state),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    let address = format!("{}:{}", state.config.host, state.config.port);

    let app = Router::new()
        .route("/health", get(health))
        .route("/assess", post(assess))
        .route("/tts", post(speak))
        .route("/chat", post(converse))
        .route("/history", get(list_history))
        .route(
            "/history",
            // History uploads are small JSON batches: cap this route well
            // below the global upload limit. The handler re-checks the
            // received bytes, so the cap holds even if a layer is skipped.
            post(upload_history).layer(DefaultBodyLimit::max(MAX_BODY_BYTES)),
        )
        // Room for the text field and the multipart framing around the audio.
        .layer(DefaultBodyLimit::max(state.config.max_upload_bytes + 64 * 1024))
        .with_state(state);

    let listener = match tokio::net::TcpListener::bind(&address).await {
        Ok(listener) => listener,
        Err(e) => {
            eprintln!("error: cannot listen on {address}: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "SpeakMind listening on http://{address} (model loaded in {:.1}s)",
        started.elapsed().as_secs_f64()
    );
    if let Err(e) = axum::serve(listener, app).with_graceful_shutdown(shutdown_signal()).await {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
