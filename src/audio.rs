//! Decode an uploaded recording to 16 kHz mono float32 with ffmpeg.

use std::io::{self, Write};
use std::process::Stdio;
use std::time::Duration;

use tokio::process::Command;

use crate::config::{Config, SAMPLE_RATE};

const FFMPEG_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Debug)]
pub enum AudioError {
    /// The recording itself is unusable; the message can be shown to the user.
    Invalid(String),
    /// The server could not run the decoder.
    Internal(String),
}

pub async fn decode(data: Vec<u8>, config: &Config) -> Result<Vec<f32>, AudioError> {
    // A temp file, not a pipe: m4a/mp4 from phones often keep their index at
    // the end of the file, which ffmpeg cannot read from a non-seekable pipe.
    let path = tokio::task::spawn_blocking(move || -> io::Result<tempfile::TempPath> {
        let mut file = tempfile::Builder::new().suffix(".audio").tempfile()?;
        file.write_all(&data)?;
        // Closes the handle so ffmpeg can open the file; it is deleted on drop.
        Ok(file.into_temp_path())
    })
    .await
    .map_err(|e| AudioError::Internal(e.to_string()))?
    .map_err(|e| AudioError::Internal(format!("cannot store the upload: {e}")))?;

    let mut command = Command::new("ffmpeg");
    command
        .args(["-v", "error", "-nostdin", "-i"])
        .arg(&*path)
        .args(["-f", "f32le", "-ac", "1", "-ar", &SAMPLE_RATE.to_string(), "pipe:1"])
        .stdin(Stdio::null())
        .kill_on_drop(true);

    let output = match tokio::time::timeout(FFMPEG_TIMEOUT, command.output()).await {
        Err(_) => return Err(AudioError::Internal("ffmpeg timed out".into())),
        Ok(Err(e)) if e.kind() == io::ErrorKind::NotFound => {
            return Err(AudioError::Internal("ffmpeg is not installed or not on PATH".into()))
        }
        Ok(Err(e)) => return Err(AudioError::Internal(format!("cannot run ffmpeg: {e}"))),
        Ok(Ok(output)) => output,
    };
    drop(path);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let detail = stderr.trim().lines().last().unwrap_or("unknown error");
        return Err(AudioError::Invalid(format!("cannot decode audio: {detail}")));
    }

    let samples: Vec<f32> = output
        .stdout
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    check_length(samples.len(), config)?;
    Ok(samples)
}

fn check_length(n_samples: usize, config: &Config) -> Result<(), AudioError> {
    let seconds = n_samples as f64 / SAMPLE_RATE as f64;
    if seconds < 0.2 {
        return Err(AudioError::Invalid("audio is shorter than 0.2 seconds".into()));
    }
    if seconds > config.max_audio_seconds {
        return Err(AudioError::Invalid(format!(
            "audio is longer than {} seconds",
            config.max_audio_seconds
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_too_short_and_too_long_audio() {
        let config = Config::for_tests();
        assert!(check_length(SAMPLE_RATE / 5, &config).is_ok());
        assert!(check_length(SAMPLE_RATE * 30, &config).is_ok());
        match check_length(SAMPLE_RATE / 5 - 1, &config) {
            Err(AudioError::Invalid(m)) => assert_eq!(m, "audio is shorter than 0.2 seconds"),
            other => panic!("unexpected: {other:?}"),
        }
        match check_length(SAMPLE_RATE * 30 + 1, &config) {
            Err(AudioError::Invalid(m)) => assert_eq!(m, "audio is longer than 30 seconds"),
            other => panic!("unexpected: {other:?}"),
        }
    }
}
