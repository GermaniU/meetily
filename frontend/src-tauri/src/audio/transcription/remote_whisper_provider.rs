// audio/transcription/remote_whisper_provider.rs
//
// Remote Whisper transcription provider: sends audio over HTTP to a
// self-hosted OpenAI-compatible `/v1/audio/transcriptions` endpoint
// (e.g. a faster-whisper server running on a machine with a dedicated GPU),
// instead of loading a model in-process on this machine.
//
// The endpoint is expected to accept a multipart/form-data POST with a
// `file` field (WAV audio) and an optional `language` field, and to return
// JSON shaped like `{"text": "..."}` — the same contract OpenAI's
// `/v1/audio/transcriptions` uses. Any faster-whisper / whisper.cpp server
// built against that convention works without changes.

use super::provider::{TranscriptionError, TranscriptionProvider, TranscriptResult};
use async_trait::async_trait;
use reqwest::multipart;
use serde::Deserialize;
use std::time::Duration;

/// Sample rate the rest of the transcription pipeline hands us audio at.
/// (See `TranscriptionProvider::transcribe` doc comment: 16kHz mono f32.)
const SAMPLE_RATE_HZ: u32 = 16_000;

/// How long we allow a single transcription request to run before giving up.
/// Large chunks on a cold GPU (model not yet resident) can take a while.
const REQUEST_TIMEOUT_SECS: u64 = 60;

#[derive(Debug, Deserialize)]
struct RemoteTranscriptionResponse {
    text: String,
}

/// Transcription provider that delegates to a remote whisper-compatible HTTP server.
pub struct RemoteWhisperProvider {
    /// Base URL of the remote server, e.g. "http://192.168.1.100:8093".
    /// No trailing slash expected; it is stripped defensively in `new`.
    base_url: String,
    client: reqwest::Client,
    model_label: String,
}

impl RemoteWhisperProvider {
    pub fn new(base_url: String) -> Self {
        let base_url = base_url.trim_end_matches('/').to_string();
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        let model_label = format!("remote ({})", base_url);

        Self {
            base_url,
            client,
            model_label,
        }
    }

    fn endpoint(&self) -> String {
        format!("{}/v1/audio/transcriptions", self.base_url)
    }

    /// Encode 16kHz mono f32 PCM samples as a WAV byte buffer (16-bit PCM).
    /// Self-contained — no extra crate dependency needed for such a simple header.
    fn encode_wav_pcm16(samples: &[f32]) -> Vec<u8> {
        let num_samples = samples.len() as u32;
        let byte_rate = SAMPLE_RATE_HZ * 2; // mono, 16-bit
        let data_size = num_samples * 2;
        let riff_size = 36 + data_size;

        let mut buf = Vec::with_capacity(44 + data_size as usize);
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&riff_size.to_le_bytes());
        buf.extend_from_slice(b"WAVE");

        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
        buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
        buf.extend_from_slice(&1u16.to_le_bytes()); // mono
        buf.extend_from_slice(&SAMPLE_RATE_HZ.to_le_bytes());
        buf.extend_from_slice(&byte_rate.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes()); // block align
        buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample

        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_size.to_le_bytes());
        for &sample in samples {
            let clamped = sample.clamp(-1.0, 1.0);
            let pcm = (clamped * i16::MAX as f32) as i16;
            buf.extend_from_slice(&pcm.to_le_bytes());
        }

        buf
    }
}

#[async_trait]
impl TranscriptionProvider for RemoteWhisperProvider {
    async fn transcribe(
        &self,
        audio: Vec<f32>,
        language: Option<String>,
    ) -> std::result::Result<TranscriptResult, TranscriptionError> {
        if audio.is_empty() {
            return Err(TranscriptionError::AudioTooShort {
                samples: 0,
                minimum: 1,
            });
        }

        let wav_bytes = Self::encode_wav_pcm16(&audio);

        let mut form = multipart::Form::new().part(
            "file",
            multipart::Part::bytes(wav_bytes)
                .file_name("chunk.wav")
                .mime_str("audio/wav")
                .map_err(|e| TranscriptionError::EngineFailed(e.to_string()))?,
        );
        if let Some(lang) = language.clone() {
            form = form.text("language", lang);
        }

        let response = self
            .client
            .post(self.endpoint())
            .multipart(form)
            .send()
            .await
            .map_err(|e| {
                TranscriptionError::EngineFailed(format!(
                    "Cannot reach remote whisper server at {}: {}",
                    self.base_url, e
                ))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(TranscriptionError::EngineFailed(format!(
                "Remote whisper server returned {}: {}",
                status, body
            )));
        }

        let parsed: RemoteTranscriptionResponse = response
            .json()
            .await
            .map_err(|e| TranscriptionError::EngineFailed(format!("Invalid response JSON: {}", e)))?;

        Ok(TranscriptResult {
            text: parsed.text,
            confidence: None, // faster-whisper server doesn't return a confidence score
            is_partial: false,
        })
    }

    async fn is_model_loaded(&self) -> bool {
        // "Loaded" here means "reachable" — the remote process owns model lifecycle.
        let health_url = format!("{}/health", self.base_url);
        matches!(
            self.client.get(&health_url).send().await,
            Ok(resp) if resp.status().is_success()
        )
    }

    async fn get_current_model(&self) -> Option<String> {
        Some(self.model_label.clone())
    }

    fn provider_name(&self) -> &'static str {
        "RemoteWhisper"
    }
}
