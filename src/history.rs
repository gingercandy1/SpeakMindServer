//! Play-history backup: append-only JSONL storage plus upload validation.
//!
//! The App keeps the authoritative local history and periodically uploads a
//! batch to `POST /history`; the server only stores it, so a record is
//! returned verbatim on `GET /history`. This module is deliberately free of
//! HTTP types so it can be unit-tested on its own.
//!
//! Storage: one JSON object per line in `<dir>/history.jsonl` (default dir
//! `data`). A line that no longer parses is skipped on read, never fatal.
//!
//! Record contract (shared with the App, field names and casing included):
//! ```json
//! {"id": "<uuid v4>", "device_id": "<uuid v4>", "level_id": "1-1",
//!  "started_at": "2026-11-18T10:00:00.000Z",
//!  "finished_at": "2026-11-18T10:04:30.000Z",
//!  "stars": 3, "xp_earned": 20, "pron_avg": 78.5, "challenge_score": 92,
//!  "rounds": 4, "hints_used": 1,
//!  "weakest_sentence": "¿Dónde está la estación？", "completed": true}
//! ```

use std::fs::OpenOptions;
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Largest request body accepted for `POST /history` (about 1 MiB).
pub const MAX_BODY_BYTES: usize = 1024 * 1024;

/// Largest batch accepted in a single `POST /history` upload.
pub const MAX_RECORDS_PER_UPLOAD: usize = 500;

/// One finished (or abandoned) level attempt. Field order matches the
/// contract above; unknown fields are ignored on read, and a missing
/// nullable field reads as `None`.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct HistoryRecord {
    pub id: String,
    pub device_id: String,
    pub level_id: String,
    pub started_at: String,
    pub finished_at: String,
    pub stars: i64,
    pub xp_earned: i64,
    pub pron_avg: Option<f64>,
    pub challenge_score: Option<i64>,
    pub rounds: i64,
    pub hints_used: i64,
    pub weakest_sentence: Option<String>,
    pub completed: bool,
}

/// Append-only store backed by `<dir>/history.jsonl`.
#[derive(Debug, Clone)]
pub struct HistoryStore {
    dir: PathBuf,
}

impl HistoryStore {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    fn path(&self) -> PathBuf {
        self.dir.join("history.jsonl")
    }

    /// Append `records` (one JSON object per line) and return how many were
    /// written. The directory is created when it does not exist yet.
    pub fn append(&self, records: &[HistoryRecord]) -> io::Result<usize> {
        std::fs::create_dir_all(&self.dir)?;
        let mut file = OpenOptions::new().create(true).append(true).open(self.path())?;
        for record in records {
            let line = serde_json::to_string(record)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            writeln!(file, "{line}")?;
        }
        Ok(records.len())
    }

    /// Records for `device_id`, newest first. A missing file reads as empty;
    /// a line that fails to parse is skipped.
    pub fn read(&self, device_id: &str, limit: usize) -> io::Result<Vec<HistoryRecord>> {
        let file = match std::fs::File::open(self.path()) {
            Ok(file) => file,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };
        let mut records = Vec::new();
        for line in BufReader::new(file).lines() {
            let line = line?;
            if let Ok(record) = serde_json::from_str::<HistoryRecord>(&line) {
                if record.device_id == device_id {
                    records.push(record);
                }
            }
        }
        records.reverse();
        records.truncate(limit);
        Ok(records)
    }
}

/// The raw request body must fit in the upload budget. Checked in the
/// handler as a backstop for the route-level `DefaultBodyLimit`: a body can
/// only be stored if it fits, and what is stored is a subset of the body.
pub fn check_body_size(len: usize) -> Result<(), String> {
    if len > MAX_BODY_BYTES {
        Err(format!("request body is too large (max {MAX_BODY_BYTES} bytes)"))
    } else {
        Ok(())
    }
}

/// Validate an upload body: `device_id` is the top-level device the batch
/// belongs to, `records` is the raw `records` value from the JSON body.
/// Returns the records that pass validation; a bad record is skipped, it
/// never fails the whole batch. Only a missing/empty `device_id`, a missing
/// or non-array `records`, or a batch over [`MAX_RECORDS_PER_UPLOAD`] fails
/// the upload.
pub fn validate_upload(
    device_id: Option<&str>,
    records: &serde_json::Value,
) -> Result<Vec<HistoryRecord>, String> {
    let device_id = device_id
        .map(str::trim)
        .filter(|d| !d.is_empty())
        .ok_or_else(|| "device_id is required".to_string())?;
    let records = records
        .as_array()
        .ok_or_else(|| "records must be an array".to_string())?;
    if records.len() > MAX_RECORDS_PER_UPLOAD {
        return Err(format!(
            "at most {MAX_RECORDS_PER_UPLOAD} records per upload"
        ));
    }
    Ok(records.iter().filter_map(|r| validate_record(device_id, r)).collect())
}

/// A record is valid when id/device_id/level_id/started_at/finished_at are
/// non-empty strings, its device_id matches the batch, stars is an integer
/// 0..=3, completed is a bool, and every optional numeric field, when
/// present, has the right shape (a JSON null counts as absent).
fn validate_record(device_id: &str, value: &serde_json::Value) -> Option<HistoryRecord> {
    let non_empty = |field: &str| {
        value
            .get(field)
            .and_then(|v| v.as_str())
            .filter(|s| !s.trim().is_empty())
    };
    let id = non_empty("id")?;
    if non_empty("device_id")? != device_id {
        return None;
    }
    let level_id = non_empty("level_id")?;
    let started_at = non_empty("started_at")?;
    let finished_at = non_empty("finished_at")?;

    let stars = as_non_neg_int(value.get("stars")?)?;
    if stars > 3 {
        return None;
    }
    let completed = value.get("completed")?.as_bool()?;

    let xp_earned = opt_non_neg_int(value, "xp_earned")?.unwrap_or(0);
    let rounds = opt_non_neg_int(value, "rounds")?.unwrap_or(0);
    let hints_used = opt_non_neg_int(value, "hints_used")?.unwrap_or(0);
    let challenge_score = opt_non_neg_int(value, "challenge_score")?;
    let pron_avg = opt_number(value, "pron_avg")?;
    let weakest_sentence = opt_string(value, "weakest_sentence")?;

    Some(HistoryRecord {
        id: id.to_string(),
        device_id: device_id.to_string(),
        level_id: level_id.to_string(),
        started_at: started_at.to_string(),
        finished_at: finished_at.to_string(),
        stars,
        xp_earned,
        pron_avg,
        challenge_score,
        rounds,
        hints_used,
        weakest_sentence,
        completed,
    })
}

/// A JSON integer >= 0, or `None` when the value is not an integer.
fn as_non_neg_int(value: &serde_json::Value) -> Option<i64> {
    if let Some(i) = value.as_i64() {
        (i >= 0).then_some(i)
    } else {
        value.as_u64().and_then(|u| i64::try_from(u).ok())
    }
}

/// `None` when the field is present but not a non-negative integer;
/// `Some(None)` when it is absent or null; `Some(Some(n))` when valid.
fn opt_non_neg_int(value: &serde_json::Value, field: &str) -> Option<Option<i64>> {
    match value.get(field) {
        None | Some(serde_json::Value::Null) => Some(None),
        Some(n) => as_non_neg_int(n).map(Some),
    }
}

/// Same shape as [`opt_non_neg_int`], for a JSON number.
fn opt_number(value: &serde_json::Value, field: &str) -> Option<Option<f64>> {
    match value.get(field) {
        None | Some(serde_json::Value::Null) => Some(None),
        Some(n) => n.as_f64().map(Some),
    }
}

/// Same shape as [`opt_non_neg_int`], for a JSON string.
fn opt_string(value: &serde_json::Value, field: &str) -> Option<Option<String>> {
    match value.get(field) {
        None | Some(serde_json::Value::Null) => Some(None),
        Some(s) => s.as_str().map(|s| Some(s.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn record(id: &str, device_id: &str) -> HistoryRecord {
        HistoryRecord {
            id: id.to_string(),
            device_id: device_id.to_string(),
            level_id: "1-1".to_string(),
            started_at: "2026-11-18T10:00:00.000Z".to_string(),
            finished_at: "2026-11-18T10:04:30.000Z".to_string(),
            stars: 3,
            xp_earned: 20,
            pron_avg: Some(78.5),
            challenge_score: Some(92),
            rounds: 4,
            hints_used: 1,
            weakest_sentence: Some("¿Dónde está la estación？".to_string()),
            completed: true,
        }
    }

    fn store() -> (tempfile::TempDir, HistoryStore) {
        let dir = tempfile::tempdir().expect("temp dir");
        let store = HistoryStore::new(dir.path().to_path_buf());
        (dir, store)
    }

    #[test]
    fn append_then_read_round_trips() {
        let (_dir, store) = store();
        let want = record("a", "dev-1");
        assert_eq!(store.append(&[want.clone()]).unwrap(), 1);
        let got = store.read("dev-1", 100).unwrap();
        assert_eq!(got, vec![want]);
    }

    #[test]
    fn read_missing_file_is_empty() {
        let (_dir, store) = store();
        assert!(store.read("dev-1", 100).unwrap().is_empty());
    }

    #[test]
    fn read_filters_by_device_and_is_newest_first() {
        let (_dir, store) = store();
        store.append(&[record("old", "dev-1")]).unwrap();
        store.append(&[record("other", "dev-2")]).unwrap();
        store.append(&[record("new", "dev-1")]).unwrap();
        let got = store.read("dev-1", 100).unwrap();
        assert_eq!(
            got.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
            ["new", "old"]
        );
        assert_eq!(store.read("dev-2", 100).unwrap().len(), 1);
        assert!(store.read("nobody", 100).unwrap().is_empty());
    }

    #[test]
    fn read_truncates_to_limit() {
        let (_dir, store) = store();
        let batch: Vec<_> = (0..5).map(|i| record(&format!("r{i}"), "dev-1")).collect();
        store.append(&batch).unwrap();
        let got = store.read("dev-1", 2).unwrap();
        assert_eq!(
            got.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
            ["r4", "r3"]
        );
    }

    #[test]
    fn read_skips_corrupt_lines() {
        let (dir, store) = store();
        store.append(&[record("good", "dev-1")]).unwrap();
        {
            let mut file = OpenOptions::new()
                .append(true)
                .open(dir.path().join("history.jsonl"))
                .unwrap();
            writeln!(file, "{{not json").unwrap();
            writeln!(file, "{{\"id\": 1}}").unwrap();
        }
        // Appending after the corrupt lines still works; the corrupt lines
        // are skipped and the good record survives.
        store.append(&[record("later", "dev-1")]).unwrap();
        let got = store.read("dev-1", 100).unwrap();
        assert_eq!(
            got.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
            ["later", "good"]
        );
    }

    #[test]
    fn validate_rejects_missing_device_id_and_non_array_records() {
        let records = json!([{"id": "a"}]);
        assert!(validate_upload(None, &records).is_err());
        assert!(validate_upload(Some(""), &records).is_err());
        assert!(validate_upload(Some("   "), &records).is_err());
        assert!(validate_upload(Some("dev-1"), &json!({"id": "a"})).is_err());
        assert!(validate_upload(Some("dev-1"), &json!(null)).is_err());
        assert!(validate_upload(Some("dev-1"), &json!("nope")).is_err());
    }

    #[test]
    fn validate_skips_bad_records_and_keeps_good_ones() {
        let good = json!({
            "id": "good", "device_id": "dev-1", "level_id": "1-1",
            "started_at": "2026-11-18T10:00:00.000Z",
            "finished_at": "2026-11-18T10:04:30.000Z",
            "stars": 3, "xp_earned": 20, "pron_avg": 78.5, "challenge_score": 92,
            "rounds": 4, "hints_used": 1,
            "weakest_sentence": "¿Dónde está la estación？", "completed": true,
            "surprise_field": "ignored",
        });
        let cases = [
            // Missing id.
            json!({"device_id": "dev-1", "level_id": "1-1",
                "started_at": "2026-11-18T10:00:00.000Z",
                "finished_at": "2026-11-18T10:04:30.000Z",
                "stars": 2, "completed": true}),
            // Device mismatch.
            json!({"id": "x", "device_id": "dev-2", "level_id": "1-1",
                "started_at": "2026-11-18T10:00:00.000Z",
                "finished_at": "2026-11-18T10:04:30.000Z",
                "stars": 2, "completed": true}),
            // Stars out of range, and a non-integer stars.
            json!({"id": "x", "device_id": "dev-1", "level_id": "1-1",
                "started_at": "2026-11-18T10:00:00.000Z",
                "finished_at": "2026-11-18T10:04:30.000Z",
                "stars": 4, "completed": true}),
            json!({"id": "x", "device_id": "dev-1", "level_id": "1-1",
                "started_at": "2026-11-18T10:00:00.000Z",
                "finished_at": "2026-11-18T10:04:30.000Z",
                "stars": 2.5, "completed": true}),
            // completed is not a bool; negative xp; pron_avg is a string.
            json!({"id": "x", "device_id": "dev-1", "level_id": "1-1",
                "started_at": "2026-11-18T10:00:00.000Z",
                "finished_at": "2026-11-18T10:04:30.000Z",
                "stars": 2, "completed": "yes"}),
            json!({"id": "x", "device_id": "dev-1", "level_id": "1-1",
                "started_at": "2026-11-18T10:00:00.000Z",
                "finished_at": "2026-11-18T10:04:30.000Z",
                "stars": 2, "xp_earned": -5, "completed": true}),
            json!({"id": "x", "device_id": "dev-1", "level_id": "1-1",
                "started_at": "2026-11-18T10:00:00.000Z",
                "finished_at": "2026-11-18T10:04:30.000Z",
                "stars": 2, "pron_avg": "high", "completed": true}),
            // Empty level_id.
            json!({"id": "x", "device_id": "dev-1", "level_id": "  ",
                "started_at": "2026-11-18T10:00:00.000Z",
                "finished_at": "2026-11-18T10:04:30.000Z",
                "stars": 2, "completed": true}),
        ];
        let mut input = vec![good];
        input.extend(cases);
        let got = validate_upload(Some("dev-1"), &json!(input)).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].id, "good");
        assert_eq!(got[0].pron_avg, Some(78.5));
    }

    #[test]
    fn validate_accepts_missing_optional_fields_and_nulls() {
        let minimal = json!([{
            "id": "m", "device_id": "dev-1", "level_id": "2-3",
            "started_at": "2026-11-18T10:00:00.000Z",
            "finished_at": "2026-11-18T10:01:00.000Z",
            "stars": 0, "completed": false,
            "pron_avg": null, "challenge_score": null, "weakest_sentence": null,
        }]);
        let got = validate_upload(Some("dev-1"), &minimal).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].xp_earned, 0);
        assert_eq!(got[0].rounds, 0);
        assert_eq!(got[0].hints_used, 0);
        assert_eq!(got[0].pron_avg, None);
        assert_eq!(got[0].challenge_score, None);
        assert_eq!(got[0].weakest_sentence, None);
    }

    #[test]
    fn validate_rejects_batches_over_500_records() {
        let one = json!({
            "id": "x", "device_id": "dev-1", "level_id": "1-1",
            "started_at": "2026-11-18T10:00:00.000Z",
            "finished_at": "2026-11-18T10:04:30.000Z",
            "stars": 1, "completed": true,
        });
        let big: Vec<_> = (0..501).map(|_| one.clone()).collect();
        assert!(validate_upload(Some("dev-1"), &json!(big)).is_err());
        let ok: Vec<_> = (0..500).map(|_| one.clone()).collect();
        assert_eq!(validate_upload(Some("dev-1"), &json!(ok)).unwrap().len(), 500);
    }

    #[test]
    fn body_size_check_rejects_oversized_bodies() {
        assert!(check_body_size(MAX_BODY_BYTES).is_ok());
        assert!(check_body_size(MAX_BODY_BYTES + 1).is_err());
        // A batch of small records is far under the budget.
        let small = record("a", "dev-1");
        assert!(check_body_size(
            serde_json::to_string(&vec![small]).unwrap().len()
        )
        .is_ok());
    }
}
