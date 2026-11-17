//! Conversation practice through any OpenAI-compatible chat endpoint.
//!
//! The API key stays on the server: the app only talks to `/chat`.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::config::{Config, Language};

const MAX_MESSAGES: usize = 40;
const MAX_MESSAGE_CHARS: usize = 2000;
/// Marks the correction note the model appends after its reply.
const TIP_MARKER: &str = "Tip:";

#[derive(Debug, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, PartialEq)]
pub struct Reply {
    /// What the partner says. This is the part to read aloud.
    pub reply: String,
    /// A short correction of the learner's last message, in Chinese.
    pub tip: Option<String>,
}

pub enum ChatError {
    /// The request is not acceptable.
    Invalid(String),
    /// Chat is not configured on this server.
    Disabled,
    /// The chat service failed or answered with something unusable.
    Upstream(String),
}

pub struct Chat {
    client: reqwest::Client,
}

impl Chat {
    pub fn new() -> Result<Self, String> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .map_err(|e| format!("cannot create HTTP client: {e}"))?;
        Ok(Self { client })
    }

    pub async fn reply(
        &self,
        config: &Config,
        language: &Language,
        messages: &[Message],
    ) -> Result<Reply, ChatError> {
        let base = config.ai_base_url.as_deref().ok_or(ChatError::Disabled)?;
        validate(messages)?;

        let mut payload = vec![json!({ "role": "system", "content": system_prompt(language) })];
        payload.extend(messages.iter().map(|m| json!({ "role": m.role, "content": m.content })));

        let mut request = self.client.post(format!("{base}/chat/completions")).json(&json!({
            "model": config.ai_model,
            "messages": payload,
            "temperature": 0.7,
            "max_tokens": 400,
        }));
        if let Some(key) = &config.ai_api_key {
            request = request.bearer_auth(key);
        }

        let response = request.send().await.map_err(|e| {
            // reqwest's message can include the URL, but never the key.
            ChatError::Upstream(format!("cannot reach the chat service: {e}"))
        })?;
        let status = response.status();
        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|_| ChatError::Upstream(format!("chat service answered {status} with no JSON")))?;
        if !status.is_success() {
            let detail = body["error"]["message"].as_str().unwrap_or("no details");
            return Err(ChatError::Upstream(format!("chat service error {status}: {detail}")));
        }
        let content = body["choices"][0]["message"]["content"]
            .as_str()
            .map(str::trim)
            .filter(|c| !c.is_empty())
            .ok_or_else(|| ChatError::Upstream("chat service returned an empty reply".into()))?;
        Ok(split_tip(content))
    }
}

fn validate(messages: &[Message]) -> Result<(), ChatError> {
    if messages.is_empty() {
        return Err(ChatError::Invalid("messages is empty".into()));
    }
    if messages.len() > MAX_MESSAGES {
        return Err(ChatError::Invalid(format!("more than {MAX_MESSAGES} messages")));
    }
    for m in messages {
        if m.role != "user" && m.role != "assistant" {
            return Err(ChatError::Invalid(format!("unknown role {:?}", m.role)));
        }
        if m.content.chars().count() > MAX_MESSAGE_CHARS {
            return Err(ChatError::Invalid(format!("a message is longer than {MAX_MESSAGE_CHARS} characters")));
        }
    }
    if messages.last().is_some_and(|m| m.role != "user") {
        return Err(ChatError::Invalid("the last message must be from the user".into()));
    }
    Ok(())
}

fn system_prompt(language: &Language) -> String {
    let name = language.name;
    format!(
        "You are a friendly {name} conversation partner for a learner whose first language is Chinese. \
Talk about everyday topics and keep the conversation going.\n\
Rules:\n\
- Write only in {name}, in simple natural sentences. Reply in 1 to 3 short sentences and end with a short question.\n\
- Plain text only: no markdown, lists, emoji or stage directions. Your reply is read aloud.\n\
- If the learner's last message has a clear mistake, add a last line that starts with \"{TIP_MARKER}\" and \
explains the correction briefly in Simplified Chinese, quoting the corrected {name} phrase. \
If there is no clear mistake, add no such line.\n\
- If the learner writes in Chinese, answer in {name} and help them say it in {name}."
    )
}

/// Separates the spoken reply from the correction note.
fn split_tip(content: &str) -> Reply {
    let mut reply_lines = Vec::new();
    let mut tip_lines = Vec::new();
    let mut in_tip = false;
    for line in content.lines() {
        if let Some(rest) = line.trim_start().strip_prefix(TIP_MARKER) {
            in_tip = true;
            tip_lines.push(rest.trim().to_string());
        } else if in_tip {
            tip_lines.push(line.trim().to_string());
        } else {
            reply_lines.push(line.trim().to_string());
        }
    }
    let join = |lines: Vec<String>| lines.into_iter().filter(|l| !l.is_empty()).collect::<Vec<_>>().join(" ");
    let (reply, tip) = (join(reply_lines), join(tip_lines));
    if reply.is_empty() {
        // The model answered with only a tip: show it as the reply.
        return Reply { reply: tip, tip: None };
    }
    Reply { reply, tip: (!tip.is_empty()).then_some(tip) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_the_tip_from_the_reply() {
        let r = split_tip("I am fine, thanks. And you?\nTip: 应说 \"I am fine\"。");
        assert_eq!(r.reply, "I am fine, thanks. And you?");
        assert_eq!(r.tip.as_deref(), Some("应说 \"I am fine\"。"));
    }

    #[test]
    fn reply_without_a_tip() {
        let r = split_tip("Nice to meet you!\nWhere are you from?");
        assert_eq!(r.reply, "Nice to meet you! Where are you from?");
        assert_eq!(r.tip, None);
    }

    #[test]
    fn rejects_bad_requests() {
        let user = |c: &str| Message { role: "user".into(), content: c.into() };
        assert!(validate(&[]).is_err());
        assert!(validate(&[Message { role: "system".into(), content: "x".into() }]).is_err());
        assert!(validate(&[Message { role: "assistant".into(), content: "x".into() }]).is_err());
        assert!(validate(&[user("hi")]).is_ok());
    }
}
