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
/// Marker the model ends its reply with when the level task is done.
/// The server strips it before returning; it must never reach the app.
const TASK_DONE_MARKER: &str = "[TASK_DONE]";
/// Bounds for the optional level context (same order as a long message).
const MAX_NPC_PROMPT_CHARS: usize = 2000;
const MAX_TASK_CHARS: usize = 1000;
const MAX_LEVEL_ID_CHARS: usize = 64;

#[derive(Debug, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Optional level context the app sends with a chat turn: who the partner
/// is and what the learner has to do in this level. All fields are
/// optional; plain chat sends none of them.
#[derive(Debug, Default)]
pub struct ChatContext {
    /// Level the conversation happens in, e.g. "1-3".
    pub level_id: Option<String>,
    /// Persona for the partner, from the level's npc_prompt.
    pub npc_prompt: Option<String>,
    /// What the learner has to do in this level; enables task_done.
    pub task: Option<String>,
}

impl ChatContext {
    fn trimmed(field: &Option<String>) -> Option<&str> {
        field.as_deref().map(str::trim).filter(|s| !s.is_empty())
    }

    /// A non-empty task. Only then does the server ask the model to signal
    /// completion with [TASK_DONE] and report task_done.
    fn task(&self) -> Option<&str> {
        Self::trimmed(&self.task)
    }

    fn npc_prompt(&self) -> Option<&str> {
        Self::trimmed(&self.npc_prompt)
    }

    fn level_id(&self) -> Option<&str> {
        Self::trimmed(&self.level_id)
    }
}

#[derive(Debug, Serialize, PartialEq)]
pub struct Reply {
    /// What the partner says. This is the part to read aloud.
    pub reply: String,
    /// A short correction of the learner's last message, in Chinese.
    pub tip: Option<String>,
    /// True when the model signalled that the level task is done. Only
    /// meaningful when the request carried a task; false otherwise. New
    /// field — old clients ignore it.
    pub task_done: bool,
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
        context: &ChatContext,
    ) -> Result<Reply, ChatError> {
        let base = config.ai_base_url.as_deref().ok_or(ChatError::Disabled)?;
        validate(messages, context)?;

        let mut payload =
            vec![json!({ "role": "system", "content": system_prompt(language, context) })];
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
        let reply = split_reply(content, context);
        if reply.reply.is_empty() {
            // The model answered with nothing but the marker: still empty.
            return Err(ChatError::Upstream("chat service returned an empty reply".into()));
        }
        Ok(reply)
    }
}

fn validate(messages: &[Message], context: &ChatContext) -> Result<(), ChatError> {
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
    // The context goes into the system prompt, so cap it like a message.
    // (Length is checked on the raw field, before trimming.)
    if context.npc_prompt.as_ref().map_or(0, |s| s.chars().count()) > MAX_NPC_PROMPT_CHARS {
        return Err(ChatError::Invalid(format!(
            "npc_prompt is longer than {MAX_NPC_PROMPT_CHARS} characters"
        )));
    }
    if context.task.as_ref().map_or(0, |s| s.chars().count()) > MAX_TASK_CHARS {
        return Err(ChatError::Invalid(format!("task is longer than {MAX_TASK_CHARS} characters")));
    }
    if context.level_id.as_ref().map_or(0, |s| s.chars().count()) > MAX_LEVEL_ID_CHARS {
        return Err(ChatError::Invalid(format!(
            "level_id is longer than {MAX_LEVEL_ID_CHARS} characters"
        )));
    }
    Ok(())
}

fn system_prompt(language: &Language, context: &ChatContext) -> String {
    let name = language.name;
    // The base prompt below is the long-standing contract: plain text,
    // 1-3 short sentences, a "Tip:" correction line only when the learner
    // made a clear mistake. It is kept verbatim; the level context is only
    // appended after it, never edited into it.
    let mut prompt = format!(
        "You are a friendly {name} conversation partner for a learner whose first language is Chinese. \
Talk about everyday topics and keep the conversation going.\n\
Rules:\n\
- Write only in {name}, in simple natural sentences. Reply in 1 to 3 short sentences and end with a short question.\n\
- Plain text only: no markdown, lists, emoji or stage directions. Your reply is read aloud.\n\
- If the learner's last message has a clear mistake, add a last line that starts with \"{TIP_MARKER}\" and \
explains the correction briefly in Simplified Chinese, quoting the corrected {name} phrase. \
If there is no clear mistake, add no such line.\n\
- If the learner writes in Chinese, answer in {name} and help them say it in {name}."
    );
    if let Some(persona) = context.npc_prompt() {
        prompt.push_str(&format!(
            "\n\nRole-play: for this conversation you are this character, not a generic conversation partner:\n{persona}\nStay in character the whole time; the rules above still apply."
        ));
    }
    if let Some(level) = context.level_id() {
        prompt.push_str(&format!("\n\nThis conversation happens in level {level}."));
    }
    if let Some(task) = context.task() {
        prompt.push_str(&format!(
            "\n\nThe learner's task in this level is: {task}\nKeep steering the conversation toward the task. When the learner has clearly completed the task, end your reply with a line that contains only {TASK_DONE_MARKER}, on its own line after the Tip: line if there is one."
        ));
    }
    prompt
}

/// Removes every line that is exactly the task-done marker and reports
/// whether one was found. Runs on every reply so the marker can never leak
/// into the text the app reads aloud, even if the model emits it without
/// being asked; whether it is *reported* is decided in split_reply.
fn strip_task_done(content: &str) -> (String, bool) {
    let mut found = false;
    let mut kept = Vec::new();
    for line in content.lines() {
        if line.trim() == TASK_DONE_MARKER {
            found = true;
        } else {
            kept.push(line);
        }
    }
    (kept.join("\n"), found)
}

/// Separates the spoken reply from the correction note, and reports
/// whether the model signalled the level task is done. The marker is
/// stripped first so it can sit after the Tip: line and still be found.
fn split_reply(content: &str, context: &ChatContext) -> Reply {
    let (cleaned, marker_found) = strip_task_done(content);
    let mut reply = split_tip(&cleaned);
    reply.task_done = marker_found && context.task().is_some();
    reply
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
        return Reply { reply: tip, tip: None, task_done: false };
    }
    Reply { reply, tip: (!tip.is_empty()).then_some(tip), task_done: false }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spanish() -> &'static Language {
        &crate::config::LANGUAGES[1]
    }

    fn task_ctx() -> ChatContext {
        ChatContext { task: Some("Order a coffee.".into()), ..Default::default() }
    }

    #[test]
    fn splits_the_tip_from_the_reply() {
        let r = split_tip("I am fine, thanks. And you?\nTip: 应说 \"I am fine\"。");
        assert_eq!(r.reply, "I am fine, thanks. And you?");
        assert_eq!(r.tip.as_deref(), Some("应说 \"I am fine\"。"));
        assert!(!r.task_done);
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
        let ctx = ChatContext::default();
        assert!(validate(&[], &ctx).is_err());
        assert!(validate(&[Message { role: "system".into(), content: "x".into() }], &ctx).is_err());
        assert!(validate(&[Message { role: "assistant".into(), content: "x".into() }], &ctx).is_err());
        assert!(validate(&[user("hi")], &ctx).is_ok());
    }

    #[test]
    fn rejects_oversized_context() {
        let user = |c: &str| Message { role: "user".into(), content: c.into() };
        let msgs = [user("hi")];
        let ctx = ChatContext { npc_prompt: Some("x".repeat(2001)), ..Default::default() };
        assert!(validate(&msgs, &ctx).is_err());
        let ctx = ChatContext { task: Some("x".repeat(1001)), ..Default::default() };
        assert!(validate(&msgs, &ctx).is_err());
        let ctx = ChatContext { level_id: Some("x".repeat(65)), ..Default::default() };
        assert!(validate(&msgs, &ctx).is_err());
        assert!(validate(&msgs, &ChatContext::default()).is_ok());
    }

    #[test]
    fn prompt_without_context_is_unchanged() {
        // The base prompt is a contract: plain chat must get exactly the
        // prompt it always got, character for character.
        let expected = "You are a friendly Spanish conversation partner for a learner whose first language is Chinese. \
Talk about everyday topics and keep the conversation going.\n\
Rules:\n\
- Write only in Spanish, in simple natural sentences. Reply in 1 to 3 short sentences and end with a short question.\n\
- Plain text only: no markdown, lists, emoji or stage directions. Your reply is read aloud.\n\
- If the learner's last message has a clear mistake, add a last line that starts with \"Tip:\" and \
explains the correction briefly in Simplified Chinese, quoting the corrected Spanish phrase. \
If there is no clear mistake, add no such line.\n\
- If the learner writes in Chinese, answer in Spanish and help them say it in Spanish.";
        assert_eq!(system_prompt(spanish(), &ChatContext::default()), expected);
    }

    #[test]
    fn prompt_with_persona_task_and_level() {
        let ctx = ChatContext {
            level_id: Some("1-3".into()),
            npc_prompt: Some("You are Carmen, owner of Café Madrid.".into()),
            task: Some("Order a coffee and ask for the bill.".into()),
        };
        let p = system_prompt(spanish(), &ctx);
        assert!(p.contains("You are Carmen, owner of Café Madrid."));
        assert!(p.contains("level 1-3"));
        assert!(p.contains("Order a coffee and ask for the bill."));
        assert!(p.contains("[TASK_DONE]"));
        // The base contract survives the additions.
        assert!(p.contains("starts with \"Tip:\""));
        assert!(p.contains("You are a friendly Spanish conversation partner"));
    }

    #[test]
    fn prompt_with_persona_only_has_no_marker() {
        let ctx = ChatContext { npc_prompt: Some("You are Carmen.".into()), ..Default::default() };
        let p = system_prompt(spanish(), &ctx);
        assert!(p.contains("You are Carmen."));
        assert!(!p.contains("[TASK_DONE]"));
    }

    #[test]
    fn strips_marker_after_tip() {
        let r = split_reply(
            "¡Claro! ¿Qué quieres tomar?\nTip: 应说 \"Quiero un café\"。\n[TASK_DONE]",
            &task_ctx(),
        );
        assert_eq!(r.reply, "¡Claro! ¿Qué quieres tomar?");
        assert_eq!(r.tip.as_deref(), Some("应说 \"Quiero un café\"。"));
        assert!(r.task_done);
        assert!(!r.reply.contains("TASK_DONE"));
    }

    #[test]
    fn marker_without_task_is_stripped_but_not_reported() {
        let r = split_reply("Hola, ¿qué tal?\n[TASK_DONE]", &ChatContext::default());
        assert_eq!(r.reply, "Hola, ¿qué tal?");
        assert!(!r.task_done);
    }

    #[test]
    fn marker_mid_reply_is_stripped() {
        let r = split_reply("¡Muy bien!\n[TASK_DONE]\n¿Quieres algo más?", &task_ctx());
        assert_eq!(r.reply, "¡Muy bien! ¿Quieres algo más?");
        assert!(r.task_done);
    }

    #[test]
    fn no_marker_means_not_done() {
        let r = split_reply("Hola, ¿qué tal?", &task_ctx());
        assert!(!r.task_done);
    }

    #[test]
    fn marker_only_reply_is_empty() {
        // reply() turns this into an upstream error; split_reply itself
        // just reports the empty text plus the flag.
        let r = split_reply("[TASK_DONE]", &task_ctx());
        assert!(r.reply.is_empty());
        assert!(r.task_done);
    }

    #[test]
    fn reply_serializes_task_done() {
        let r = split_reply("¡Perfecto!\n[TASK_DONE]", &task_ctx());
        let v = serde_json::to_value(&r).unwrap();
        assert_eq!(v["task_done"], true);
        assert_eq!(v["reply"], "¡Perfecto!");
        assert_eq!(v["tip"], serde_json::Value::Null);
    }
}
