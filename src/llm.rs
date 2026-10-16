//! Minimal OpenAI-compatible chat client for the director (llama-server, Ollama, or a hosted API).

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::time::Duration;

use crate::director::{Completer, Message};

pub struct ChatClient {
    base_url: String,
    pub model: String,
    api_key: Option<String>,
    pub temperature: f64,
    pub max_tokens: u32,
    /// Extra top-level request fields. The default turns off "thinking" on hybrid models such as
    /// Qwen3 (llama-server honours `chat_template_kwargs`; other servers ignore unknown fields).
    pub extra_body: Value,
    agent: ureq::Agent,
}

impl ChatClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: "local".into(),
            api_key: None,
            temperature: 0.2,
            max_tokens: 4096,
            extra_body: json!({"chat_template_kwargs": {"enable_thinking": false}}),
            agent: ureq::AgentBuilder::new()
                .timeout_connect(Duration::from_secs(10))
                .timeout_read(Duration::from_secs(900))
                .build(),
        }
    }

    pub fn with_model(mut self, model: Option<String>) -> Self {
        if let Some(m) = model {
            self.model = m;
        }
        self
    }

    pub fn with_api_key(mut self, key: Option<String>) -> Self {
        self.api_key = key.filter(|k| !k.is_empty());
        self
    }

    pub fn request_body(&self, messages: &[Message], schema: &Value) -> Value {
        let mut body = json!({
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            // llama.cpp documents `schema` directly under response_format; OpenAI nests it under
            // `json_schema`. Sending both keeps either kind of server constrained.
            "response_format": {
                "type": "json_schema",
                "schema": schema,
                "json_schema": {"name": "directions", "schema": schema},
            },
        });
        if let (Some(obj), Some(extra)) = (body.as_object_mut(), self.extra_body.as_object()) {
            obj.extend(extra.clone());
        }
        body
    }
}

impl Completer for ChatClient {
    fn chat_json(&self, messages: &[Message], schema: &Value) -> Result<String> {
        let mut req = self.agent.post(&format!("{}/v1/chat/completions", self.base_url));
        if let Some(k) = &self.api_key {
            req = req.set("Authorization", &format!("Bearer {k}"));
        }
        let reply: Value = req
            .send_json(self.request_body(messages, schema))
            .context("director request failed")?
            .into_json()
            .context("director reply was not JSON")?;
        match reply.pointer("/choices/0/message/content") {
            Some(Value::String(s)) => Ok(s.clone()),
            Some(Value::Null) | None => Ok(String::new()),
            Some(other) => bail!("unexpected content in director reply: {other}"),
        }
    }
}
