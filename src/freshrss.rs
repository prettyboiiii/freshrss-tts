//! FreshRSS client over its Google Reader compatible API (`/api/greader.php`).
//!
//! Selection works through labels: put a story under the label `narrate` (or `narrate-scary`, ...)
//! from any FreshRSS client and the worker picks it up. Enable the API and set an "API password" in
//! FreshRSS (Settings > Profile). Endpoints and label/starred streams are the ones in FreshRSS's
//! `p/api/greader.php`. Only exercised against mocks so far; `freshrss-tts check` is the live test.

use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// Long form: `tag:google.com,2005:reader/item/<hex>`.
    pub id: String,
    pub title: String,
    pub url: String,
    pub html: String,
    pub feed: String,
    pub published: i64,
    pub labels: Vec<String>,
}

pub struct FreshRss {
    api: String,
    user: String,
    api_password: String,
    agent: ureq::Agent,
    auth: Option<String>,
    token: Option<String>,
}

/// Percent-encode everything except RFC 3986 unreserved characters.
pub fn percent_encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

impl FreshRss {
    pub fn new(base_url: &str, user: &str, api_password: &str) -> Self {
        Self {
            api: format!("{}/api/greader.php", base_url.trim_end_matches('/')),
            user: user.to_string(),
            api_password: api_password.to_string(),
            agent: ureq::AgentBuilder::new()
                .timeout_connect(Duration::from_secs(10))
                .timeout_read(Duration::from_secs(30))
                .build(),
            auth: None,
            token: None,
        }
    }

    pub fn login(&mut self) -> Result<()> {
        let resp = match self
            .agent
            .post(&format!("{}/accounts/ClientLogin", self.api))
            .send_form(&[("Email", &self.user), ("Passwd", &self.api_password)])
        {
            Ok(r) => r,
            Err(ureq::Error::Status(code, _)) => bail!("ClientLogin failed: HTTP {code}"),
            Err(e) => return Err(e).context("ClientLogin request failed"),
        };
        let body = resp.into_string()?;
        match body.lines().find_map(|l| l.strip_prefix("Auth=")) {
            Some(auth) => {
                self.auth = Some(auth.trim().to_string());
                Ok(())
            }
            None => bail!("ClientLogin response had no Auth= line"),
        }
    }

    fn auth_header(&mut self) -> Result<String> {
        if self.auth.is_none() {
            self.login()?;
        }
        Ok(format!("GoogleLogin auth={}", self.auth.as_deref().unwrap()))
    }

    fn write_token(&mut self) -> Result<String> {
        if self.token.is_none() {
            let h = self.auth_header()?;
            let t = self
                .agent
                .get(&format!("{}/reader/api/0/token", self.api))
                .set("Authorization", &h)
                .call()
                .context("token request failed")?
                .into_string()?;
            self.token = Some(t.trim().to_string());
        }
        Ok(self.token.clone().unwrap())
    }

    /// Newest-first entries carrying `label` (read or unread).
    pub fn label_entries(&mut self, label: &str, limit: usize) -> Result<Vec<Entry>> {
        let h = self.auth_header()?;
        let url = format!(
            "{}/reader/api/0/stream/contents/user/-/label/{}",
            self.api,
            percent_encode(label)
        );
        let reply: Value = self
            .agent
            .get(&url)
            .query("output", "json")
            .query("n", &limit.to_string())
            .set("Authorization", &h)
            .call()
            .with_context(|| format!("reading label {label:?}"))?
            .into_json()?;
        reply["items"]
            .as_array()
            .map(|items| items.iter().map(parse_entry).collect::<Result<Vec<_>>>())
            .unwrap_or_else(|| Ok(Vec::new()))
    }

    /// Add and/or remove labels on entries (ids in the long form returned by [`label_entries`]).
    pub fn edit_labels(&mut self, ids: &[String], add: &[&str], remove: &[&str]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let token = self.write_token()?;
        let h = self.auth_header()?;
        // edit-tag takes repeated keys: i=..&i=..&a=..&r=..
        let adds: Vec<String> = add.iter().map(|n| format!("user/-/label/{n}")).collect();
        let removes: Vec<String> = remove.iter().map(|n| format!("user/-/label/{n}")).collect();
        let mut form: Vec<(&str, &str)> = vec![("T", token.as_str())];
        form.extend(ids.iter().map(|i| ("i", i.as_str())));
        form.extend(adds.iter().map(|a| ("a", a.as_str())));
        form.extend(removes.iter().map(|r| ("r", r.as_str())));
        let answer = self
            .agent
            .post(&format!("{}/reader/api/0/edit-tag", self.api))
            .set("Authorization", &h)
            .send_form(&form)
            .context("edit-tag request failed")?
            .into_string()?;
        if answer.trim() != "OK" {
            bail!(
                "edit-tag answered {:?}",
                answer.trim().chars().take(80).collect::<String>()
            );
        }
        Ok(())
    }
}

fn parse_entry(item: &Value) -> Result<Entry> {
    let text = |v: &Value| v.as_str().unwrap_or_default().to_string();
    let link = item["canonical"]
        .get(0)
        .or_else(|| item["alternate"].get(0))
        .map(|l| text(&l["href"]))
        .unwrap_or_default();
    let body = if item["content"].is_object() {
        &item["content"]
    } else {
        &item["summary"]
    };
    let labels = item["categories"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|c| c.as_str()?.split_once("/label/").map(|(_, l)| l.to_string()))
        .collect();
    Ok(Entry {
        id: item["id"].as_str().context("entry without id")?.to_string(),
        title: text(&item["title"]),
        url: link,
        html: text(&body["content"]),
        feed: text(&item["origin"]["title"]),
        published: item["published"].as_i64().unwrap_or(0),
        labels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percent_encoding() {
        assert_eq!(percent_encode("narrate-scary"), "narrate-scary");
        assert_eq!(percent_encode("night reads/é"), "night%20reads%2F%C3%A9");
    }

    #[test]
    fn entry_parsing_prefers_content_over_summary_and_collects_labels() {
        let item = serde_json::json!({
            "id": "tag:google.com,2005:reader/item/00000000000000a1",
            "title": "The Lighthouse Keeper",
            "published": 1760000000,
            "canonical": [{"href": "https://example.org/story"}],
            "summary": {"content": "<p>teaser</p>"},
            "content": {"content": "<p>Full text.</p>"},
            "origin": {"title": "NoSleep"},
            "categories": ["user/-/state/com.google/reading-list", "user/-/label/narrate-scary"],
        });
        let e = parse_entry(&item).unwrap();
        assert_eq!(e.html, "<p>Full text.</p>");
        assert_eq!(e.labels, ["narrate-scary"]);
        assert_eq!(
            (e.url.as_str(), e.feed.as_str()),
            ("https://example.org/story", "NoSleep")
        );
        assert!(parse_entry(&serde_json::json!({"title": "x"})).is_err());
    }
}
