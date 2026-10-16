mod common;

use common::{fake_freshrss, MockServer, Req, Resp, ITEM_ID};
use freshrss_tts::freshrss::FreshRss;

fn form(req: &Req) -> Vec<(String, String)> {
    req.body
        .split('&')
        .filter(|p| !p.is_empty())
        .map(|p| {
            let (k, v) = p.split_once('=').unwrap();
            (
                k.to_string(),
                v.replace("%2F", "/")
                    .replace("%3A", ":")
                    .replace("%2C", ",")
                    .replace("%40", "@"),
            )
        })
        .collect()
}

#[test]
fn login_then_label_stream() {
    let server = MockServer::start(fake_freshrss);
    let mut fr = FreshRss::new(&server.url, "me", "apipass");
    let entries = fr.label_entries("narrate-scary", 5).unwrap();

    let e = &entries[0];
    assert_eq!(
        (e.title.as_str(), e.url.as_str(), e.html.as_str(), e.feed.as_str()),
        (
            "The Lighthouse Keeper",
            "https://example.org/story",
            "<p>Hello.</p>",
            "NoSleep"
        )
    );
    assert_eq!(e.id, ITEM_ID);
    assert_eq!(e.labels, ["narrate-scary"]);

    let reqs = server.requests();
    assert_eq!(reqs[0].path(), "/api/greader.php/accounts/ClientLogin");
    assert_eq!(
        form(&reqs[0]),
        [("Email".into(), "me".into()), ("Passwd".into(), "apipass".into())]
    );
    let stream = &reqs[1];
    assert_eq!(stream.headers["authorization"], "GoogleLogin auth=me/abc");
    assert!(stream
        .target
        .starts_with("/api/greader.php/reader/api/0/stream/contents/user/-/label/narrate-scary?"));
    assert!(stream.target.contains("output=json") && stream.target.contains("n=5"));
}

#[test]
fn labels_with_spaces_are_url_encoded_and_login_happens_once() {
    let server = MockServer::start(fake_freshrss);
    let mut fr = FreshRss::new(&server.url, "me", "pw");
    fr.label_entries("night reads", 1).unwrap();
    fr.label_entries("night reads", 1).unwrap();
    let reqs = server.requests();
    assert!(
        reqs[1].target.contains("/label/night%20reads?"),
        "{}",
        reqs[1].target
    );
    assert_eq!(
        reqs.iter().filter(|r| r.path().ends_with("ClientLogin")).count(),
        1
    );
}

#[test]
fn bad_password_is_an_error() {
    let server = MockServer::start(|_| Resp::status(401, "Unauthorized"));
    let err = FreshRss::new(&server.url, "me", "bad").login().unwrap_err();
    assert!(err.to_string().contains("401"), "{err}");
}

#[test]
fn login_without_auth_line_is_an_error() {
    let server = MockServer::start(|_| Resp::text("SID=x\n"));
    let err = FreshRss::new(&server.url, "me", "pw").login().unwrap_err();
    assert!(err.to_string().contains("Auth="), "{err}");
}

#[test]
fn edit_labels_sends_token_repeated_ids_add_and_remove() {
    let server = MockServer::start(fake_freshrss);
    let mut fr = FreshRss::new(&server.url, "me", "pw");
    fr.edit_labels(
        &["id1".to_string(), "id2".to_string()],
        &["narrated"],
        &["narrate", "narrate-scary"],
    )
    .unwrap();
    let post = server
        .requests()
        .into_iter()
        .find(|r| r.path().ends_with("/edit-tag"))
        .unwrap();
    assert_eq!(post.headers["authorization"], "GoogleLogin auth=me/abc");
    let got: Vec<(String, String)> = form(&post);
    let want = [
        ("T", "tok123"),
        ("i", "id1"),
        ("i", "id2"),
        ("a", "user/-/label/narrated"),
        ("r", "user/-/label/narrate"),
        ("r", "user/-/label/narrate-scary"),
    ];
    assert_eq!(got, want.map(|(k, v)| (k.to_string(), v.to_string())));
}

#[test]
fn edit_labels_unexpected_answer_errors_and_empty_is_a_noop() {
    let server = MockServer::start(|req| match req.path() {
        p if p.ends_with("ClientLogin") => Resp::text("Auth=x"),
        p if p.ends_with("/token") => Resp::text("t"),
        _ => Resp::text("Error"),
    });
    let mut fr = FreshRss::new(&server.url, "me", "pw");
    assert!(fr.edit_labels(&["a".into()], &["x"], &[]).is_err());
    let before = server.requests().len();
    fr.edit_labels(&[], &["x"], &[]).unwrap();
    assert_eq!(server.requests().len(), before);
}
