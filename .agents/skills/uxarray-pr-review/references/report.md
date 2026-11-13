# Review report

## The briefing (for the user only, never posted)

Before the draft, a few lines on where the PR stands. Leave out any line with nothing to say.

```
State:          rounds so far (who, verdict, at which commit), who is still requested, CI
Earlier points: one line each, e.g. "Sevans711 #2 regression test: addressed in a03c46b6, verified"
Scope:          drift from the issue or the original description, if any
Needs your call: disagreements or unanswered questions, both positions in one line each
```

## The draft shown to the user

Show exactly what will be posted, nothing more, followed by one short line on the verification behind it.
If other reviewers have already commented, the summary says which of their points you verified as
addressed. Replies to existing threads are listed separately from new inline comments.
Keep the long evidence (tables, scripts, dropped candidates) in a scratch file and mention its path; the
user can ask for it.

```
Event: COMMENT | APPROVE | REQUEST_CHANGES   (default COMMENT; the user decides)

Summary (posted as the review body)
  - One sentence: does it do what the linked issue asks?
  - 2-4 bullets: what was verified, and how.
  - One line: what is requested before merge (or "nothing").

Inline comments (at most 5, ordered by severity)
  file:line — one comment each:
    the problem, the evidence (what was run and what happened), the ask.
    Mark optional ones "nit" or "non-blocking". Mark unverified ones as not reproduced.
```

Style rules:

- Under ~80 words per inline comment. No praise filler, no restating the diff, no long bullet lists:
  UXarray reviewers push back on LLM-style verbosity, and that applies to reviews too.
- One specific sentence on what the PR does well is welcome; skip it if there is nothing specific.
- Only CONFIRMED and PLAUSIBLE findings. Say which, in plain words ("I reproduced…", "not reproduced, needs
  YAC").
- Give evidence a maintainer can rerun: the snippet and its output, or the mutated hunk and the test that
  still passed.
- Things the author need not change go in the same comment as "no change needed", or are left out.
- Problems that predate the PR are fine to mention only when the PR touches that exact code; say they
  predate it, and suggest a separate issue rather than growing the PR.

## Posting

Only after the user approves the text.

1. Check the PR head has not moved since the review: `gh api repos/UXARRAY/uxarray/pulls/<N> -q .head.sha`
   must equal the commit you reviewed. If it moved, tell the user and re-verify.
2. Check every inline anchor is inside a diff hunk on the new side (GitHub rejects others):
   `gh api repos/UXARRAY/uxarray/pulls/<N>/files -q '.[] | .filename + " " + ([.patch|scan("@@ [^@]+ @@")]|join(" "))'`
3. Post one review (not separate comments) from a JSON file:

```json
{
  "commit_id": "<head sha>",
  "event": "COMMENT",
  "body": "<summary>",
  "comments": [{"path": "uxarray/grid/x.py", "line": 123, "side": "RIGHT", "body": "<comment>"}]
}
```

```bash
gh api -X POST repos/UXARRAY/uxarray/pulls/<N>/reviews --input review.json -q '.html_url'
```

4. Post replies to existing inline threads separately (they are not part of the review JSON):
   `gh api -X POST repos/UXARRAY/uxarray/pulls/<N>/comments/<comment-id>/replies -f body='<reply>'`.
   For points raised in a review body (no thread), mention them in your summary instead. Don't resolve
   threads you didn't start.
5. Report the review URL to the user and confirm each inline comment landed on the intended line.
