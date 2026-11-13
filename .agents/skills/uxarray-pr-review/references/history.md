# Review history

A PR often has reviews, threads and new commits before you arrive. Read them before looking for
problems, so the review builds on the discussion instead of repeating it.

## Fetch everything in one query

```bash
gh api graphql -F n=<N> -f query='
query($n:Int!){repository(owner:"UXARRAY",name:"uxarray"){pullRequest(number:$n){
  headRefOid reviewDecision lastEditedAt isDraft
  labels(first:20){nodes{name}}
  reviews(first:100){nodes{author{login} state submittedAt commit{oid} body}}
  reviewThreads(first:100){nodes{isResolved isOutdated path line
    comments(first:50){nodes{databaseId author{login} createdAt body}}}}
  comments(first:100){nodes{author{login} createdAt body}}
  commits(last:100){nodes{commit{oid committedDate messageHeadline}}}
  userContentEdits(first:20){nodes{editedAt editor{login}}}
  timelineItems(first:100,itemTypes:[HEAD_REF_FORCE_PUSHED_EVENT,REVIEW_REQUESTED_EVENT,RENAMED_TITLE_EVENT]){nodes{
    __typename
    ... on HeadRefForcePushedEvent{createdAt}
    ... on ReviewRequestedEvent{createdAt requestedReviewer{... on User{login}}}
    ... on RenamedTitleEvent{previousTitle currentTitle}}}
}}}'
gh pr checks <N>
```

Each list returns at most 100 items (50 comments per thread) and the query does not paginate; for a PR
with more, add `pageInfo{hasNextPage endCursor}` and page with `after:`. `databaseId` is the REST id used
to reply to a thread (report.md §Posting).

Bots to skip, but read for what they tell you:
- `github-actions`: the **ASV Benchmarking** comment (present only with the `run-benchmark` label). Note
  which benchmarks changed and whether anyone discussed them.
- `pre-commit-ci`: lint results; it does not push fixes unless someone comments `pre-commit.ci autofix`.
- `review-notebook-app` (ReviewNB): notebook diffs; maintainers sometimes comment there instead of on
  the PR.
- Copilot reviews: treat its points as unverified suggestions; count one only if a human endorsed it or
  you verify it yourself.

## Build these from it

1. **Rounds.** Each review: who, verdict, date, and the commit it reviewed (`commit.oid`). Who is requested
   but has not reviewed yet. Two approvals is the norm before merge.
2. **Ledger of points raised.** One line per point, from review bodies, inline threads and comments:
   who raised it → what the author did (commit that addressed it, reply, nothing) → status:
   - **Addressed**: verify it in the current code (a mutation test or a quick check), don't trust "done".
   - **Open**: still unanswered or unresolved. Authors don't resolve threads (reviewers resolve their
     own), so an unresolved thread may be addressed but not yet confirmed by its reviewer.
   - **Declined**: the author pushed back. Summarize both positions neutrally.
   - **Superseded**: the code it referred to is gone (`isOutdated`).
   Reviewers mark points "nit" or "non-blocking"; keep that label in the ledger.
3. **What changed since the last review.** Commits after the last review's commit, and their diff
   restricted to the PR's files: `git diff <reviewed-oid> HEAD -- <files>`. Merges from `main` also show
   up here; set aside what came from the base (`git log --first-parent`). If the branch was force-pushed
   after a review, the reviewed commit may be gone: say so and review the whole PR.
4. **Scope drift.** Compare the original description and linked issue with what the PR does now. Edited
   descriptions (`userContentEdits`) and fix commits often move scope; note any that were not discussed.
   Check whether "Closes #N" still holds or should become "Related to #N".
5. **Where people agree and disagree.** Reviewer vs author, and reviewer vs reviewer. Points where they
   disagree, or questions nobody answered (e.g. "did you run the benchmarks?"), are the candidates for
   **Needs your call**. Decisions reported from maintainer meetings ("From internal discussion...",
   "Reportback") settle a point; don't reopen them.

## Use it in the review

- **Don't repeat** a point already raised. If you have new evidence about it, reply in that thread, or
  mention it in the summary with credit ("Sevans711's point 2, now fixed — verified by …").
- **Focus on the delta.** When the user or anyone else already reviewed, look hardest at the commits after
  that review: fixes made in response to a review are new, often unreviewed code.
- **Re-review.** If the current `gh` user reviewed before, check each of their own points (addressed? how?)
  before anything else, and keep the new review to what changed.
- **Explain stale text.** A PR description that no longer matches the code after review-driven changes is a
  routine "please update the description", not a mismatch. Say which change made it stale.
