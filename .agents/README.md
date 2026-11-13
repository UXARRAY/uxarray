# Agent skills for UXarray

Reusable instructions for AI coding agents working on this repository. The repo-wide context every
agent should read is [`AGENTS.md`](../AGENTS.md) at the repo root; the skills here build on it.

| Skill | Use it to |
|---|---|
| [`uxarray-pr-review`](skills/uxarray-pr-review/SKILL.md) | Review a PR and verify every finding by running code before reporting it |
| [`uxarray-issue-triage`](skills/uxarray-issue-triage/SKILL.md) | Classify an issue, find duplicates and related work, try to reproduce it, and suggest labels |

Each skill is a folder with a `SKILL.md` (name, description, steps) and optional `references/` files that
the steps link to.

## Using them

- **Claude Code**: `.claude/skills/` holds symlinks to these folders, so the skills are available as
  `/uxarray-pr-review <PR number>` and `/uxarray-issue-triage <issue number>`. Claude Code reads
  `AGENTS.md` itself in recent versions (tested with 2.1.323; 2.1.270 does not); if it isn't loaded,
  run `claude update`.
- **Other agents**: agents that read `.agents/skills/` pick them up directly. For any other agent, point
  it at the skill's `SKILL.md`, e.g. "Follow `.agents/skills/uxarray-pr-review/SKILL.md` for PR 1889".

Windows: the `.claude/skills/` symlinks need `git config core.symlinks true` (and Developer Mode or admin
rights) before checkout; otherwise they appear as small text files.

## Ground rules shared by the skills

- Never post to GitHub, label, or push without the user's approval of the exact text.
- Work in git worktrees and a scratch environment; never switch branches in the user's checkout.
- Report a finding as fact only after reproducing it; otherwise say it was not reproduced and why.

## Adding a skill

Create `.agents/skills/<name>/SKILL.md` with `name` and `description` front matter, add a symlink
`ln -s ../../.agents/skills/<name> .claude/skills/<name>`, and add a row to the table above.
