#!/usr/bin/env python3
"""Prepare artifacts for the manual monthly UXarray release flow.

This helper is used by `.github/workflows/prepare-monthly-release.yml` to:
- compute the next calendar-based release tag
- generate draft release notes
- generate the release issue body from the existing issue template

It prepares text artifacts only. Publishing the GitHub Release still happens
manually, which then triggers the existing PyPI publish workflow. Conda-forge
steps remain manual.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

VERSION_RE = re.compile(r"^v(?P<year>\d{4})\.(?P<month>\d{2})\.(?P<patch>\d+)$")
PR_RE = re.compile(r"\(#(?P<number>\d+)\)")


@dataclass(frozen=True)
class VersionTag:
    raw: str
    year: int
    month: int
    patch: int


@dataclass(frozen=True)
class ReleaseEntry:
    sha: str
    title: str
    author: str
    pr_number: str | None
    category: str


def _run_git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _parse_version_tag(tag: str) -> VersionTag | None:
    match = VERSION_RE.match(tag)
    if not match:
        return None

    return VersionTag(
        raw=tag,
        year=int(match.group("year")),
        month=int(match.group("month")),
        patch=int(match.group("patch")),
    )


def _list_version_tags() -> list[VersionTag]:
    raw_tags = _run_git("tag", "--list")
    parsed_tags = []

    for raw_tag in raw_tags.splitlines():
        parsed = _parse_version_tag(raw_tag)
        if parsed is not None:
            parsed_tags.append(parsed)

    return sorted(parsed_tags, key=lambda tag: (tag.year, tag.month, tag.patch))


def _resolve_today(today_override: str | None) -> date:
    if today_override:
        return date.fromisoformat(today_override)

    return datetime.now(timezone.utc).date()


def _resolve_version(today: date, version_override: str | None) -> str:
    """Return the next release tag for the current month or an explicit override."""
    if version_override:
        if _parse_version_tag(version_override) is None:
            raise ValueError(
                "Version override must match the release tag format 'vYYYY.MM.PATCH'."
            )
        return version_override

    monthly_tags = [
        tag
        for tag in _list_version_tags()
        if tag.year == today.year and tag.month == today.month
    ]

    next_patch = monthly_tags[-1].patch + 1 if monthly_tags else 0
    return f"v{today.year:04d}.{today.month:02d}.{next_patch}"


def _latest_tag() -> str | None:
    version_tags = _list_version_tags()
    return version_tags[-1].raw if version_tags else None


def _categorize_title(title: str) -> str:
    lowered = title.lower()

    if any(token in lowered for token in ["doc", "readme", "notebook", "tutorial"]):
        return "docs"
    if any(
        token in lowered
        for token in ["fix", "bug", "correct", "error", "issue", "typo"]
    ):
        return "fixes"
    if any(token in lowered for token in ["test", "ci", "workflow", "pre-commit"]):
        return "testing"
    if any(
        token in lowered
        for token in ["add", "support", "implement", "introduce", "enable", "new"]
    ):
        return "features"
    return "maintenance"


def _collect_release_entries(previous_tag: str | None) -> list[ReleaseEntry]:
    log_range = f"{previous_tag}..HEAD" if previous_tag else "HEAD"
    raw_log = _run_git(
        "log",
        "--first-parent",
        "--pretty=format:%H%x1f%s%x1f%an",
        log_range,
    )

    entries = []
    for line in raw_log.splitlines():
        sha, title, author = line.split("\x1f")
        pr_match = PR_RE.search(title)
        entries.append(
            ReleaseEntry(
                sha=sha,
                title=title,
                author=author,
                pr_number=pr_match.group("number") if pr_match else None,
                category=_categorize_title(title),
            )
        )

    return entries


def _format_entry(entry: ReleaseEntry) -> str:
    if entry.pr_number is not None:
        return f"- {entry.title} by {entry.author}"

    return f"- {entry.title} ({entry.sha[:7]}) by {entry.author}"


def _build_release_notes(
    version: str, previous_tag: str | None, entries: list[ReleaseEntry]
) -> str:
    """Build editable release notes with highlights first and a flat change list."""
    top_candidates = entries[:5]
    contributor_counts = Counter(entry.author for entry in entries)

    lines = [f"## UXarray {version}", ""]
    if previous_tag:
        lines.append(f"_Changes since {previous_tag}_")
    else:
        lines.append("_Initial release prep draft_")
    lines.extend(["", "### Top Contributions"])

    if top_candidates:
        lines.append("- Curate the highlights below before publishing ✍️")
        lines.extend(_format_entry(entry) for entry in top_candidates[:3])
    else:
        lines.append("- Add this month's highlights here ✍️")

    lines.extend(["", "### What's Changed"])
    if entries:
        lines.extend(_format_entry(entry) for entry in entries)
    else:
        lines.append("- No merged changes found since the previous release")

    lines.extend(["", "### Contributors"])
    if contributor_counts:
        for author, count in contributor_counts.most_common(5):
            lines.append(f"- {author} ({count} commits)")
    else:
        lines.append("- No release entries found")

    return "\n".join(lines) + "\n"


def _extract_issue_template_body() -> str:
    template_path = Path(".github/ISSUE_TEMPLATE/release_request.md")
    raw_text = template_path.read_text()

    if raw_text.startswith("---\n"):
        parts = raw_text.split("---\n", 2)
        if len(parts) == 3:
            return parts[2].lstrip()

    return raw_text


def _build_release_issue(version: str, previous_tag: str | None, today: date) -> str:
    """Build the release tracking issue body from the repository template."""
    body = _extract_issue_template_body()
    body = body.replace(
        "Date of intended release:", f"Date of intended release: {today.isoformat()}"
    )

    context_lines = [f"Release version: `{version}`"]
    if previous_tag:
        context_lines.append(f"Previous release tag: `{previous_tag}`")

    context = "\n".join(context_lines)
    return f"{context}\n\n{body}".rstrip() + "\n"


def _build_summary(
    version: str, previous_tag: str | None, entries: list[ReleaseEntry]
) -> str:
    lines = [f"Version: {version}"]
    lines.append(f"Previous tag: {previous_tag or 'none'}")
    lines.append(f"Commits included: {len(entries)}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare local artifacts for the monthly UXarray release workflow."
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory for generated release artifacts."
    )
    parser.add_argument(
        "--version", help="Optional release tag override, e.g. v2026.03.1."
    )
    parser.add_argument("--today", help="Optional ISO date override for local testing.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    today = _resolve_today(args.today)
    version = _resolve_version(today=today, version_override=args.version)
    previous_tag = _latest_tag()
    entries = _collect_release_entries(previous_tag=previous_tag)

    release_notes = _build_release_notes(
        version=version, previous_tag=previous_tag, entries=entries
    )
    release_issue = _build_release_issue(
        version=version, previous_tag=previous_tag, today=today
    )
    summary = _build_summary(
        version=version, previous_tag=previous_tag, entries=entries
    )

    metadata = {
        "version": version,
        "previous_tag": previous_tag,
        "commit_count": len(entries),
        "generated_on": today.isoformat(),
    }

    (output_dir / "release-notes.md").write_text(release_notes)
    (output_dir / "release-issue.md").write_text(release_issue)
    (output_dir / "summary.txt").write_text(summary)
    (output_dir / "release-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print(summary, end="")


if __name__ == "__main__":
    main()
