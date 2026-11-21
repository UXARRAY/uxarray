"""Exercise the actual privileged comment script without network or PR code."""

import json
import shutil
import subprocess
import unittest
from pathlib import Path

WORKFLOW = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/asv-benchmarking-comment.yml"
)


def render(results, comments=None):
    """Run the workflow's inline JavaScript with mocked artifact and API calls."""
    text = WORKFLOW.read_text(encoding="utf-8")
    script = text.split("          script: |\n", 1)[1]
    script = "\n".join(line[12:] for line in script.splitlines())
    harness = r"""
const results = JSON.parse(process.argv[1]);
const comments = JSON.parse(process.argv[2]);
const calls = [];
const fs = {readFileSync(path) {
  if (path.endsWith('pr_number.txt')) return '42\n';
  if (path.endsWith('asv_compare_results.txt')) return results;
  throw new Error('Unexpected read: ' + path);
}};
const require = name => {
  if (name !== 'fs') throw new Error('Unexpected require: ' + name);
  return fs;
};
const context = {repo: {owner: 'UXARRAY', repo: 'uxarray'}};
const github = {rest: {issues: {
  async listComments(args) {return {data: comments};},
  async updateComment(args) {calls.push({operation: 'update', ...args});},
  async createComment(args) {calls.push({operation: 'create', ...args});},
}}};
(async () => {
SCRIPT
  console.log(JSON.stringify(calls));
})().catch(error => {console.error(error); process.exit(1);});
""".replace("SCRIPT", script)
    run = subprocess.run(
        ["node", "-e", harness, json.dumps(results), json.dumps(comments or [])],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(run.stdout)[0]


def section(title, marker):
    """Create a small ASV-shaped table with a unique benchmark marker."""
    return f"{title}\n\n| Change | Ratio | Benchmark |\n|---|---|---|\n| + | 14.0 | {marker} |\n"


@unittest.skipUnless(shutil.which("node"), "Node.js is needed for github-script tests")
class TestAsvComment(unittest.TestCase):
    def test_regressions_first_and_only_unchanged_collapsed(self):
        before = section("Benchmarks that have improved:", "faster")
        same = section("Benchmarks that have stayed the same:", "stable")
        worse = section("Benchmarks that have got worse:", "slower")
        other = section("Benchmarks that are not comparable:", "unknown")
        body = render(f"{before}\n{same}\n{worse}\n{other}")["body"]
        self.assertLess(body.index("slower"), body.index("faster"))
        self.assertLess(body.index("faster"), body.index("<details>"))
        self.assertLess(body.index("unknown"), body.index("<details>"))
        self.assertGreater(body.index("stable"), body.index("<details>"))
        self.assertEqual(body.count("<details>"), 1)
        for block in (before, same, worse, other):
            self.assertIn(block.strip(), body)

    def test_missing_categories_and_crlf(self):
        for title in (
            "Benchmarks that have got worse:",
            "Benchmarks that have gotten worse:",
            "Benchmarks that have improved:",
            "Benchmarks that are not comparable:",
        ):
            with self.subTest(title=title):
                body = render(section(title, "result").replace("\n", "\r\n"))["body"]
                self.assertIn("result", body)
                self.assertNotIn("<details>", body)

    def test_unchanged_only(self):
        body = render(section("Benchmarks that have stayed the same:", "stable"))[
            "body"
        ]
        self.assertGreater(body.index("stable"), body.index("<details>"))
        self.assertIn("</details>", body)

    def test_unknown_and_empty_output_visible(self):
        for text in (
            "new ASV format\n| arbitrary | table |",
            "",
            "No benchmarks found.",
            "constructor",
            "__proto__",
        ):
            with self.subTest(text=text):
                body = render(text)["body"]
                self.assertTrue(body.startswith("## ASV Benchmarking"))
                self.assertIn(text.strip(), body)
                self.assertNotIn("<details>", body)

    def test_preamble_and_repeated_sections_preserved(self):
        text = "ASV diagnostic\n" + section("Benchmarks that have got worse:", "one")
        text += "\n" + section("Benchmarks that have got worse:", "two")
        body = render(text)["body"]
        for marker in ("ASV diagnostic", "one", "two"):
            self.assertEqual(body.count(marker), 1)
        self.assertNotIn("<details>", body)

    def test_unfamiliar_category_after_unchanged_is_visible(self):
        text = section("Benchmarks that have stayed the same:", "stable")
        text += "\n" + section("Benchmarks with a new status:", "future")
        body = render(text)["body"]
        self.assertLess(body.index("future"), body.index("<details>"))
        self.assertGreater(body.index("stable"), body.index("<details>"))

    def test_create_and_update_existing_bot_comment(self):
        created = render("result")
        self.assertEqual(created["operation"], "create")
        self.assertEqual(created["issue_number"], 42)
        self.assertEqual(created["owner"], "UXARRAY")
        self.assertEqual(created["repo"], "uxarray")
        comments = [
            {"id": 1, "user": {"login": "human"}, "body": "## ASV Benchmarking"},
            {
                "id": 2,
                "user": {"login": "github-actions[bot]"},
                "body": "other bot comment",
            },
            {
                "id": 3,
                "user": {"login": "github-actions[bot]"},
                "body": "## ASV Benchmarking old",
            },
        ]
        updated = render("new result", comments)
        self.assertEqual(updated["operation"], "update")
        self.assertEqual(updated["comment_id"], 3)
        self.assertIn("new result", updated["body"])


if __name__ == "__main__":
    unittest.main()
