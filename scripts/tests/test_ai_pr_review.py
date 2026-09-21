import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class FakeOpenAI:
    init_kwargs = None
    request_kwargs = None

    def __init__(self, **kwargs):
        type(self).init_kwargs = kwargs
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        type(self).request_kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"summary": "No findings", "findings": []}'))]
        )


def load_review_module():
    script_path = Path(__file__).resolve().parents[1] / "ai_pr_review.py"
    module_name = "ai_pr_review_under_test"
    previous_openai = sys.modules.get("openai")
    sys.modules["openai"] = SimpleNamespace(OpenAI=FakeOpenAI)

    try:
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_openai is None:
            sys.modules.pop("openai", None)
        else:
            sys.modules["openai"] = previous_openai


class DeepSeekReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review = load_review_module()

    def test_calls_deepseek_v4_flash_through_the_deepseek_endpoint(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=True):
            response = self.review.call_deepseek("review this diff")

        self.assertEqual(response, '{"summary": "No findings", "findings": []}')
        self.assertEqual(FakeOpenAI.init_kwargs, {
            "api_key": "test-key",
            "base_url": "https://api.deepseek.com",
        })
        self.assertEqual(FakeOpenAI.request_kwargs["model"], "deepseek-v4-flash")

    def test_selects_deepseek_as_the_reviewer(self):
        reviewer_name, call_reviewer = self.review.get_reviewer()

        self.assertEqual(reviewer_name, "DeepSeek V4 Flash")
        self.assertIs(call_reviewer, self.review.call_deepseek)

    def test_labels_inline_findings_with_the_deepseek_reviewer_name(self):
        review = self.review.build_github_review(
            {
                "summary": "One issue",
                "findings": [
                    {
                        "path": "app.py",
                        "line": 4,
                        "severity": "high",
                        "body": "Handle this failure.",
                    }
                ],
            },
            {"app.py": {4}},
            False,
            "commit-sha",
        )

        self.assertEqual(review["comments"][0]["body"], "DeepSeek V4 Flash high: Handle this failure.")


def one_hunk_file(path, body_lines=2, body_len=200):
    body = "\n".join("+ " + ("x" * body_len) for _ in range(body_lines))
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1,{body_lines} +1,{body_lines} @@\n{body}\n"


def many_hunk_file(path, n_hunks, per_hunk=10, body_len=200):
    parts = [f"diff --git a/{path} b/{path}", f"--- a/{path}", f"+++ b/{path}"]
    for index in range(n_hunks):
        start = index * (per_hunk + 1) + 1
        parts.append(f"@@ -{start},{per_hunk} +{start},{per_hunk} @@")
        parts.extend("+ " + ("x" * body_len) for _ in range(per_hunk))
    return "\n".join(parts) + "\n"


class DiffChunkingTests(unittest.TestCase):
    """The partitioning replaced truncation, so these guard the properties that
    made it worth doing: full coverage, no mid-hunk cut, and honest labelling
    of which parts the model could actually see."""

    @classmethod
    def setUpClass(cls):
        cls.review = load_review_module()

    def test_small_diff_is_a_single_uncut_part(self):
        parts, capped, _ = self.review.chunk_diff(one_hunk_file("a.ts", 2))

        self.assertEqual(len(parts), 1)
        self.assertFalse(parts[0][1])
        self.assertFalse(capped)

    def test_multi_file_diff_partitions_without_losing_content(self):
        diff = "".join(one_hunk_file(f"src/file{i}.ts", 40) for i in range(20))

        parts, capped, _ = self.review.chunk_diff(diff)

        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(text) <= self.review.MAX_DIFF_CHARS for text, _ in parts))
        self.assertTrue(all(not cut for _, cut in parts))
        self.assertFalse(capped)
        self.assertEqual(len("".join(text for text, _ in parts)), len(diff))

    def test_oversized_single_file_splits_at_hunk_boundaries(self):
        parts, capped, _ = self.review.chunk_diff(many_hunk_file("src/huge.ts", 400))

        self.assertGreater(len(parts), 1)
        self.assertTrue(all(text.startswith("diff --git a/src/huge.ts") for text, _ in parts))
        self.assertTrue(all(not cut for _, cut in parts))
        self.assertFalse(capped)

    def test_unsplittable_hunk_is_cut_on_a_line_boundary(self):
        parts, _, _ = self.review.chunk_diff(one_hunk_file("src/giant.ts", 4000))

        self.assertTrue(parts[0][1], "a hunk larger than the budget must be flagged as cut")
        last_line = parts[0][0].rstrip("\n").split("\n")[-1]
        self.assertEqual(len(last_line), 202, "the model must never see a half-written line")

    def test_oversized_unit_is_cut_at_a_line_boundary_not_mid_line(self):
        # A unit always begins with a diff header, so a newline is guaranteed
        # inside the budget and the cut lands before the oversized content
        # rather than part-way through a line.
        diff = one_hunk_file("src/long.ts", 1, body_len=self.review.MAX_DIFF_CHARS + 100)

        parts, capped, _ = self.review.chunk_diff(diff)

        self.assertTrue(parts[0][1], "an oversized unit must be flagged as cut")
        self.assertFalse(capped)
        text = parts[0][0]
        self.assertTrue(text.endswith("\n"))
        for line in text.split("\n"):
            self.assertLessEqual(
                len(line),
                self.review.MAX_DIFF_CHARS,
                "no line may be a partial fragment of source",
            )
        self.assertNotIn("xxxx", text, "the oversized body line should be excluded, not halved")

    def test_git_quoted_paths_are_recognised_as_file_boundaries(self):
        # Git quotes paths with special characters; treating one as a
        # continuation of the previous file would merge two files into one unit.
        quoted = (
            'diff --git "a/src/my file.ts" "b/src/my file.ts"\n'
            '--- "a/src/my file.ts"\n'
            '+++ "b/src/my file.ts"\n'
            "@@ -1,1 +1,1 @@\n+x\n"
        )
        diff = one_hunk_file("src/first.ts", 2) + quoted

        parts, _, _ = self.review.chunk_diff(diff)
        joined = "".join(text for text, _ in parts)

        self.assertEqual(
            joined.count("diff --git"),
            2,
            "both the plain and the quoted file must be separate sections",
        )
        self.assertNotIn(
            'src/first.ts" "b/src/my file.ts',
            joined,
            "the quoted header must not be glued onto the previous file",
        )

    def test_cap_limits_parts_without_falsely_marking_a_complete_part_cut(self):
        diff = "".join(many_hunk_file(f"src/f{i}.ts", 60) for i in range(60))

        parts, capped, _ = self.review.chunk_diff(diff)

        self.assertTrue(capped)
        self.assertEqual(len(parts), self.review.MAX_REVIEW_PARTS)
        self.assertTrue(
            all(not cut for _, cut in parts),
            "a part that was reviewed in full must not be described to the model as truncated",
        )

    def test_capped_chunking_reports_the_total_part_count_before_the_cap(self):
        diff = "".join(many_hunk_file(f"src/f{i}.ts", 60) for i in range(60))

        parts, capped, total_part_count = self.review.chunk_diff(diff)

        self.assertTrue(capped)
        self.assertEqual(len(parts), self.review.MAX_REVIEW_PARTS)
        self.assertGreater(total_part_count, len(parts))


class ReviewParsingTests(unittest.TestCase):
    """Aggregation iterates parse_review_json's findings without a type check,
    so the normalisation it performs is load-bearing and worth pinning."""

    @classmethod
    def setUpClass(cls):
        cls.review = load_review_module()

    def test_parse_review_json_always_yields_a_findings_list(self):
        for raw in (
            '{"summary": "x", "findings": null}',
            '{"summary": "x"}',
            '{"summary": "x", "findings": "oops"}',
            '{"summary": "x", "findings": []}',
        ):
            with self.subTest(raw=raw):
                parsed = self.review.parse_review_json(raw, "R")
                self.assertIsInstance(parsed["findings"], list)

class MergePartsTests(unittest.TestCase):
    """The aggregation behind one-request-per-part, asserted against production
    code rather than a re-implementation of its own expression."""

    @classmethod
    def setUpClass(cls):
        cls.review = load_review_module()

    def test_findings_from_every_part_are_kept(self):
        parts = [
            {"summary": "a", "findings": [{"path": "a.py", "line": 1, "severity": "medium", "body": "one"}]},
            {"summary": "b", "findings": [{"path": "b.py", "line": 2, "severity": "high", "body": "two"}]},
        ]

        merged = self.review.merge_parts(parts, 2, False, [])

        self.assertEqual([f["body"] for f in merged["findings"]], ["one", "two"])

    def test_summary_dedupes_repeated_part_summaries_and_notes_the_part_count(self):
        parts = [{"summary": "same", "findings": []}, {"summary": "same", "findings": []}]

        merged = self.review.merge_parts(parts, 2, False, [])

        self.assertIn("Split into 2 parts", merged["summary"])
        self.assertEqual(merged["summary"].count("same"), 1)

    def test_capped_is_reported_in_the_summary(self):
        merged = self.review.merge_parts([{"summary": "a", "findings": []}], 20, True, [])

        self.assertIn("Only the first", merged["summary"])
        self.assertIn("remainder was not", merged["summary"])
        self.assertIn("20", merged["summary"], "the cap note must match the count it was given")

    def test_cap_alone_does_not_claim_truncation(self):
        # A capped run has no cut part, so the truncation note would assert a
        # cause that did not happen.
        merged = self.review.merge_parts([{"summary": "a", "findings": []}], 12, True, [], False)

        self.assertIn("Only the first", merged["summary"])
        self.assertNotIn("truncated", merged["summary"].lower())

    def test_capped_summary_reports_reviewed_and_total_part_counts(self):
        merged = self.review.merge_parts(
            [{"summary": "a", "findings": []}],
            12,
            True,
            [],
            total_part_count=14,
        )

        self.assertIn("Only the first 12 of 14 parts were reviewed", merged["summary"])
        self.assertNotIn("Split into 12 parts", merged["summary"])

    def test_a_genuine_cut_does_claim_truncation(self):
        merged = self.review.merge_parts([{"summary": "a", "findings": []}], 1, False, [], True)

        self.assertIn("truncated", merged["summary"].lower())

    def test_failure_count_cannot_contradict_the_reviewed_count(self):
        merged = self.review.merge_parts(
            [{"summary": "a", "findings": []}], 12, False, ["p1", "p2", "p3", "p4", "p5"]
        )

        self.assertIn("5 of 12 part(s) could not be reviewed", merged["summary"])
        self.assertNotIn(
            "Reviewed in 12",
            merged["summary"],
            "the summary must not claim to have reviewed parts it did not",
        )

    def test_part_failures_are_counted_without_publishing_exception_text(self):
        merged = self.review.merge_parts(
            [{"summary": "a", "findings": []}], 3, False, ["part 2/3: https://api.example/bad?token=x"]
        )

        self.assertIn("could not be reviewed", merged["summary"])
        self.assertNotIn(
            "token=x",
            merged["summary"],
            "this summary is posted publicly, so exception text must not leak into it",
        )

    def test_a_null_summary_does_not_raise_and_falls_back(self):
        merged = self.review.merge_parts([{"summary": None, "findings": []}], 1, False, [])

        self.assertIsInstance(merged["summary"], str)
        self.assertTrue(merged["summary"])


class FailureArtifactTests(unittest.TestCase):
    """The failure artifact is posted as a public PR comment, so it must never
    carry the raw reviewer error."""

    @classmethod
    def setUpClass(cls):
        cls.review = load_review_module()

    def test_failure_artifact_does_not_publish_reviewer_error_text(self):
        secret = "https://api.deepseek.com/v1?key=SECRET-abc"
        diff_path = Path("/tmp/ai-pr-review-leak-test.diff")
        diff_path.write_text(
            "diff --git a/a.ts b/a.ts\n--- a/a.ts\n+++ b/a.ts\n@@ -1 +1 @@\n+x\n",
            encoding="utf-8",
        )
        out_path = Path("/tmp/ai-pr-review-leak-test.out")

        class ExplodingOpenAI:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=self)

            def create(self, **kwargs):
                raise RuntimeError(secret)

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=True):
            with patch.object(self.review, "OpenAI", ExplodingOpenAI):
                argv = sys.argv
                sys.argv = ["ai_pr_review.py", "--diff", str(diff_path), "--output", str(out_path)]
                try:
                    with self.assertRaises(SystemExit) as raised:
                        self.review.main()
                    self.assertEqual(raised.exception.code, 1)
                finally:
                    sys.argv = argv

        body = out_path.read_text(encoding="utf-8")
        self.assertIn("failed before producing findings", body)
        self.assertNotIn("SECRET-abc", body, "the reviewer error must stay in the log")
        self.assertNotIn("api.deepseek.com", body, "exception text must not be published")


class ReviewableLinesTests(unittest.TestCase):
    """chunk_diff repeats a split file's header in every part, so the helper
    must accumulate lines per path instead of resetting on each header."""

    @classmethod
    def setUpClass(cls):
        cls.review = load_review_module()

    def test_duplicate_file_headers_merge_rather_than_dropping_lines(self):
        diff = many_hunk_file("src/huge.ts", 400)
        parts, _, _ = self.review.chunk_diff(diff)
        self.assertGreater(len(parts), 1, "this diff must actually split for the test to mean anything")
        joined = "".join(text for text, _ in parts)

        from_joined = self.review.reviewable_lines_by_file(joined)
        from_single = self.review.reviewable_lines_by_file(diff)

        self.assertEqual(
            from_joined,
            from_single,
            "splitting a file into parts must not drop its earlier hunks' lines",
        )
        self.assertTrue(from_joined["src/huge.ts"])


class PublishedBodyTests(unittest.TestCase):
    """The non-commit artifact is posted verbatim as the PR comment, so its
    format and its disclosures are what a reviewer actually reads."""

    @classmethod
    def setUpClass(cls):
        cls.review = load_review_module()

    def _run(self, diff_text, content):
        diff_path = Path("/tmp/ai-pr-review-body.diff")
        diff_path.write_text(diff_text, encoding="utf-8")
        out_path = Path("/tmp/ai-pr-review-body.out")

        class Fixed:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=self)

            def create(self, **kwargs):
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
                )

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=True):
            with patch.object(self.review, "OpenAI", Fixed):
                argv = sys.argv
                sys.argv = ["ai_pr_review.py", "--diff", str(diff_path), "--output", str(out_path)]
                try:
                    self.review.main()
                finally:
                    sys.argv = argv

        return out_path.read_text(encoding="utf-8")

    def test_body_is_markdown_not_a_json_blob(self):
        body = self._run(
            one_hunk_file("src/small.ts", 2),
            json.dumps({"summary": "looks fine", "findings": []}),
        )

        self.assertIn("## AI PR Review", body)
        self.assertIn("looks fine", body)
        self.assertNotIn('"findings"', body, "the published body must be prose, not raw JSON")
        self.assertIn("Reviewed by DeepSeek V4 Flash", body)

    def test_findings_are_rendered_as_readable_bullets(self):
        body = self._run(
            one_hunk_file("src/small.ts", 2),
            json.dumps(
                {
                    "summary": "one issue",
                    "findings": [
                        {"path": "a.ts", "line": 3, "severity": "high", "body": "fix it"}
                    ],
                }
            ),
        )

        self.assertIn("### Findings (1)", body)
        self.assertIn("`a.ts:3`", body)
        self.assertIn("(high)", body)
        self.assertIn("fix it", body)

    def test_a_cut_review_discloses_truncation_without_relying_on_the_model(self):
        # The prompt asks the model to mention partiality, but that is not a
        # guarantee; the summary must carry it structurally.
        body = self._run(
            one_hunk_file("src/giant.ts", 4000),
            json.dumps({"summary": "all good", "findings": []}),
        )

        self.assertIn("truncated", body.lower())

    def test_a_capped_review_discloses_the_total_and_reviewed_part_counts(self):
        diff = "".join(many_hunk_file(f"src/f{i}.ts", 60, per_hunk=60) for i in range(60))

        body = self._run(
            diff,
            json.dumps({"summary": "all good", "findings": []}),
        )

        self.assertRegex(body, r"Only the first 12 of \d+ parts were reviewed")
        self.assertNotIn("Split into 12 parts", body)


class PartialFailureOutputTests(unittest.TestCase):
    """A lone surviving part must never be published as a complete review."""

    @classmethod
    def setUpClass(cls):
        cls.review = load_review_module()

    def test_lone_surviving_part_is_disclosed(self):
        reviewer_body = json.dumps({"summary": "looks fine", "findings": []})
        diff_path = Path("/tmp/ai-pr-review-partial.diff")
        diff_path.write_text(many_hunk_file("src/huge.ts", 400), encoding="utf-8")
        parts, _, _ = self.review.chunk_diff(diff_path.read_text(encoding="utf-8"))
        self.assertGreater(len(parts), 2, "fixture must produce enough parts to fail some")
        out_path = Path("/tmp/ai-pr-review-partial.out")

        calls = {"n": 0}

        class MostlyFailing:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=self)

            def create(self, **kwargs):
                calls["n"] += 1
                if calls["n"] > 1:
                    raise RuntimeError("429 too many requests")
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=reviewer_body))]
                )

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=True):
            with patch.object(self.review, "OpenAI", MostlyFailing):
                argv = sys.argv
                sys.argv = ["ai_pr_review.py", "--diff", str(diff_path), "--output", str(out_path)]
                try:
                    self.review.main()
                finally:
                    sys.argv = argv

        body = out_path.read_text(encoding="utf-8")
        self.assertIn("## AI PR Review", body, "the published body must be markdown")
        self.assertNotIn(
            reviewer_body,
            body,
            "a lone surviving part must not be published as if it were the whole review",
        )
        self.assertIn("could not be reviewed", body, "the unreviewed parts must be disclosed")


class EmptyCompletionTests(unittest.TestCase):
    """An empty reviewer completion is a failed review, not a clean one. This
    previously returned a benign payload, so the workflow reported a passing
    check while nothing had been reviewed."""

    @classmethod
    def setUpClass(cls):
        cls.review = load_review_module()

    def test_empty_completion_exits_non_zero_and_reports_failure(self):
        diff_path = Path("/tmp/ai-pr-review-empty.diff")
        diff_path.write_text(one_hunk_file("src/small.ts", 2), encoding="utf-8")
        out_path = Path("/tmp/ai-pr-review-empty.out")
        if out_path.exists():
            out_path.unlink()

        class EmptyOpenAI:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=self)

            def create(self, **kwargs):
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=""))]
                )

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=True):
            with patch.object(self.review, "OpenAI", EmptyOpenAI):
                argv = sys.argv
                sys.argv = ["ai_pr_review.py", "--diff", str(diff_path), "--output", str(out_path)]
                try:
                    with self.assertRaises(SystemExit) as raised:
                        self.review.main()
                    self.assertEqual(raised.exception.code, 1)
                finally:
                    sys.argv = argv

        body = out_path.read_text(encoding="utf-8")
        self.assertIn("failed before producing findings", body)
        self.assertNotIn("No review output", body, "the benign payload must be gone")


if __name__ == "__main__":
    unittest.main()
