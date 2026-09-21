# .github/scripts/ai_pr_review.py

import argparse
import json
import os
import re
import sys
from pathlib import Path

from openai import OpenAI


DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_REVIEWER_NAME = "DeepSeek V4 Flash"
MAX_DIFF_CHARS = 120_000
MAX_INLINE_COMMENTS = 10
# Each part is a separate API call, so an unbounded count multiplies latency,
# cost and 429 exposure inside one job, and can exceed the workflow timeout.
MAX_REVIEW_PARTS = 12


IGNORED_FILE_PATTERNS = [
    r"package-lock\.json$",
    r"pnpm-lock\.yaml$",
    r"yarn\.lock$",
    r"\.min\.js$",
    r"\.map$",
    r"\.snap$",
    r"dist/",
    r"build/",
    r"coverage/",
    r"vendor/",
    r"generated/",
]


SYSTEM_PROMPT = """You are a strict senior software engineer reviewing a GitHub pull request.

Review only the supplied diff. Do not speculate about code you cannot see. Do not invent missing context.

Report only issues that are likely to be real, actionable, and worth a developer's time.

Focus on:
- correctness bugs
- broken API contracts
- security regressions
- authentication or authorization mistakes
- data loss risks
- migration risks
- concurrency or race issues
- missing validation
- error handling bugs
- material test gaps

Ignore:
- style nits
- formatting
- naming preferences
- generic advice
- subjective refactors
- comments about unchanged code

Return JSON only. Do not wrap it in markdown fences.

Schema:
{
  "summary": "one short sentence",
  "findings": [
    {
      "path": "repository-relative file path from the diff",
      "line": 123,
      "severity": "blocking|high|medium",
      "body": "issue, why it matters, and suggested fix"
    }
  ]
}

If there are no blocking, high-risk, or medium-risk issues, return an empty findings array.
"""


def read_file(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        print(f"Diff file not found: {path}", file=sys.stderr)
        return


def should_ignore_file(filename: str) -> bool:
    return any(re.search(pattern, filename) for pattern in IGNORED_FILE_PATTERNS)


def filter_diff(diff: str) -> str:
    sections = re.split(r"(?=^diff --git a/)", diff, flags=re.MULTILINE)
    kept = []

    for section in sections:
        if not section.strip():
            continue

        first_line = section.splitlines()[0] if section.splitlines() else ""
        match = re.match(r"diff --git a/(.*?) b/(.*)", first_line)

        if not match:
            kept.append(section)
            continue

        old_file, new_file = match.groups()
        filename = new_file or old_file

        if should_ignore_file(filename):
            continue

        kept.append(section)

    return "\n".join(kept)


def reviewable_lines_by_file(diff: str) -> dict[str, set[int]]:
    reviewable: dict[str, set[int]] = {}
    current_file = None
    new_line = None

    for line in diff.splitlines():
        if line.startswith("diff --git "):
            current_file = None
            new_line = None
            match = re.match(r"diff --git a/(.*?) b/(.*)", line)
            if match:
                _old_file, new_file = match.groups()
                current_file = new_file
                reviewable.setdefault(current_file, set())
            continue

        if line.startswith("@@ "):
            match = re.search(r"\+(\d+)(?:,\d+)?", line)
            new_line = int(match.group(1)) if match else None
            continue

        if current_file is None or new_line is None:
            continue

        if line.startswith("+") and not line.startswith("+++"):
            reviewable[current_file].add(new_line)
            new_line += 1
        elif line.startswith(" "):
            new_line += 1
        elif line.startswith("-"):
            continue

    return reviewable


def _diff_units(diff: str) -> list[str]:
    """Split a filtered diff into units that each stand alone.

    Units are whole files where they fit, and single hunks otherwise. The point
    is to never cut mid-hunk: a character cut leaves the reviewer reading a
    fragment of a function and inferring the rest, which is exactly how a
    truncated review produced a confident finding about a query it could not
    see. Each hunk keeps its file header so it is self-describing.
    """
    units: list[str] = []

    # Git quotes paths containing special characters, emitting
    # `diff --git "a/some file.ts" "b/some file.ts"`. Matching only the
    # unquoted form would treat such a file as a continuation of the previous
    # section rather than a boundary of its own.
    for section in re.split(r'(?=^diff --git (?:a/|"a/))', diff, flags=re.MULTILINE):
        if not section.strip():
            continue
        if len(section) <= MAX_DIFF_CHARS:
            units.append(section)
            continue

        parts = re.split(r"(?=^@@ )", section, flags=re.MULTILINE)
        header, hunks = parts[0], parts[1:]
        if not hunks:
            units.append(section)
            continue
        units.extend(header + hunk for hunk in hunks)

    return units


def chunk_diff(diff: str) -> tuple[list[tuple[str, bool]], bool, int]:
    """Pack diff units into chunks that each fit the review budget.

    Returns `(parts, capped, total_part_count)` where each part is
    `(text, was_cut)`. The cut flag is per part, so a complete part is not
    described to the model as truncated merely because a different part had to
    be. `capped` reports that more parts existed than MAX_REVIEW_PARTS allows,
    and `total_part_count` preserves the pre-cap count for the public summary.
    """
    parts: list[tuple[str, bool]] = []
    current = ""
    current_cut = False

    def flush() -> None:
        nonlocal current, current_cut
        if current:
            parts.append((current, current_cut))
        current = ""
        current_cut = False

    for unit in _diff_units(diff):
        cut = False
        if len(unit) > MAX_DIFF_CHARS:
            # Last resort: a unit larger than the whole budget. Cut at the last
            # line boundary inside the budget so the model never reads a
            # half-written line and infers code from it.
            #
            # This always lands on a boundary and never mid-line: every unit
            # from _diff_units begins with a diff file header (and, for a split
            # file, a hunk header), so a newline is guaranteed well inside the
            # first MAX_DIFF_CHARS characters. A single line longer than the
            # entire budget is therefore cut off before it starts rather than
            # part-way through.
            unit = unit[:MAX_DIFF_CHARS].rsplit("\n", 1)[0] + "\n"
            cut = True
        if current and len(current) + len(unit) > MAX_DIFF_CHARS:
            flush()
        current += unit
        current_cut = current_cut or cut

    flush()

    total_part_count = len(parts)
    if total_part_count > MAX_REVIEW_PARTS:
        # Do NOT mark the last kept part as cut. Its text is complete and it is
        # reviewed in full; the cut flag means "the model cannot see all of this
        # part", which would be false and would push it to hedge or drop valid
        # findings near the end. That coverage was incomplete is carried by
        # `capped`, which the caller states in the summary.
        return parts[:MAX_REVIEW_PARTS], True, total_part_count

    return parts, False, total_part_count


def build_user_prompt(
    diff: str,
    was_truncated: bool,
    part: int | None = None,
    parts: int | None = None,
) -> str:
    note = ""
    if was_truncated:
        note += (
            "Note: this portion of the diff was truncated because it was too large. "
            "Mention that the review is partial.\n"
        )
    if parts and parts > 1:
        note += (
            f"Note: this is part {part} of {parts} of a large pull request, split at file "
            "and hunk boundaries. Other parts are reviewed separately. Do not report a "
            "missing declaration, selection, import, guard, or field as a finding when it "
            "could plausibly be declared outside this part. If a concern depends on code "
            "you cannot see here, omit it rather than asserting it.\n"
        )
    if note:
        note += "\n"

    return f"""{note}Review this pull request diff:

```diff
{diff}
```"""


def call_deepseek(prompt: str) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY")

    if not api_key:
        print("Missing DEEPSEEK_API_KEY", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
    )

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )

    content = response.choices[0].message.content

    # An empty completion is a FAILED review, not a clean one. Returning a
    # benign payload here (which this function used to do) made main() exit 0
    # and the workflow report a passing check while nothing had been reviewed —
    # the same silent-pass failure that the surrounding workflow was changed to
    # remove. Raising routes it into main()'s handler, which reports the failure
    # and exits non-zero, so the check goes red instead of falsely green.
    #
    # `content.strip()` matters as well as the falsiness check: a whitespace-only
    # completion also used to reach parse_review_json as "non-JSON output" and be
    # reported as a review with no findings.
    if not content or not content.strip():
        raise RuntimeError("model returned an empty completion")

    return content.strip()


def get_reviewer():
    return DEEPSEEK_REVIEWER_NAME, call_deepseek


def parse_review_json(raw_review: str, reviewer_name: str = DEEPSEEK_REVIEWER_NAME) -> dict:
    if not isinstance(raw_review, str):
        return {
            "summary": f"{reviewer_name} returned an invalid review payload.",
            "findings": [],
        }

    stripped = raw_review.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if fence_match:
        stripped = fence_match.group(1)

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return parse_markdown_review(raw_review, reviewer_name)

    if not isinstance(parsed, dict):
        return {"summary": f"{reviewer_name} returned an invalid review payload.", "findings": []}

    summary = parsed.get("summary")
    findings = parsed.get("findings")
    return {
        "summary": summary if isinstance(summary, str) and summary.strip() else f"{reviewer_name} review completed.",
        "findings": findings if isinstance(findings, list) else [],
    }


def parse_markdown_review(raw_review: str, reviewer_name: str = DEEPSEEK_REVIEWER_NAME) -> dict:
    findings = []
    severity = "medium"

    for line in raw_review.splitlines():
        normalized = line.strip()
        heading = normalized.lower()
        if heading.startswith("###"):
            if "blocking" in heading or "high" in heading:
                severity = "high"
            elif "medium" in heading:
                severity = "medium"
            continue

        match = re.match(r"^- `([^`:]+(?:/[^`:]+)*):(\d+)(?:[-:]\d+)?`\s+[—-]\s+(.+)$", normalized)
        if not match:
            continue

        path_value, line_value, body_value = match.groups()
        findings.append(
            {
                "path": path_value,
                "line": int(line_value),
                "severity": severity,
                "body": body_value.strip(),
            }
        )

    if findings:
        return {
            "summary": f"{reviewer_name} returned markdown; converted actionable findings to inline comments.",
            "findings": findings,
        }

    return {
        "summary": f"{reviewer_name} returned non-JSON review output.",
        "findings": [],
    }


def merge_parts(
    parsed_parts: list[dict],
    part_count: int,
    capped: bool,
    part_failures: list[str],
    truncated: bool = False,
    reviewer_name: str = DEEPSEEK_REVIEWER_NAME,
    total_part_count: int | None = None,
) -> dict:
    """Combine per-part reviews into the single review that gets published.

    Extracted from main() so it can be asserted on directly. It is the
    load-bearing part of issuing one request per part, and while it lived
    inline the only way to reach it was to run main() end to end.
    """
    summaries = [
        part["summary"].strip()
        for part in parsed_parts
        if isinstance(part.get("summary"), str) and part["summary"].strip()
    ]
    summary = " ".join(dict.fromkeys(summaries)) or f"{reviewer_name} review completed."

    notes = []
    if capped:
        if total_part_count is not None and total_part_count > part_count:
            notes.append(f"Split into {total_part_count} parts because the diff did not fit one request.")
            notes.append(
                f"Only the first {part_count} of {total_part_count} parts were reviewed; the remainder was not."
            )
        else:
            notes.append(f"Only the first {part_count} parts were reviewed; the remainder was not.")
    elif part_count > 1:
        # "Split into", not "Reviewed in": some parts may have failed, and the
        # failure note below counts against this same number. Wording this as
        # reviewed would let the summary say "Reviewed in 12 parts" and
        # "5 part(s) could not be reviewed" in the same sentence.
        notes.append(f"Split into {part_count} parts because the diff did not fit one request.")
    if part_failures:
        # Count only. This summary is published as a PR comment, and exception
        # text from the HTTP client can carry URLs or response bodies. The
        # detail is logged instead.
        notes.append(
            f"{len(part_failures)} of {part_count} part(s) could not be reviewed; see the workflow log."
        )
    if truncated:
        # Otherwise a cut part's partiality would rest only on the model
        # remembering to mention it, which the prompt asks for but cannot
        # guarantee.
        notes.append("Part of the diff was truncated because a single hunk exceeded the size budget.")
    if notes:
        summary = " ".join(notes) + " " + summary

    return {
        "summary": summary,
        "findings": [finding for part in parsed_parts for finding in part.get("findings", [])],
    }


def render_markdown(parsed: dict) -> str:
    """Render a merged review as the markdown body posted to the pull request.

    Markdown rather than the raw reviewer JSON: the body is read by a human on
    the PR, and the failure path already writes markdown, so both outcomes of
    the same branch now share one format.
    """
    lines = ["## AI PR Review", "", parsed.get("summary", "").strip(), ""]
    findings = [f for f in (parsed.get("findings") or []) if isinstance(f, dict)]
    if findings:
        lines.append(f"### Findings ({len(findings)})")
        lines.append("")
        for finding in findings:
            path = finding.get("path", "?")
            line = finding.get("line", "?")
            severity = str(finding.get("severity", "medium")).strip().lower()
            body = str(finding.get("body", "")).strip().replace("\n", " ")
            lines.append(f"- **`{path}:{line}`** ({severity}) — {body}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_github_review(
    parsed: dict,
    reviewable_lines: dict[str, set[int]],
    was_truncated: bool,
    commit_id: str,
    reviewer_name: str = DEEPSEEK_REVIEWER_NAME,
) -> dict:
    comments = []
    skipped = 0

    for finding in parsed["findings"]:
        if len(comments) >= MAX_INLINE_COMMENTS:
            skipped += 1
            continue
        if not isinstance(finding, dict):
            skipped += 1
            continue

        path_value = finding.get("path")
        line_value = finding.get("line")
        body_value = finding.get("body")
        severity_value = finding.get("severity", "medium")

        if not isinstance(path_value, str) or not isinstance(line_value, int) or not isinstance(body_value, str):
            skipped += 1
            continue
        if line_value not in reviewable_lines.get(path_value, set()):
            skipped += 1
            continue

        severity = str(severity_value).strip().lower()
        comments.append(
            {
                "path": path_value,
                "line": line_value,
                "side": "RIGHT",
                "body": f"{reviewer_name} {severity}: {body_value.strip()}",
            }
        )

    body_parts = [parsed["summary"].strip()]
    if was_truncated:
        body_parts.append("Review was partial because the diff was truncated.")
    if skipped:
        body_parts.append(f"Skipped {skipped} finding(s) that could not be placed on changed lines or exceeded the cap.")
    if comments:
        body_parts.append(f"Posted {len(comments)} inline finding(s).")

    return {
        "commit_id": commit_id,
        "event": "COMMENT",
        "body": " ".join(body_parts),
        "comments": comments,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", help="Emit a GitHub review payload when supplied by an inline workflow.")
    parser.add_argument("--diff", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        reviewer_name, call_reviewer = get_reviewer()
    except ValueError as error:
        parser.error(str(error))

    raw_diff = read_file(args.diff)
    filtered_diff = filter_diff(raw_diff)

    if not filtered_diff.strip():
        if args.commit:
            review = {
                "commit_id": args.commit,
                "event": "COMMENT",
                "body": f"{reviewer_name} review skipped: only ignored/generated files changed.",
                "comments": [],
            }
            Path(args.output).write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")
        else:
            Path(args.output).write_text(
                f"## AI PR Review\n\n{reviewer_name} review skipped: only ignored/generated files changed.\n",
                encoding="utf-8",
            )
        return

    parts, capped, total_part_count = chunk_diff(filtered_diff)
    reviewable_lines = reviewable_lines_by_file("".join(text for text, _ in parts))
    parsed_parts = []
    part_failures = []

    for index, (chunk, chunk_cut) in enumerate(parts, start=1):
        prompt = build_user_prompt(
            chunk,
            chunk_cut,
            part=index if len(parts) > 1 else None,
            parts=len(parts) if len(parts) > 1 else None,
        )
        try:
            parsed_parts.append(parse_review_json(call_reviewer(prompt), reviewer_name))
        except Exception as error:
            # Keep the parts that succeeded. With one call per part, discarding
            # everything on a single transient error grows with the part count.
            part_failures.append(f"part {index}/{len(parts)}: {error}")

    if part_failures:
        # Logged here, not only in the all-parts-failed branch: the published
        # summary sends the reader to the workflow log, so the log has to
        # actually contain the cause.
        for failure in part_failures:
            print(f"{reviewer_name} {failure}", file=sys.stderr)

    if not parsed_parts:
        # Detail goes to the log, not the artifact. The artifact is posted as a
        # public PR comment, and a raw HTTP-client exception can carry URLs,
        # response bodies, or request metadata that should not be published.
        detail = "; ".join(part_failures) or "the reviewer produced no output"
        print(f"{reviewer_name} review failed before producing findings: {detail}", file=sys.stderr)
        body = f"{reviewer_name} review failed before producing findings. See the workflow log for this run."
        if args.commit:
            review = {
                "commit_id": args.commit,
                "event": "COMMENT",
                "body": body,
                "comments": [],
            }
            Path(args.output).write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")
        else:
            Path(args.output).write_text(f"## AI PR Review\n\n{body}\n", encoding="utf-8")
        sys.exit(1)

    # `capped` has its own note; folding it in here would make the truncation
    # note assert a cause that did not happen, since a capped run has no cut
    # part. Only a genuine cut passes as truncated.
    parsed = merge_parts(
        parsed_parts,
        len(parts),
        capped,
        part_failures,
        any(cut for _, cut in parts),
        reviewer_name,
        total_part_count=total_part_count,
    )

    if not args.commit:
        # Rendered rather than passed through, so a review that was cut or
        # partially failed cannot publish as if it were complete: whatever
        # merge_parts recorded in the summary is what the reader sees.
        Path(args.output).write_text(
            render_markdown(parsed) + "\n---\nReviewed by DeepSeek V4 Flash via GitHub Actions.\n",
            encoding="utf-8",
        )
        return

    review = build_github_review(
        parsed,
        reviewable_lines,
        any(cut for _, cut in parts) or capped,
        args.commit,
        reviewer_name,
    )

    Path(args.output).write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
