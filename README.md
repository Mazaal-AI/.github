# .github

Organisation-wide GitHub configuration for [Mazaal AI](https://github.com/Mazaal-AI).

## What is here

| Path | Purpose |
|---|---|
| `.github/workflows/ai-pr-review.yml` | The AI PR reviewer as a **reusable workflow**. One copy, every repository. |
| `scripts/ai_pr_review.py` | The reviewer itself: diff in, review comment out. |
| `workflow-templates/` | Starter workflow so any repository can adopt it from the Actions tab. |

## Adopting the reviewer in a repository

Either pick **AI PR Review** under *Actions → New workflow*, or add this file:

```yaml
name: AI PR Review
on:
  pull_request_target:
    types: [opened, synchronize, reopened]
permissions:
  contents: read
  pull-requests: write
jobs:
  review:
    uses: Mazaal-AI/.github/.github/workflows/ai-pr-review.yml@main
    secrets: inherit
```

## Secret

Each repository needs `DEEPSEEK_API_KEY` as a **repository** secret:

```
gh secret set DEEPSEEK_API_KEY --repo Mazaal-AI/<repository>
```

An organisation secret with visibility `all` is also configured, and the
REST API reports it as available to every repository — but it does not
reach a workflow on this plan. Without the repository secret, the calling
job fails at evaluation time, before a runner is ever assigned:

```
Error when evaluating 'secrets'. .github/workflows/ai-pr-review.yml
(Line: 22, Col: 11): Secret DEEPSEEK_API_KEY is required, but not
provided while calling.
```

Measured 2026-09-21: `secrets: inherit` and an explicit
`secrets: DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}` both fail the
same way with only the organisation secret present, and both succeed once
the repository secret exists. Organisation secrets for private
repositories need a paid plan; if the organisation moves to Team, the
organisation secret starts working and these per-repository ones can go.

## Runners

The default runner label is `ai-review`, served by self-hosted runners
registered at the organisation level so that every repository — including ones
created later — is covered without registering anything per repository:

```
runs-on: [self-hosted, linux, x64, ai-review]
```

Public repositories should pass `runner-labels: '["ubuntu-latest"]'`. The jobs
never check out or execute pull-request code, so a fork diff cannot reach a
self-hosted machine, but public traffic does not belong on a private box.

## Safety model

- The pull request is **never checked out**. The diff is fetched through the API
  and consumed as data by the model.
- Review runs only for same-repository branches; fork PRs get an explicit
  "no automated review" comment instead of a silent skip.
- The reviewer script is checked out from this repository at a pinned ref, so a
  pull request cannot alter the reviewer that judges it.
