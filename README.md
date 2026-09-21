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

The repository needs `DEEPSEEK_API_KEY` available — an organisation secret
covers every repository at once, which is the point of this setup.

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

<!-- smoke test for the shared reviewer -->
2026-09-21T11:26:55Z
2026-09-21T11:31:58Z orghost
