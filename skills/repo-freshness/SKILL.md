---
name: repo-freshness
description: Catch documentation that has drifted from the code and data it describes. Use when auditing a repo before handoff or submission, after an experiment writes new results, or when the user asks whether docs are still accurate. Also use when a README references result files, when the same number appears in both prose and data, or when the user suspects a doc is outdated. Produces concrete drift findings, each tied to the file that proves it wrong.
license: MIT
allowed-tools: Read, Write, Edit, Bash, Grep, Glob
metadata:
  category: maintenance
  version: 1.0
---

# Repo freshness

Documentation drift is the failure mode that **nothing else catches**. It
produces no compile error, no failing test, no stack trace. The PDF renders,
the scripts run, the tests pass — only the content is wrong. So it survives
every automated gate a repo normally has, and is found by a reader instead.

This skill finds it before the reader does.

## The eight drift types

Ranked by how often they bite, from experience on a real research repo:

| # | Type | Example seen in the wild |
|---|---|---|
| 1 | Status claim vs reality | README says "not yet run" while `results/exp1.json` exists |
| 2 | Number in prose vs number in data | Doc says 5.5, data says 4.3 |
| 3 | Same quantity, two documents disagree | 17.5 in one file, 18.0 in another |
| 4 | **Right number, wrong quantity** | "81% of headroom" when it is 81% *of the ceiling* |
| 5 | Config claim vs config block | Doc says 7B reader, result file says 0.5B |
| 6 | File inventory drift | README lists deleted files, omits new ones |
| 7 | Dependency drift | `requirements.txt` misses a package the code imports |
| 8 | Cross-source config mismatch | Script joins two datasets computed under different settings |

Type 4 is the nastiest: an automated number-check **passes**, because the
number is right. Only reading the sentence catches it. Budget attention there.

## How to run an audit

1. **Automated pass first.** If the repo has `scripts/check_repo.py` and
   `scripts/verify_report_numbers.py`, run both. If not, generate them from
   [templates](scripts/) — they are repo-agnostic apart from a config block.

2. **Then the manual pass**, which is where types 3 and 4 live:
   - Grep each headline number across *all* docs; if it appears in two places,
     confirm both say the same thing.
   - For every ratio or percentage, ask **what is the denominator**, and check
     the prose names it correctly.
   - Open the `config` block of each result file and compare to how the doc
     describes that run.

3. **Report findings with the evidence**, not just the claim: name the file
   and line that proves the doc wrong.

## The rule that prevents most of this

**Every number in a document must be derivable from a file in the repo.** If a
number cannot be traced to `results/`, it is either wrong or unverifiable —
both are problems. Prefer generating tables from data over typing them.

## Verifying a fix actually applied

A string-replacement edit that does not match **silently does nothing**. This
has happened in practice: a fix was reported as done, and the checker caught
the same issue on the next run. After any scripted edit, re-run the checker
rather than trusting the edit reported success.

## Files

- [scripts/check_repo.py](scripts/check_repo.py) — drift types 1, 5, 6, 7.
  Edit only the config block at the top; the rest is repo-agnostic.
- [scripts/verify_numbers.py](scripts/verify_numbers.py) — drift type 2.
  A template: fill in one `chk()` per number the docs quote.
- [scripts/validate_skill.py](scripts/validate_skill.py) — checks a skill
  against the Agent Skills spec. Copied from `anthropics/skills`; use
  `--allow-extra parent` for vendor bundles that add their own fields.
- [reference/manual-audit.md](reference/manual-audit.md) — types 3, 4, 8,
  the ones a script cannot catch.
- [reference/adopting.md](reference/adopting.md) — wiring into a new repo,
  including a pre-commit hook.

## Keeping the skill itself current

This skill claims things about the Agent Skills spec, and that spec moves.
`validate_skill.py` is a verbatim copy of the official validator, so re-copying
it from `anthropics/skills` is how you re-sync. If the copy starts rejecting
skills that the official one accepts, the copy is stale — replace it rather
than editing the rules by hand.
