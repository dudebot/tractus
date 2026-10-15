# Task 006 — Week 1 housekeeping sweep

> **Blocker:** none. Pre-Week-2 hygiene. One worker sweeps everything
> in this list and reports a single LOG.md entry at the end. No
> inter-item dependencies.

## Goal

Clear the deferred-but-recordable items from `docs/week1/findings.md`
§"Open issues" and refresh the workplan paperwork so a Week-2 worker
inheriting the repo doesn't trip on stale state.

## Items

Each item is small. Bundle them into one PR / one commit per logical
group; don't sprawl.

### A. Refresh `workplan/NOW.md`

Currently NOW.md still says:
- Task 001 status = `dispatchable` (it's done — Week 1 closed)
- Task 002 status = `blocked-on-001` (now unblocked / actively next)
- Immediate-actions §1 says "kill the R1-distill worker" (presumed done)
- Immediate-actions §2 says "record the Week-1 reference audio" (done; lives at `data/probes/juice_lemons_ref.wav`)

Update the status table, drop the completed immediate-actions, and
add a one-line pointer to `docs/week1/findings.md` for the closure
narrative. Keep NOW.md ≤ 200 lines.

### B. Verify `workplan/LOG.md` has the Week-1 closure entry

The closure narrative lives in `docs/week1/findings.md`. Confirm
LOG.md has a corresponding short entry (date, "Week 1 closure",
verdict, link to findings). If missing, append using the existing
LOG.md format.

### C. Fix the gate2 spec/script mismatch in 001

`workplan/tasks/00_active/001_neural_field_phrase_probe.md:191`
specifies "MR-STFT-only" for Gate 2; the Week-1 implementation
gates on composite acoustic loss (log-mel + MR-STFT). Either is
defensible; the spec should match the implementation. Edit the
task file to say "composite acoustic loss (log-mel + MR-STFT)".
Add a one-line note that the spec was amended post-implementation
to match what was actually shipped.

### D. Fix `train_step` pre-update-loss comment

`scripts/copyfit_neural_field_phrase.py` ~line 818: comment claims
the returned loss is post-update. The function actually returns the
*pre-update* loss; line ~818 runs one extra Adam step to true up
the final report. Update the comment to match. Cosmetic; no
functional change.

### E. Fix copyfit area-function plot shape mismatch

The area-function plot in copyfit tries to overlay shapes (40,) and
(93,) and silently fails. Plot is just not produced; nothing else
breaks. Trace the source — likely a slice index referring to the
old `FREE_LO:FREE_HI` (40-section) range vs the full 93-section
display array — and fix so the plot renders. If the fix is
non-obvious in 15 min, drop the plot call entirely with a comment
and skip to the next item.

### F. `.gitattributes` for runbook CRLF

`runbooks/*.sh` and `runbooks/*.py` are checked in with CRLF line
endings due to `core.autocrlf=true`, breaking bash invocation on
Linux. Add a repo-root `.gitattributes` with:

```
* text=auto eol=lf
*.bat text eol=crlf
*.cmd text eol=crlf
```

Then `git add --renormalize .` to fix the in-tree files. Verify
`bash runbooks/whisper_gate.sh --help` works after.

### G. `tractus_engine_advanced` namespace package re-exports

Top-level package re-exports don't resolve when the package is
installed in namespace mode. Sub-module imports work fine
(`from tractus_engine_advanced.tds.mna import ...`). Cosmetic for
now — fix only if the worker can do it in <30 min by adding
explicit re-exports to `__init__.py`. Otherwise document in
findings.md "Open issues" with a workaround.

## Anti-deliverables

- No new features.
- No refactoring beyond what's needed for the items above.
- Don't fix items not in this list, even if you spot something —
  surface to the developer instead.
- Don't delete `legacy/` artifacts even if they look unused. The
  course-correction doc says they're preserved for archaeology.

## Acceptance gates

### Gate 1 — NOW.md is current
A fresh reader can read NOW.md alone and know what to dispatch next.
Specifically: the status table reflects Week-1 done + Week-2 next,
no completed immediate-actions remain.

### Gate 2 — LOG.md has Week-1 closure
One entry exists referencing findings.md's verdict.

### Gate 3 — runbooks/*.sh runs from bash on Linux
`bash runbooks/whisper_gate.sh --help` exits 0 (or with the help
text, whichever the script returns). No CRLF / `\r` errors in
output.

### Gate 4 — copyfit script comment + plot
Either: plot renders correctly, OR the plot call is dropped with a
comment explaining why. `train_step` comment matches behavior.

## Reporting

One LOG.md entry at completion:

```
## YYYY-MM-DD — Week 1 housekeeping sweep
- NOW.md refresh:      done
- LOG.md verification: done | already-present
- 001 spec amend:      done
- copyfit fixes:       done | partial (note which deferred)
- .gitattributes:      done
- namespace re-export: done | deferred (workaround in findings.md)
- Verdict:             GREEN | PARTIAL
```

## References

- `docs/week1/findings.md` §"Open issues" — full original list
- `workplan/NOW.md` — file to refresh
- `scripts/copyfit_neural_field_phrase.py` — items D, E
