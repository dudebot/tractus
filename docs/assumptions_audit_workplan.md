# Assumptions audit — Category 3 (workplan / scope)

> Part of the broader audit dispatched by `workplan/tasks/00_active/008_assumptions_audit.md`.
> Sister docs: `assumptions_audit_sparc.md` (Cat 1), `assumptions_audit_tea_loss.md`
> (Cat 2 + 4). This doc covers workplan budgets, sprint-pacing
> assumptions, and load-bearing roadmap claims.

## TL;DR — top 3 concerns

1. **"Kinematic supervision is structurally required"** is the most
   load-bearing claim in the entire workplan — it justifies all of
   Weeks 2-4 (SPARC integration, body-model training, controller
   rewrite). After Codex adversarial review, the verdict is
   **UNDETERMINED**, not proven. The Week-1 evidence that produced
   this claim was a non-converged optimizer with frication disabled.
   See §2 below for the full takedown.

2. **The 4-week sprint structure is self-imposed**, not externally
   calendared, but is already biting us. Week 1 closed with a
   "phonetic wall" verdict on a non-converged run because the
   100-iter budget was a script default, not a convergence criterion.
   Sprint pacing forced premature closure.

3. **Inference budgets (≤ 8 GB / ≤ 30 M params)** are stated without
   citation to a target hardware profile. "Consumer hardware" and
   "smaller than Kokoro 82 M" are positioning statements, not
   requirements. If these were product decisions made elsewhere, that
   should be referenced; if not, the budgets should be challenged
   before they constrain architecture.

---

## 1. Per-assumption entries

### 1.1 ≤ 8 GB inference VRAM

- **Source:** `ARCHITECTURE.md:36`, table cell. Cross-reference
  `ARCHITECTURE.md:141` annotates "Consumer hardware". No further
  citation. The course-correction doc at line 253 uses 8 GB as a
  load-bearing constraint to disqualify SPARC at inference time
  (WavLM Large 1.26 GB / 316 M params).
- **Verdict:** `inherited unverified`. The number is plausible (RTX
  3060/4060/5060 mid-range cards have 8 GB), but no target device or
  hardware survey is cited. If the project's actual deployment target
  is "anything ≥ 8 GB," that should be stated; if it's a specific
  device (Jetson? Steam Deck? developer's laptop?), that should be
  named.
- **Failure mode if wrong:** silent over-engineering. SPARC was ruled
  out at inference partly because of this number. If the real budget
  is 12 GB (typical 2025 mid-range), SPARC could potentially be
  in-stack. If the real budget is 4 GB (mobile / edge), even TEA's
  per-frame MNA solve may be too heavy and the architecture needs
  rethinking entirely.
- **Suggested verification:** name the target device (or top-3 target
  devices) explicitly in `ARCHITECTURE.md`. If the answer is "we don't
  know yet," delete the budget and replace with "TBD pending product
  decision."

### 1.2 ≤ 30 M trainable params

- **Source:** `ARCHITECTURE.md:37`, table cell. Cross-reference
  `ARCHITECTURE.md:142` annotates "Far smaller than Kokoro 82 M".
  Breakdown given: planner ≤ 10 M, controller ~5 M, body ~0.5 M,
  speaker enc ~5–15 M.
- **Verdict:** `cargo-cult` candidate. "Smaller than Kokoro" is not a
  requirement — it's a positioning statement. The breakdown numbers
  also don't trace to anything load-bearing (why is the planner
  capped at 10 M? what fails at 12 M?).
- **Failure mode if wrong:** if 30 M is a soft target masquerading as
  a hard cap, architecture decisions get over-constrained for no
  product reason. Conversely, if 30 M is genuinely required (mobile
  inference, on-device training, etc.), but the project ships
  something like a 50 M speaker encoder, the deployment target is
  silently broken.
- **Suggested verification:** same as 1.1 — name the constraint that
  produces 30 M, or convert it to a positioning statement ("compact;
  comparable in scale to Kokoro") and stop using it as a budget.

### 1.3 Sample rate 22.05 kHz target / 16 kHz floor

- **Source:** `ARCHITECTURE.md:40`. The copyfit script uses 22.05 kHz
  (`scripts/copyfit_neural_field_phrase.py:75`). The Week-2 SPARC
  task spec originally said 22.05 kHz and was wrong — SPARC needs
  16 kHz.
- **Verdict:** `inherited unverified`. 22.05 kHz is half-CD-rate, a
  long-standing audio convention; 16 kHz is the speech-recognition
  / SSL-feature-extractor convention. Both are real conventions but
  neither has a tractus-specific rationale.
- **Failure mode if wrong:** sample-rate mismatches between training
  data, supervision labels (SPARC at 16 kHz), and rendering (TEA at
  22.05 kHz internally) create silent label-noise. The recent
  Week-2 dispatch review (Codex thread `[session identifier omitted]`) flagged this
  exact concern.
- **Suggested verification:** decide a single project-wide sample
  rate (recommend 16 kHz to align with all SSL teachers and avoid
  resampling chains) and document in `ARCHITECTURE.md` why. If
  22.05 kHz is preserved for TEA-internal reasons, document those
  reasons explicitly.

### 1.4 30 s consented reference audio for speaker fit

- **Source:** `ARCHITECTURE.md:39`. Not elaborated.
- **Verdict:** `cargo-cult` until proven otherwise. 30 s is a common
  voice-cloning convention (XTTS, OpenVoice, etc.) but tractus's
  speaker fitting isn't a few-shot LLM voice clone — it's an
  optimization fit on a per-speaker anatomy + source object.
- **Failure mode if wrong:** if 30 s is too little to fit the
  speaker object, the demo will silently underperform on cloning
  quality. If 60 s would be substantially better, we've committed to
  a worse user experience for no reason.
- **Suggested verification:** once the body model is in place, run
  speaker-fit ablations on 10 s / 30 s / 60 s / 120 s of reference
  audio and measure fit quality. Until then, mark "30 s" as a
  placeholder rather than a requirement.

### 1.5 ≤ 5× realtime inference latency

- **Source:** `ARCHITECTURE.md:41`.
- **Verdict:** `inherited unverified`. Where did 5× come from? Real
  product decisions usually constrain this from the user-experience
  side (interactive ≤ 2x, batch ≤ 100x), not "5x" specifically.
- **Failure mode if wrong:** TEA's per-sample MNA solve makes 5x a
  challenging bar; if the real bar is 10x (acceptable for non-
  interactive use cases), some of the optimization complexity in TEA
  may be unnecessary.
- **Suggested verification:** name the user-facing use case. If
  "interactive demo response in browser," 5x is too slow already; if
  "offline render of scripts," 5x is fast. The number is meaningless
  without context.

### 1.6 4-week sprint structure

- **Source:** `workplan/handoff/2026-05-09_course_correction.md` —
  pivot rationale. Self-imposed pacing.
- **Verdict:** `inherited unverified` with a recent failure mode.
  Not externally calendared. The sprint already produced a premature
  Week-1 closure (the "phonetic wall" verdict was declared on a non-
  converged run, partly to keep the schedule).
- **Failure mode if wrong:** sprint-pacing biases toward "ship the
  verdict" over "verify the verdict." See §2 — the most load-bearing
  Week-1 conclusion is now under review.
- **Suggested verification:** make the sprint pacing soft. Add an
  explicit "before closing a week, verify the closure verdict has
  not been overclaimed from incomplete data" gate. Or: drop the
  sprint frame entirely and operate on "what's the next experiment"
  rather than "what's this week's deliverable."

### 1.7 Story 1996 area-function reference (current usage)

- **Source:** Originally used as primary supervision target in the
  pre-pivot architecture; demoted to held-out sanity gate after the
  2026-05-09 pivot.
- **Verdict:** `verified` for current usage (sanity gate only). The
  Story-anchor data is mathematically uncontroversial as a vowel
  formant reference; it's only problematic when used as the
  optimization objective (which the pivot correctly stopped).
- **Failure mode if wrong:** none in current usage — the gate is
  purely for sanity-checking F1/F2 against published vowel formants.
  If anyone proposes re-promoting Story to a primary loss term,
  re-audit; the prior failure pattern was real.
- **Suggested verification:** none needed currently.

---

## 2. The big one — "kinematic supervision is structurally required"

This claim, from `docs/week1/findings.md` Week-1 closure verdict,
justifies the entire downstream Weeks 2-4 plan: SPARC extraction
pipeline, speaker-conditioned body model, controller-with-SPARC,
constrained demo. If false, ~3 weeks of work is misdirected.

### Codex adversarial review — full transcript reference

- Thread ID: `[session identifier omitted]`
- Mode: adversarial verification, fresh thread, no prior context
- Verdict-forcing prompt: 6 questions ranking evidence
  STRONG/WEAK/VOID, naming the cheapest discriminating experiment,
  giving a final PROVEN/PROBABLE/UNDETERMINED/DISPROVEN verdict.

### Codex's verdict per evidence stream

| Evidence | Codex verdict |
|---|---|
| Real-audio juice_lemons_run100 | **VOID** for the structural claim — disabled frication + non-plateaued 100-iter stop makes the failed Whisper transcript non-diagnostic |
| Synth-/A/ probe | **WEAK** — establishes "this run did not recover articulation," not "same acoustics, wrong articulation" |
| Inferential chain "non-uniqueness → acoustic loss can't supervise" | **WEAK to overclaimed** — non-uniqueness is real but supports "may be ill-conditioned," not "structurally required" |

### Final Codex verdict

**UNDETERMINED.** Quote: *"The current evidence does not prove the
claim. It also does not disprove that kinematic supervision may
become useful or even necessary later. The correct statement is
weaker: 'Current acoustic-only inverse fitting is not yet demonstrated
to recover reliable articulation, and the inverse problem is likely
ill-conditioned.' That is materially different from 'kinematic
supervision is structurally required.' The latter is a roadmap-grade
claim and needs a converged, in-model, frication-capable negative
result before it can carry 3 weeks of downstream work."*

### Codex's recommended experiment

**TEA-to-TEA dynamic copyfit with known ground-truth articulation,
frication enabled, longer optimization, multi-seed, both vowel-heavy
and consonant-heavy targets.** Measure both:

- Acoustic recovery (log-mel, MR-STFT, formants, Whisper)
- Kinematic recovery (area-function error, constriction location +
  timing, frication-event timing, trajectory correlation across seeds)

3 seeds × 2 targets = 6 runs. CPU time only after JIT compile;
total maybe 1-2 days, not 3 weeks. If acoustic-only inversion can
recover the known TEA articulators in-model after convergence, the
"structurally required" claim is falsified or narrowed.

### Codex's recommended action

> "Run the cheap discriminating experiment from Q4 first, and pause
> any SPARC work that is only justified by the structural-requirement
> claim. Do not abandon SPARC as a possible useful teacher. But do
> not spend 3 weeks integrating it because of Week-1's verdict."

### My verdict

I agree with Codex. The Week-1 closure was overclaimed. The user's
own instinct ("a lot of the [SPARC] requirements seemed a little
concerning ... it's been a repeating process where we assume one
requirement and the cargo cult engineering fucks us later") is
correct.

The frication-enabled 200-iter rerun (in progress at time of writing,
output to `outputs/copyfit_neural_field/juice_lemons_frication200/`)
addresses confounders A and B. Codex's TEA-to-TEA experiment addresses
confounder C and gets us a properly-controlled in-model test.

---

## What would change if these assumptions are wrong

- **If 1.7 (Codex's verdict) is correct and acoustic-only inversion
  works in-model:** Weeks 2-4 SPARC integration is
  postpone-or-cancel. The body model can train on synthetic-target
  copyfit pairs (Codex's Q4 experiment is essentially a small
  corpus generator). SPARC moves from "structurally required
  teacher" to "potentially useful auxiliary signal" status, and the
  Week-2 dispatch we just GREEN-stamped becomes a YELLOW pending
  the discriminating experiment.

- **If 1.1 / 1.2 / 1.5 (budgets) turn out to be soft / undocumented:**
  the SPARC-at-inference vs SPARC-at-training distinction loses some
  of its weight. Could still be the right call, but the rationale
  would need to be "we want to train fast / iterate on architecture
  freely" rather than "we cannot fit SPARC in the inference budget."

- **If 1.6 (sprint structure) gets relaxed:** Week-1-style premature
  closures become less likely. Week 1 demonstrably suffered from
  this (loss not plateaued, frication off, gates failed but verdict
  declared anyway).

---

## Recommended follow-up tasks

1. **`009_tea_to_tea_discriminating_experiment.md`** — run Codex's
   recommended experiment. Highest priority. Blocks the SPARC
   roadmap until results are in. ~1-2 days of compute.

2. **`010_workplan_budget_decisions.md`** — surface the 8 GB / 30 M /
   30 s / 5x numbers to the developer for either "yes these are
   product constraints, here's why" or "delete from
   ARCHITECTURE.md." Cheap admin task.

3. **Pause Task 002** (`002_sparc_extraction_pipeline.md`) until
   Task 009 reports. The Week-2 SPARC dispatch we GREEN-stamped is
   technically correct but its motivation is now under review. Don't
   dispatch a worker yet.

---

## References

- `docs/week1/findings.md` — the closure verdict under review
- `workplan/handoff/2026-05-09_course_correction.md` — pivot
  rationale, where the "SPARC as kinematic teacher" plan was first
  committed to
- `ARCHITECTURE.md` §0.2 (table) — budget claims
- Codex thread `[session identifier omitted]` — adversarial
  review of the structural claim (continue if needed via codex-reply)
- `outputs/copyfit_neural_field/juice_lemons_run100/report.json` —
  the non-converged trace
- `outputs/copyfit_neural_field/juice_lemons_frication200/` — in-
  progress frication-enabled rerun (will land within ~2.5h of run
  start; check `run.log` for status)
