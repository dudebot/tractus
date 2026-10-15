# LOG — daily progress notes

> Short entries. Date + task + verdict + artifact pointer. Keep tight.
> Long-form rationale goes in `handoff/` documents.

## 2026-05-09 — course correction landed

- Audited project viability + pace; identified parameterization +
  workplan ceremony as the two big drift sources.
- Multi-turn Codex deliberation (thread `[session identifier omitted]`) settled
  the new direction: continuous neural-field `A_θ(s, t)` + SPARC
  training-time supervision + acoustic loss through TEA. Story
  anchors demoted to held-out sanity gate.
- Greenfield rewrite of `ARCHITECTURE.md` (~600 lines) around the
  new five-module shape.
- Workplan flattened: 33 prepivot task files + ceremony moved to
  `workplan/archive/prepivot/` via `git mv` (history preserved).
  Five new active task files seeded under `workplan/tasks/00_active/`.
- Conv-MLP body model + training scripts moved to `legacy/`.
  `tractus_controller.body_model` package still imports clean
  (20 channels).
- See `workplan/handoff/2026-05-09_course_correction.md` for the
  full narrative.
- **Next:** developer kills `pretraining/002b` R1-distill worker
  if alive, records Week-1 reference audio, dispatches Task 001.

## 2026-05-13 — task 001 neural-field phrase probe (Week 1)

- Verdict: **PARTIAL** — pipeline correct end-to-end, throughput
  at the upper edge of the task's 3-10 hour wall-clock budget on
  the 16 GB GPU / a virtualized GPU environment box. Gate run not completed by the
  worker. Handoff doc has the math + recommendation.
- Gates measured: Gate 2 (loss improvement): init acoustic =
  4.06, iter-1 acoustic = 4.06 (essentially equal — last-layer
  near-zero init means iter 1 barely moves the field, as
  expected). Final not reached. Gates 1/3/4/5 not run.
- Artifact: `scripts/copyfit_neural_field_phrase.py` (worker
  branch `worktree-agent-af90ebf604d72cf51`). Tests:
  per-iter wall ~30-60s @ K=10 frame_chunk_size, ~3-4 min compile.
  Outputs gitignored.
- Notes: (1) Reference audio is ElevenLabs "Natasha" synthesis,
  developer-confirmed — Week-1 is a TTS-target probe, not a
  developer-voice clone. (2) Codex review (thread
  `[session identifier omitted]`) validated K-bump strategy + cheap-canary
  seed strategy. (3) Per-sample MNA backward at T=120 is the
  throughput bottleneck; the production 250-iter × 3-seed run
  is **at the limit** of feasibility (~11 hours), 1-seed run is
  ~3.5 hours.
- Next: developer decides between (a) running the 1-seed × 250
  CLI invocation as-is (3-4 hours wall-clock), then
  `runbooks/whisper_gate.sh` on the output for Gate 1, or
  (b) jumping to Risk-C salvage (Chen-style FD renderer). See
  `workplan/handoff/2026-05-13_week1_throughput.md`.

## 2026-05-13 — sustained-vowel hand-tune probe (Week-0 sanity check)

- Verdict: **GREEN-ish.** TEA produces 11 distinguishable vowel
  timbres from Story 1996 area functions with constant voiced
  source (F0=120, PR=8000). Robotic but recognizable — expected
  given no prosody / breath / pitch variation in the source schedule.
- Developer-side observation: `U.wav` vs `u.wav` "might be
  flipped"; `E.wav` "sounds a bit like uuh" but `e.wav` is fine.
  Likely a Story-key labeling artifact (uppercase = lax / secondary
  variant, lowercase = primary tense), not a TEA bug. Worth a
  5-min audit of `tractus_controller/.../body_model/anchors.py`
  + the Story-1996 paper's vowel taxonomy if the inversion
  bothers the developer.
- Velum A/B and sinus-ablation A/B were rendered but not
  separately reported on; audio at
  `outputs/hand_tune_vowels/nasal_ab/` and `sinus_ablation/`.
- Artifact: `scripts/sustained_vowel_probe.py` + 17 WAVs + 17
  spectrograms under `outputs/hand_tune_vowels/` (audio gitignored).
- Implication: TEA's instrument is **fit for purpose** on the
  oral cavity. Neural-field + SPARC plan unblocked on the
  simulator side. Next gate: Task 001 Gate-1 — run the af90
  worker's `scripts/copyfit_neural_field_phrase.py` for 1 seed
  × 250 iters (~3-4 hours wall-clock).

## 2026-05-13 — task 001 merged from worker af90ebf6

- Merge commit: `de9bf8a` (`--no-ff` to preserve worker history).
- Brings in `scripts/copyfit_neural_field_phrase.py` (1038 lines,
  syntax + `--help` verified), worker's LOG entry above this one,
  and `workplan/handoff/2026-05-13_week1_throughput.md`.
- Worktree + branch (`worktree-agent-af90ebf604d72cf51`) deleted
  after merge. Master is now the single source of Task-001 work.

<!--
Template for future entries:

## YYYY-MM-DD — task NNN <title>

- Verdict: GREEN | RED | PARTIAL
- Gates: <which passed / which failed>
- Artifact: outputs/<path>
- Notes: <1–3 sentences, surprising findings only>
- Next: <next dispatchable task>
-->
