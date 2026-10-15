# Session handoff — 2026-05-09 (purification + Story-anchor calibration)

Long working session 2026-05-08 → 2026-05-09. Pivot points and final
state below; intended for the next Claude session that picks up.

## What landed

### Purification branch (still local, not pushed)

Branch `purification` off master. ~12 commits. Headline numbers:
**29 deleted files, ~3000 lines net removed** across two waves of
purification work, plus ~1500 lines added for the body-model session
1+2 + Story anchor ingest + transfer-function calibration probes.

**Wave 1 — VTL/JD3 purge.** All references to "VocalTractLab", "JD3",
"birkholz", "SpeakerJD3" removed from working tree. The lint script
`scripts/ci/lint_no_vtl_leftovers.py` deleted (its job is done; the
brand should not return). Default-speaker fixture renamed
`jd3_default_tube.npz → default_tube.npz`. Historical `plan/*` files
deleted; the working tree should not retain the brand.

**Wave 2 — slop audit.** Stub lints for unbuilt modules
(`lint_channel_smallness.py`, `lint_residual_energy_cap.py`)
deleted. Placeholder skip-only tests removed. Dead runbooks
(`anticheat_unreachable.py`, `dimensionality_report.py`,
`plausibility_panel.py`, `_matched_reference_impl.py`,
`matched_reference_eval.sh`, 4 schemas, all test_fixtures) removed —
they referenced future tasks not in scope. Survives:
`runbooks/{whisper_gate.sh, source_tract_sep.py, _whisper_gate_impl.py,
_lib/}` because the m7 chain still references them.

**Cosmetic structural cleanup.** TEA's over-decomposed
`glottis/` and `output/` subpackages collapsed to flat
`glottis_geometric.py` and `output_iir.py`. `workplan/runbooks/`
moved to top-level `runbooks/`.

**Verified live throughout:** TEA fricative tests 17/17, TC tests
76/5sk, tests/ci 10/2sk, smoke CLI produces audio at unchanged RMS,
inverse-fit demo bit-identical to pre-purification baseline (md5).

### Pretraining/002 re-scope and anchor calibration

The original `pretraining/002` ("body model on 1M synthetic gesture-
sweep corpus") was re-scoped after Codex review revealed the
implementation had R1 NN as its teacher — distilling a known-failing
baseline (R1's own gates report `all_pass: false`) on synthetic
perturbations. That's a smoother R1, not real progress.

Three task files now own the work:

- **`pretraining/002a`** — Story 1996 vocal-tract area-function
  anchor ingestion. `done` (data layer). 11 of 12 vowels via Story
  2008 open-access (1996 raw tables paywalled). `/ɝ/` + 3 nasals + 3
  plosives still missing. NPZ at
  `tractus_controller/outputs/fixtures/story_1996_anchors.npz` (gitignored,
  regen via `scripts/build_story_anchors.py`). Loader at
  `tractus_controller/tractus_controller/body_model/anchors.py`.
- **`pretraining/002b`** — R1-distill body baseline. `in-progress`
  (background worker dispatched 2026-05-08, last seen still running
  ~2-3 hours in; corpus was at 22% at last check, training loop
  build-out alongside). Its artifact must be labeled
  `r1_distill_body_baseline`, NOT `pretraining/002 v0`.
  Quarantine framing per Codex.
- **`pretraining/002`** — anchor-grounded body model. `pending`
  (blocked by 002a + 002b). Re-scoped acceptance gates:
  - Primary: model `area_cm2_tea` + `length_cm_tea` reproduce Story
    values within RMSE/std ≤ 0.10 channel-wise.
  - Secondary: TEA-rendered transfer function of model output matches
    the SAME TEA-rendered Story-anchor baseline (NOT Story's separate
    simulator). Decouples body grade from TEA calibration.
  - Optional audit: Story-published formant deltas, informational only.

**`pretraining/006`** (SPARC bootstrap) — still `decision`-blocked.
Codex's analysis (2026-05-09) confirmed SPARC outputs articulatory
KINEMATICS (12-d EMA + pitch + loudness + 64-d speaker emb), not TEA
renderer-interface targets. SPARC's role is auxiliary supervision on
a body-state latent, NOT primary teacher. Updated spec includes
verified facts (license MIT, deps include WavLM Large 1.26 GB) +
hard gate on model-weights license check + sharpened adoption gate
(must improve TEA-facing validation, not just SPARC MSE). Decision
parked until 002a + 002b land.

### TEA calibration question (Story F2 reaches target)

Long thread (Codex `[session identifier omitted]`, ~5 turns)
to settle "is TEA correctly producing literature-formant vowels."
Three calibration probe iterations under the project's `scripts/`:

1. `sanity_check_vowel_tube.py` — hand-crafted /a, i, u/ tubes,
   voiced glottis at F0=120. Showed simulator responsive to
   constrictions but not exact match to literature targets.
2. `probe_story_spectrum.py` — Story tubes, voiced glottis sweep
   F0 = 120 / 250 / 400 Hz. Showed F0=120 picks F0 harmonics not
   formants; F0=400 gets F2 close but F1=F0.
3. `transfer_function_oral_only.py` — Story tubes, impulse at
   subglottal-pressure boundary, glottis open + side branches
   ablated + mouth-flow output + target-windowed peak assignment.
   Front-vowel F2 reaches target cleanly (/i/ F2=2175 vs Story 2139,
   1.7% off). Back-vowel F1 runs ~15-25% high.

**Verdict** (Codex, settled): TEA is in the ballpark. ±20% on
formants vs Story's separate simulator is acceptable for a v0 body
model that uses **area-function MSE as primary loss** (not formant
matching). The "TEA-Story calibration gap" framing from the 002a
worker was a measurement methodology artifact (LPC + voiced source
at F0=120 + parasitic side-branch peaks).

**Filed as follow-up:** `correctness/009 — TEA fixed-mode
characterization`. Parasitic peaks at ~1287, 2719, 3400, 4054, 4667
Hz appear in every vowel's transfer function under oral-only
isolation — likely subglottal closed-closed modes or wall vibration
modes. Does NOT block pretraining/002.

## Background workers

- **agent-a95e149a2b77a43de** (`worktree-agent-a95e149a2b77a43de`):
  pretraining/002 session 1 — body_model package + corpus generator
  scaffolding. Done; cherry-picked into purification (commit
  `7edbff0`). Worktree may still be locked from the agent process.
- **agent-a642d4dbaa26ae299** (`worktree-agent-a642d4dbaa26ae299`):
  pretraining/002b session 2 — R1-distill training. Cherry-picked
  (commits `4583959` + `79fd62c`). Added `scripts/train_body_model.py`
  (724 lines), `scripts/eval_body_model.py` (573 lines),
  `tractus_controller/tests/body_model/test_train_body_model.py`
  (152 lines, 4 new smoke tests). Body-model test count is now
  16/16 passing on purification HEAD. The training script does NOT
  yet add the provenance discipline (artifact label
  `r1_distill_body_baseline`, teacher-provenance metadata, R1-vs-distill
  comparison probe per Codex 2026-05-08 thread). Add those when first
  running full training; the eval script `scripts/eval_body_model.py`
  is a reasonable place to add the comparison probe.
- **agent-ac8a5f72dbbf86638** (`worktree-agent-ac8a5f72dbbf86638`):
  pretraining/002a session — Story anchor ingest. Done; cherry-picked
  into purification (commit `368068c`).

## Next dispatchable work

In order of likely priority:

1. **Confirm 002b worker status.** If still running, let it finish
   and then merge. If silent-stuck, unstick it.
2. **Dispatch pretraining/002 (anchor-grounded body model)** once
   002b lands. The task spec is up to date; primary loss is
   area-function MSE against Story anchors; eval is the
   transfer-function-parity gate using `transfer_function_oral_only.py`
   methodology (NOT the original 002a worker's voiced-source LPC gate).
3. **User decision on pretraining/006 (SPARC).** Whether to dispatch
   the 2-session analysis-only phase. Cheap and informative; produces
   the data needed for the eventual adopt-vs-defer call.
4. **Package renames** (TC → tc, TEA → tea, shared) — task #2 in
   the local Claude task list. Authorized by the user but deferred
   for risk reasons (venv rebuild). Not blocking anything; pure
   ergonomics work.
5. **correctness/009** (TEA fixed-mode characterization) — S effort,
   useful diagnostic but does NOT block anything.
6. **Re-run inverse-fit demo with better loss** — the
   `spectral_template_loss_jnp` used in `scripts/inverse_fit_demo.py`
   has the same F0-harmonic pollution as the worker's LPC gate
   originally did. With a transfer-function-based loss instead, the
   "F2 stuck at 1227" finding might dissolve. Not on critical path
   (pretraining/002 supersedes inverse-fit polish) but useful as a
   parity check that the optimizer story changes.

## Files that matter

New (this session):
- `tractus_controller/tractus_controller/body_model/{__init__,model,interp,channels,reachability,anchors}.py`
- `tractus_controller/tests/body_model/test_body_model.py`
- `scripts/{gen_body_corpus,build_story_anchors,sanity_check_story_anchors,sanity_check_vowel_tube,probe_story_spectrum,transfer_function_story_anchors,transfer_function_oral_only}.py`
- `workplan/tasks/04_pretraining/{002a_story_anchor_ingest,002b_r1_distill_baseline}.md`
- `workplan/tasks/01_correctness/009_tea_fixed_modes.md`

Updated:
- `workplan/tasks/04_pretraining/{002_body_model_pretrain,006_sparc_bootstrap}.md`
- `workplan/INDEX.md` (status table + dependency graph)
- `ARCHITECTURE.md`, `CLAUDE.md`, multiple READMEs (purification scrubs)

Deleted (this session):
- `tractus_controller/plan/` (entire directory: `00-overview.md`, `01-decisions.md`, `80-vtl-removal-handoff.md`, `90-dependency-removal.md`)
- `scripts/ci/lint_{no_vtl_leftovers,channel_smallness,residual_energy_cap}.py`
- `tests/ci_fixtures/violations/has_vtl_in_module.py`
- `workplan/runbooks/{anticheat_unreachable,dimensionality_report,plausibility_panel,_matched_reference_impl}.py`
- `workplan/runbooks/matched_reference_eval.sh`
- `workplan/runbooks/schemas/{anticheat,dimensionality,matched_reference,plausibility}.schema.json`
- `workplan/runbooks/test_fixtures/` (entire directory)
- `workplan/tasks/00_infra/004_remove_vtl_anchoring.md`
- 27 untracked `.py` files under `tractus_controller/outputs/{copy_synth_runners,bench,...}/` and `tractus_engine_advanced/outputs/`

## Codex thread to continue if needed

`[session identifier omitted]` — multi-turn thread covering:
- Initial data-source critique (R1 distill = bad teacher, anchors needed)
- Adversarial steelman of "let session 2 finish anyway"
- SPARC fact-grounded analysis (license, deps, output schema)
- Story-anchor calibration triangulation (3 probe iterations)
- Final: "good enough, move on" verdict for pretraining/002 path

Use `codex-reply` to continue; don't start fresh unless cold review
needed.
