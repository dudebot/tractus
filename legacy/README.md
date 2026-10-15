# legacy/

Code that was load-bearing in a previous design iteration and is no
longer in any active path. Kept for archaeology and so reviewers can
see what the previous shape looked like; not imported by anything
in the active tree.

| File | Retired | Replaced by | Note |
|---|---|---|---|
| `body_model_conv_mlp.py` | 2026-05-09 | forthcoming `body_model/neural_field.py` | Generic conv-MLP that emitted 385 renderer fields directly (incl. raw 93-section areas). Rejected per the 2026-05-09 course correction. |
| `gen_body_corpus.py` | 2026-05-09 | (none — corpus paradigm replaced by SPARC label extraction in active task 002) | Synthetic gesture-perturbation corpus generator built around the conv-MLP body model. |
| `train_body_model.py` | 2026-05-09 | forthcoming Week-2 trainer (active task 003) | 724-line training loop with channel-norm MSE + smoothness + static + refusal losses. Targeted area-MSE against R1-distill / Story anchors as primary loss — paradigm rejected. |
| `eval_body_model.py` | 2026-05-09 | (TBD) | 573-line eval script for the conv-MLP body model. |
| `test_body_model.py` | 2026-05-09 | (TBD) | 12 unit tests for the conv-MLP architecture. |
| `test_train_body_model.py` | 2026-05-09 | (TBD) | 4 smoke tests for the conv-MLP training loop. |

See `workplan/handoff/2026-05-09_course_correction.md` for the
full rationale and the new architecture.
