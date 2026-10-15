#!/usr/bin/env bash
#
# whisper_gate.sh — ASR transcription + word-WER + Jaccard intelligibility gate.
#
# Word-level WER from Whisper segments + word-set Jaccard against the
# ground-truth transcript. Per ARCH §1147-1150 this gate alone is a
# noisy stop-loss; pair with the plausibility panel and a
# template-only baseline before declaring intelligibility.
#
# This script is a thin shell wrapper around _whisper_gate_impl.py so
# that callers can invoke it as `bash runbooks/whisper_gate.sh`
# while we keep the metrics in Python.
#
# Implementation notes:
# - Standalone WER + Jaccard utility under runbooks/_lib/. Does NOT
#   import from tractus_controller.copy_synth.gates.
# - ASR backend: faster_whisper preferred, whisperx fallback. Set
#   WHISPER_GATE_MOCK_TRANSCRIPT to bypass real ASR (smoke fixtures).

set -euo pipefail

usage() {
    cat <<'EOF'
whisper_gate.sh — ASR + word-WER + Jaccard pass/fail gate.

Required:
  --audio PATH               Audio file to transcribe.
  --target "..."             Ground-truth transcript.

Optional:
  --threshold-wer FLOAT      Pass requires WER < threshold. Default 0.25.
  --threshold-jaccard FLOAT  Pass requires Jaccard > threshold. Default 0.30.
  --model NAME               faster_whisper / whisperx model. Default "base".

Output:
  JSON on stdout (validated against schemas/whisper_gate.schema.json).
  One-line PASS/FAIL summary on stderr.

Exit:
  0 = pass (WER < wer-threshold AND Jaccard > jaccard-threshold).
  1 = fail.
  2 = bad arguments.

Env:
  WHISPER_GATE_MOCK_TRANSCRIPT  If set, skip ASR and use literal value.
EOF
}

if [[ $# -eq 0 ]] || [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    usage
    exit 0
fi

# Pass through to the Python implementation. argparse handles arg
# validation; we forward everything verbatim.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/_whisper_gate_impl.py" "$@"
