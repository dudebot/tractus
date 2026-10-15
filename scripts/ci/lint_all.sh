#!/usr/bin/env bash
# Run every cheap CI guardrail lint. Exits non-zero if any one fails.
#
# Used by:
#   - GitHub Actions CI (.github/workflows/ci.yml: lint job)
#   - `make ci` (Makefile)
#   - pre-commit hook (.pre-commit-config.yaml: lint-all id)
#
# Each script under scripts/ci/lint_*.py prints `OK: ...` or `FAIL: ...`
# and exits accordingly. We aggregate so the user sees ALL failures at
# once, not just the first.

set -u
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

PY="${PYTHON:-python3}"

scripts=(
  "${script_dir}/lint_privacy.py"
  "${script_dir}/lint_surrogate_ban.py"
  "${script_dir}/lint_non_circularity.py"
  "${script_dir}/lint_doc_coherence.py"
  "${script_dir}/lint_reference_pack.py"
  "${script_dir}/lint_consent.py"
)

failed=()
for s in "${scripts[@]}"; do
  echo "--- $(basename "$s") ---"
  if ! "${PY}" "${s}"; then
    failed+=("$(basename "$s")")
  fi
done

echo
if [ ${#failed[@]} -ne 0 ]; then
  echo "CI lint FAILED: ${failed[*]}"
  exit 1
fi
echo "CI lint OK"
exit 0
