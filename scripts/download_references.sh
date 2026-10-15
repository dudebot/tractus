#!/usr/bin/env bash
#
# Mirror references for ARCHITECTURE.md into docs/references/.
#
# License posture: each PDF is fetched directly from its canonical
# host the same way clicking a link in a browser would. Nothing is
# redistributed from this repo; PDFs are gitignored
# (docs/references/**/*.pdf). Pages that are HTML landing pages
# (project sites, GitHub repos, dataset documentation pages, ICSLP
# archive index pages) are listed without a fetch.
#
# Run from repo root:
#   bash scripts/download_references.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p docs/references/core
mkdir -p docs/references/foundations
mkdir -p docs/references/baselines
mkdir -p docs/references/datasets
mkdir -p docs/references/pages

download_url () {
    local label="$1"
    local url="$2"
    local out_path="$3"
    if [ -f "$out_path" ]; then
        echo "[skip]  $out_path (exists)"
        return
    fi
    echo "[fetch] $label -> $out_path"
    curl --fail --silent --show-error --location \
        --user-agent "tractus-references/1.0 (academic mirror; non-redistributing)" \
        "$url" \
        --output "$out_path" || {
            echo "[warn] failed to fetch $label ($url)"
            rm -f "$out_path"
        }
}

download_arxiv () {
    local arxiv_id="$1"
    local out_path="$2"
    download_url "arXiv:$arxiv_id" \
        "https://arxiv.org/pdf/${arxiv_id}.pdf" \
        "$out_path"
}

# Some legitimate academic hosts have outdated SSL chains
# (e.g. sail.usc.edu's cert chain). curl -k skips verification —
# same effect as a browser user clicking through the warning.
download_url_insecure () {
    local label="$1"
    local url="$2"
    local out_path="$3"
    if [ -f "$out_path" ]; then
        echo "[skip]  $out_path (exists)"
        return
    fi
    echo "[fetch-insecure] $label -> $out_path"
    curl --fail --silent --show-error --location --insecure \
        --user-agent "tractus-references/1.0 (academic mirror; non-redistributing)" \
        "$url" \
        --output "$out_path" || {
            echo "[warn] failed to fetch $label ($url)"
            rm -f "$out_path"
        }
}

note_landing_only () {
    local label="$1"
    local url="$2"
    echo "[landing-only] $label  -- $url"
}

note_textbook () {
    local label="$1"
    local isbn_or_ref="$2"
    echo "[textbook]     $label  -- $isbn_or_ref"
}

echo "=== Frontier core (arXiv) ==="

download_arxiv "2406.12998" "docs/references/core/2024_sparc_vocal_tract_kinematics.pdf"
download_arxiv "2409.02451" "docs/references/core/2024_ddsp_articulatory_vocoder.pdf"
download_arxiv "2506.10289" "docs/references/core/2025_rt_vc_articulatory_voice_conversion.pdf"
download_arxiv "2510.05619" "docs/references/core/2025_teaching_machines_to_speak.pdf"

echo
echo "=== Differentiable simulator / inverse-fitting ==="

download_url "Chen & Sitzmann 2025 (DBM Findings)" \
    "https://openreview.net/pdf?id=75AUKPeJme" \
    "docs/references/core/2025_chen_sitzmann_diff_vocal_simulator.pdf"
download_url "VocalTrax (OpenReview)" \
    "https://openreview.net/pdf?id=kIDxwvWznF" \
    "docs/references/core/2024_vocaltrax_articulatory_optimization.pdf"
download_arxiv "2307.04702" \
    "docs/references/core/2023_dafx_vocal_tract_area_gradient_descent.pdf"

echo
echo "=== Frication / noise-source physics ==="

download_url "Narayanan & Alwan 2000 fricative noise (UCLA)" \
    "https://www.seas.ucla.edu/spapl/paper/narayanan_sap00.pdf" \
    "docs/references/core/2000_narayanan_alwan_fricative_noise.pdf"

echo
echo "=== Modern TTS / VC baselines (arXiv) ==="

download_arxiv "2410.06885" "docs/references/baselines/2024_f5_tts.pdf"
download_arxiv "2407.05407" "docs/references/baselines/2024_cosyvoice.pdf"
download_arxiv "2412.10117" "docs/references/baselines/2024_cosyvoice_2.pdf"
download_arxiv "2502.18924" "docs/references/baselines/2025_megatts3.pdf"

echo
echo "=== Acoustic-to-articulatory inversion (recent arXiv) ==="

download_arxiv "2603.11845" "docs/references/core/2026_aai_clean_speech_mri_trained.pdf"
download_arxiv "2603.11847" "docs/references/core/2026_vocal_tract_reconstruction_phonetic_segmentation.pdf"
download_arxiv "2603.28723" "docs/references/core/2026_complete_vocal_tract_inversion_rtmri.pdf"
download_arxiv "2511.00428" "docs/references/core/2025_pinn_speech_production.pdf"

echo
echo "=== Gesture / motor-control foundations ==="

download_arxiv "2210.09334" "docs/references/foundations/2022_torchdiva.pdf"
download_url_insecure "Saltzman & Munhall 1989 task-dynamic model (USC)" \
    "https://sail.usc.edu/~lgoldste/ArtPhon/Papers/Saltzman%26Munhall.PDF" \
    "docs/references/foundations/1989_saltzman_munhall_task_dynamic.pdf"
download_url "Browman & Goldstein 1992 articulatory phonology (UMD)" \
    "https://users.umiacs.umd.edu/~nhf/courses/ling620/BrowmanGoldstein92.pdf" \
    "docs/references/foundations/1992_browman_goldstein_articulatory_phonology.pdf"

echo
echo "=== Acoustic-phonetic anatomy reference (textbooks; not auto-mirrored) ==="

note_textbook "Stevens 1998 *Acoustic Phonetics* (MIT Press)" "ISBN 978-0262692502"
note_textbook "Fant 1960 *Acoustic Theory of Speech Production* (Mouton)" "out of print; library/used market"
note_textbook "Hillenbrand et al. 1995 American English vowel formants" "JASA 97"

echo
echo "=== Landing-only (HTML pages or repos; not direct PDFs) ==="

note_landing_only "SPARC repo" \
    "https://github.com/Berkeley-Speech-Group/Speech-Articulatory-Coding"
note_landing_only "DDSP-Articulatory repo" \
    "https://github.com/mctlab/DDSP-Articulatory-Synthesizer"
note_landing_only "RT-VC repo" \
    "https://github.com/Berkeley-Speech-Group/RT-VC"
note_landing_only "VocalTrax repo" \
    "https://github.com/hunter-heidenreich/VocalTrax"
note_landing_only "Kokoro repo" \
    "https://github.com/hexgrad/kokoro"
note_landing_only "Maeda 1990 ICSLP (HTML index page)" \
    "https://www.isca-archive.org/icslp_1990/maeda90b_icslp.html"

echo
echo "=== Datasets - direct (free, no agreement) ==="

# MOCHA-TIMIT: free for non-commercial use per CSTR Edinburgh.
# Two English speakers (fsew0 female, msak0 male) + prompts + license.
# Tarballs are ~150MB each; gitignored under docs/references/**/*.tar.gz.
mkdir -p docs/references/datasets/mocha_timit
download_url "MOCHA-TIMIT prompts" \
    "https://data.cstr.ed.ac.uk/mocha/mocha-timit.txt" \
    "docs/references/datasets/mocha_timit/mocha-timit.txt"
download_url "MOCHA-TIMIT license" \
    "https://data.cstr.ed.ac.uk/mocha/LICENCE.txt" \
    "docs/references/datasets/mocha_timit/LICENCE.txt"
download_url "MOCHA-TIMIT README v1.2" \
    "https://data.cstr.ed.ac.uk/mocha/README_v1.2.txt" \
    "docs/references/datasets/mocha_timit/README_v1.2.txt"
download_url "MOCHA-TIMIT fsew0 (female speaker, ~155MB)" \
    "https://data.cstr.ed.ac.uk/mocha/fsew0_v1.1.tar.gz" \
    "docs/references/datasets/mocha_timit/fsew0_v1.1.tar.gz"
download_url "MOCHA-TIMIT msak0 (male speaker, ~154MB)" \
    "https://data.cstr.ed.ac.uk/mocha/msak0_v1.1.tar.gz" \
    "docs/references/datasets/mocha_timit/msak0_v1.1.tar.gz"

# Hillenbrand 1995 vowel formant data - original WMU page (404) is
# mirrored on GitHub. The .dat files (per-speaker formant measurements)
# are inside the tarball.
mkdir -p docs/references/datasets/hillenbrand_1995
download_url "Hillenbrand 1995 vowel formant data (GitHub mirror)" \
    "https://github.com/santiagobarreda/hillenbrand_et_al_1995/archive/refs/heads/main.tar.gz" \
    "docs/references/datasets/hillenbrand_1995/hillenbrand_et_al_1995.tar.gz"

echo
echo "=== Datasets - gated (require human-in-the-loop) ==="

note_landing_only "mngu0 (email Korin Richmond for UserID)" \
    "https://homepages.inf.ed.ac.uk/korin/site/research/mngu0/"
note_landing_only "USC-TIMIT (sign SAIL release agreement)" \
    "https://sail.usc.edu/span/usc-timit/"
note_landing_only "USC Morphology MRI (sign SAIL release agreement)" \
    "https://sail.usc.edu/span/morphdb/"

echo
echo "Done. Re-run to fetch any missing PDFs."
echo "All canonical URLs are in docs/references/pages/reference_links.md."
