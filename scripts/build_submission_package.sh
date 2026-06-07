#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/user/ssdmain/dcase-adqa
SUB=${ROOT}/submission
PKG=${ROOT}/outputs/final_submission/package
TASK=${PKG}/task5
LABEL=Nam_IND_task5

rm -rf "${PKG}"
mkdir -p "${TASK}"

cp "${ROOT}/paper/main.pdf" "${TASK}/${LABEL}.technical_report.pdf"

declare -A outputs=(
  [1]="${ROOT}/outputs/final_submission/csv_basejudge/Nam_IND_task5_A_train_only_empty5_2k_basejudge.output.csv"
  [2]="${ROOT}/outputs/final_submission/csv_basejudge/Nam_IND_task5_B_train_dev_empty5_2k_basejudge.output.csv"
  [3]="${ROOT}/outputs/final_submission/csv_basejudge/Nam_IND_task5_C_train_dev_empty2p5_3k_basejudge.output.csv"
  [4]="${ROOT}/outputs/final_submission/csv_qwen3_8b_judge/Nam_IND_task5_ensemble_ABC_Btie_qwen3_8b_judge.output.csv"
)

for idx in 1 2 3 4; do
  dir="${TASK}/${LABEL}_${idx}"
  mkdir -p "${dir}"
  cp "${outputs[$idx]}" "${dir}/${LABEL}_${idx}.output.csv"
  cp "${SUB}/task5/${LABEL}_${idx}/${LABEL}_${idx}.meta.yaml" "${dir}/${LABEL}_${idx}.meta.yaml"
  cp "${SUB}/post_process.py" "${dir}/${LABEL}_${idx}.post_process.py"
done

cd "${PKG}"
zip -r "${ROOT}/outputs/final_submission/${LABEL}.zip" task5
echo "wrote ${ROOT}/outputs/final_submission/${LABEL}.zip"
