# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A self-study curriculum (Korean-language, followed via external Notion lecture notes) built around the SECOM semiconductor manufacturing dataset. The goal stated in `README.md`: analyze process sensor readings to flag flows likely to fail inspection early (반도체 공정 데이터로 불량 조기 판별). There is no application code, package, or test suite — the repository is a sequence of Jupyter notebooks, one lab per subfolder, that progressively clean the raw sensor data and build toward a classification model. It is not a git repository.

## Environment

- Python 3.14.7, packages installed via `pip install pandas numpy matplotlib seaborn scikit-learn jupyter` (pandas 3.0.5, numpy 2.5.1, matplotlib 3.11.1, seaborn 0.13.2, scikit-learn 1.9.0).
- No build, lint, or test commands — run notebook cells directly (VS Code Jupyter extension or `jupyter lab`). There is no CLI entry point and nothing to compile.
- Windows/OneDrive path: `C:\Users\solbe\OneDrive\바탕 화면\secom-project`. Folder and file names contain Korean text and spaces.
- matplotlib on this machine needs `plt.rcParams["font.family"] = "Malgun Gothic"` and `plt.rcParams["axes.unicode_minus"] = False` before plotting anything with Korean labels/titles, or text renders as missing-glyph boxes.

## Data flow architecture

The core thing to understand before touching any notebook is the **cross-lab data pipeline** — each lab reads a file produced by an earlier lab, and the chain only makes sense in order:

1. `data/04_secom.csv` — raw SECOM dataset, 1567 rows × 592 columns (`measured_at`, `sensor_001`…`sensor_590`, `result` where result is `양품`/`불량`, ~6.6% 불량). Treated as read-only source data.
2. `day01/` — pandas fundamentals labs (`lab01_review`, `lab02_workspace`, `lab03_process-card`, `lab04_control-chart`, `lab05_sensor-diagnosis`). Exploratory; nothing here is a required dependency for later days.
3. `day02/lab06_clean-dataset/lab06_clean_dataset.ipynb` — the real pipeline start. Loads the raw CSV and derives, in order: `df` → `df1` (drop sensor columns that are >50% missing, constant, or std ≤ 0.001) → `df2` (drop one column from every pair of sensors with |correlation| ≥ 0.9, keeping whichever has fewer missing values) → `rank_table` (sensors ranked by |correlation| with a binary `불량여부` label) → `df3`/`df3b` (top-50 / top-20 sensors by that ranking). Final output written to `day02/lab06_clean-dataset/results/secom_clean.csv` (1567 rows × 51 columns: top-50 sensors + `result`) and `secom_clean_b.csv` (top-20 variant).
4. `day03/lab07_train-test-split/` and `day03/lab08_baseline-model/` — load `secom_clean.csv` (via relative path `../../day02/lab06_clean-dataset/results/secom_clean.csv`), median-fill remaining NaNs in sensor columns only, add a numeric `불량여부` (불량=1/양품=0) column alongside the original `result`, split with `train_test_split(..., test_size=0.2, stratify=y, random_state=42)`, then train/evaluate classifiers (a from-scratch all-zero baseline first, then e.g. `DecisionTreeClassifier` or a `StandardScaler`+`LogisticRegression` pair) against that fixed split.

When adding a new `dayNN/labNN_*` notebook that continues the pipeline, load the previous stage's `results/*.csv` rather than re-deriving it from raw data, unless the task explicitly asks to rebuild that stage.

## Notebook conventions (established across existing labs — follow them when editing)

- **Folder/file naming**: `dayNN/labNN_topic-name/labNN_topic_name.ipynb` — hyphen in the folder name, underscore in the matching file name.
- **Path convention**: notebooks reference data with relative paths from their own folder, e.g. `pd.read_csv("../../data/04_secom.csv")` or `pd.read_csv("../../day02/lab06_clean-dataset/results/secom_clean.csv")`. A lab that saves output writes to a `results/` subfolder next to itself, created with `os.makedirs("results", exist_ok=True)`.
- **Notebook structure**: a title markdown cell, then `## Step N. <설명>` markdown headings each immediately followed by one code cell, each immediately followed by a `### 결과 정리 (실행 결과 기준)` markdown cell restating the code cell's actual printed/tabular output (never invented numbers — always what running the cell produced).
- **Variable names are Korean and are load-bearing**: `df`, `df1`, `df2`, `df3` are pipeline stage names used consistently across labs; other identifiers like `센서`, `평균`, `표준편차`, `위선`/`아래선` (UCL/LCL), `sensor_cols`, `남는센서`, `불량여부`, `rank_table`, `X`/`y`, `X_train`/`X_test`/`y_train`/`y_test` are reused by name across later cells and later notebooks. Keep new code consistent with these names rather than introducing English equivalents.
- **Sort determinism**: when sorting by a column that can have tied values (e.g. correlation coefficients that hit exactly 1.0, or 0% missing-ratio columns), pass `kind="mergesort"` to `sort_values` — the default quicksort's tie order is not guaranteed stable and previously produced results that didn't match what was written up.
- Cells must be run top-to-bottom; a fresh kernel run out of order raises `NameError` for `df` or other stage variables. This trips people up regularly — if a notebook errors with a `NameError` on a variable defined in an earlier cell, the fix is running from the top (or Run All), not editing code.
