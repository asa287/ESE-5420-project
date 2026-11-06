# ESE 542 baseline checkpoint

The submission files are [`baseline.txt`](baseline.txt) and optional
[`baseline.png`](baseline.png). Before submitting, confirm and replace the Canvas
group-number placeholder. The Code URL points to [`baseline.ipynb`](baseline.ipynb)
in this repository. Only the `.txt` and `.png` go to the assignment upload.

Open the notebook in [Colab](https://colab.research.google.com/github/asa287/ESE-5420-project/blob/main/baseline.ipynb).

## Reproduce

Use Python 3.9 and the pinned requirements:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python run_baseline.py
```

Alternatively upload `baseline.ipynb` to Colab and run all cells. It downloads the
official UCI ZIP. No article text or URL list is exported. The notebook contains the
same source as the script; JSON/CSV files expose the exact numeric results.

## Frozen evaluation design

- Keep the latest 145 observed publication days (9,112 articles) for the final paper.
- Earlier 579 days form development. The last 96 development days are the checkpoint
  holdout; the preceding 483 days are for fitting and five expanding-window CV folds.
- Form folds on unique publication days, never individual shuffled rows. Equal observed
  day counts can cover slightly different calendar durations; fold SD is descriptive.
- Exclude all nine `kw_` and three `self_reference_` share-statistic fields. Exclude
  `weekday_is_sunday` (reference category) and redundant `is_weekend`. Drop URL and
  timedelta from predictors; timedelta only orders days. Keep 44 predictors.
- Three lexical ratios outside [0,1] become missing using a fixed rule. Fit median
  imputation and scaling independently in every training fold and bootstrap refit.
- Tune 20 alpha values with `np.logspace(-4,0,20)`, using equal-weight mean fold MAE.
  Refit the selected pipeline on the entire pre-checkpoint training period.
- Report checkpoint MAE with a 2,000-draw percentile interval from seven-calendar-day
  moving blocks. Keep all articles on each sampled day. A final partial block is
  truncated to preserve the original number of calendar days. Sample counts vary.
  The training-mean comparator uses only each fold's or fit period's targets.
- The evaluation interval conditions on this fitted model. It does not include tuning
  uncertainty, training variability, or unobserved future drift. The 20 training block
  refits separately check selection variability at a fixed alpha; they do not establish
  true feature recovery. Selection means absolute standardized coefficient > 1e-8.

The initial seed is 542 for evaluation and 543 for training resampling. No seed search.
`protocol.json` is written before CV and scoring, and `results.json` records the ZIP
checksum, environment versions, splits, scores, and pilot list sizes.

## Feedback incorporated and planned comparison

Five temporal folds replace three. Add the training-mean comparator and the 20-refit
selection pilot. Reserve a checkpoint holdout to avoid scoring the final test prematurely.
For the future elastic-net versus lasso comparison, define delta as elastic-net minus
lasso test MAE. An interval entirely at or below zero supports no increase; an interval
including both signs is inconclusive rather than evidence of equivalence. Report the
paired seven-day block-bootstrap 95% interval after all models and rules are frozen.
Also add the suggested known-truth simulation, correlated predictor copies and
precision/recall, and scale selection refits to 200 before the early draft if feasible.
Compare Jaccard overlap at matched list sizes; omit forests first if time is limited.

## Limitations and source checks

The actual official ZIP has 39,644 rows; the UCI landing-page instance count differs.
No rows are removed. Raw lexical anomalies are handled without fitting on validation.
Upstream LDA/topic construction and article-share exposure time remain limitations:
fold-local scaling cannot undo upstream construction. This is predictive association,
not causation or discovery of the true feature list. The saved script does not score
the final test, though the archive is loaded to establish membership and row counts.

Source: Fernandes, K., Vinagre, P., Cortez, P., & Sernadela, P. (2015).
[Online News Popularity](https://doi.org/10.24432/C5NS3V). UCI Machine Learning Repository.
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Derived analysis applies
exclusions, missing-value rules and a log target; the original data is unmodified.
Implementation references: [scikit-learn pipelines and leakage](https://scikit-learn.org/stable/common_pitfalls.html)
and [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).

AI assistance: Codex read the supplied proposal and feedback, drafted and executed the
pipeline, prepared the checkpoint text/figure and checked outputs. Team members must
review and rerun the code themselves before stating that they independently verified it.
