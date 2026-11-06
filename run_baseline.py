"""Reproducible ESE 542 checkpoint. Final 145 observed days are never scored.

Run: python run_baseline.py (downloads the official UCI ZIP if absent).
"""
from pathlib import Path
import hashlib
import json
import platform
import warnings
import zipfile
import urllib.request

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
SEED = 542
ALPHAS = np.logspace(-4, 0, 20)
RATIOS = ['n_unique_tokens', 'n_non_stop_words', 'n_non_stop_unique_tokens']
warnings.filterwarnings('error', category=ConvergenceWarning)
warnings.filterwarnings('error', category=RuntimeWarning)


def pipeline(alpha):
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                         Lasso(alpha=alpha, max_iter=100000, tol=1e-6,
                               selection='cyclic'))


def blocks(day, rng):
    """Moving blocks of seven calendar days, truncated to original day span.

    Resample days (including empty calendar days), then keep every article
    from each sampled day. This yields article-weighted MAE with variable N.
    """
    calendar = np.arange(day.min(), day.max() + 1)
    starts = rng.integers(0, len(calendar) - 6,
                          size=int(np.ceil(len(calendar) / 7)))
    sampled = np.concatenate([calendar[s:s+7] for s in starts])[:len(calendar)]
    rows = {d: np.flatnonzero(day == d) for d in calendar}
    return np.concatenate([rows[d] for d in sampled])


def main():
    (ROOT / 'data').mkdir(exist_ok=True)
    archive = ROOT / 'data/online_news_popularity.zip'
    if not archive.exists():
        urllib.request.urlretrieve(
            'https://archive.ics.uci.edu/static/public/332/online+news+popularity.zip', archive)
    with zipfile.ZipFile(archive) as z:
        df = pd.read_csv(z.open('OnlineNewsPopularity/OnlineNewsPopularity.csv'),
                         skipinitialspace=True)
    df = df.sort_values('timedelta', ascending=False, kind='stable').reset_index(drop=True)
    day = -df.timedelta.to_numpy(dtype=int)
    observed = np.unique(day)
    development_days, final_days = observed[:-145], observed[-145:]
    fit_days, check_days = development_days[:-96], development_days[-96:]
    fit = np.flatnonzero(np.isin(day, fit_days))
    check = np.flatnonzero(np.isin(day, check_days))
    test = np.flatnonzero(np.isin(day, final_days))
    excluded = [c for c in df if c.startswith('kw_') or c.startswith('self_reference_')]
    excluded += ['weekday_is_sunday', 'is_weekend']
    features = [c for c in df if c not in excluded + ['url', 'timedelta', 'shares']]
    assert len(features) == 44 and len(excluded) == 14
    assert len(df) == 39644 and len(development_days) == 579
    assert len(test) == 9112 and len(fit) + len(check) == 30532
    assert df.url.duplicated().sum() == 0
    # Final-test predictors and targets are not used in preprocessing or fitting.
    dev = df.iloc[np.r_[fit, check]].copy()
    invalid = {}
    for col in RATIOS:
        bad = ~dev[col].between(0, 1)
        invalid[col] = int(bad.sum())
        dev.loc[bad, col] = np.nan
    X = dev[features]
    y = np.log1p(dev.shares.to_numpy())
    Xfit, Xcheck = X.iloc[:len(fit)], X.iloc[len(fit):]
    yfit, ycheck = y[:len(fit)], y[len(fit):]
    # Build folds on distinct publication days so a day cannot straddle folds.
    splits = []
    fold_info = []
    for tr, va in TimeSeriesSplit(n_splits=5).split(fit_days):
        train_rows = np.flatnonzero(np.isin(day[fit], fit_days[tr]))
        val_rows = np.flatnonzero(np.isin(day[fit], fit_days[va]))
        assert day[fit][train_rows].max() < day[fit][val_rows].min()
        splits.append((train_rows, val_rows))
        fold_info.append(dict(train_days=len(tr), validation_days=len(va),
                              train_rows=len(train_rows), validation_rows=len(val_rows)))
    config = dict(seed=SEED, alphas=ALPHAS.tolist(), features=features,
                  excluded=excluded, invalid_ratio_rule='outside [0,1] -> NaN',
                  folds=fold_info, final_test='latest 145 observed days; untouched',
                  checkpoint='last 96 development days; tuning uses preceding 483 days',
                  bootstrap='7 calendar-day moving blocks; 2000 evaluation draws; 20 training refits',
                  coefficient_threshold=1e-8,
                  future_comparison='elastic-net minus lasso MAE; report paired 95% interval; '
                  'upper bound <= 0 supports no increase; interval including zero is inconclusive')
    # Persist the protocol before inspecting tuning or checkpoint scores.
    (ROOT / 'protocol.json').write_text(json.dumps(config, indent=2) + '\n')
    search = GridSearchCV(pipeline(0.01), {'lasso__alpha': ALPHAS}, cv=splits,
                          scoring='neg_mean_absolute_error', refit=True, n_jobs=1,
                          error_score='raise')
    search.fit(Xfit, yfit)
    alpha = float(search.best_params_['lasso__alpha'])
    best = search.best_index_
    fold_maes = [-float(search.cv_results_[f'split{k}_test_score'][best]) for k in range(5)]
    mean_fold_maes = [float(np.mean(np.abs(yfit[va] - yfit[tr].mean()))) for tr, va in splits]
    errors = np.abs(ycheck - search.predict(Xcheck))
    mean_errors = np.abs(ycheck - yfit.mean())
    rng = np.random.default_rng(SEED)
    draws = np.empty((2000, 3))
    for b in range(len(draws)):
        idx = blocks(day[check], rng)
        draws[b] = [errors[idx].mean(), mean_errors[idx].mean(),
                    (errors[idx] - mean_errors[idx]).mean()]
    intervals = np.quantile(draws, [0.025, 0.975], axis=0).T.tolist()
    rng = np.random.default_rng(SEED + 1)
    coefficients = []
    for b in range(20):
        idx = blocks(day[fit], rng)
        model = pipeline(alpha).fit(Xfit.iloc[idx], yfit[idx])
        coefficients.append(model.named_steps['lasso'].coef_)
        print(f'Bootstrap refit {b+1}/20', flush=True)
    coefficients = np.array(coefficients)
    selected = np.abs(coefficients) > 1e-8
    frequency = selected.mean(axis=0)
    date = lambda d: str((pd.Timestamp('2015-01-08') + pd.Timedelta(days=int(d))).date())
    results = dict(n=len(df), inputs=44, development_n=len(dev), fit_n=len(fit),
                   checkpoint_n=len(check), final_test_n=len(test),
                   publication_dates=dict(fit=[date(fit_days[0]), date(fit_days[-1])],
                                          checkpoint=[date(check_days[0]), date(check_days[-1])],
                                          final=[date(final_days[0]), date(final_days[-1])]),
                   invalid_development_cells=invalid, dropped_rows=0, alpha=alpha,
                   cv_mae=fold_maes, cv_mean_mae=float(np.mean(fold_maes)),
                   cv_sd=float(np.std(fold_maes, ddof=1)), mean_predictor_cv_mae=mean_fold_maes,
                   checkpoint_mae=float(errors.mean()), checkpoint_ci95=intervals[0],
                   mean_predictor_mae=float(mean_errors.mean()), mean_predictor_ci95=intervals[1],
                   paired_difference=float((errors-mean_errors).mean()), paired_ci95=intervals[2],
                   bootstrap_selection_sizes=selected.sum(axis=1).tolist(),
                   sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                   versions=dict(python=platform.python_version(), numpy=np.__version__,
                                 pandas=pd.__version__, sklearn=sklearn.__version__),
                   test_scored=False)
    (ROOT / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    cv = pd.DataFrame(search.cv_results_)
    cv.to_csv(ROOT / 'cv_results.csv', index=False)
    pd.DataFrame(coefficients, columns=features).to_csv(ROOT / 'bootstrap_coefficients.csv', index=False)
    pd.DataFrame({'predictor':features, 'selection_frequency':frequency}).to_csv(
        ROOT / 'selection_frequencies.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(15, 12), gridspec_kw={'width_ratios':[1, 1.6]})
    scores = [errors.mean(), mean_errors.mean()]
    ci = np.array(intervals[:2])
    axes[0].errorbar([0,1], scores, yerr=[np.array(scores)-ci[:,0], ci[:,1]-scores],
                     fmt='o', capsize=6, color='#226080', markersize=8)
    axes[0].set_xticks([0,1], ['Lasso', 'Training mean'])
    axes[0].set_ylabel('MAE on log(1 + shares)')
    axes[0].set_xlabel('Predictor')
    axes[0].set_title('Checkpoint prediction error\n95% seven-day block-bootstrap intervals', fontsize=12)
    axes[0].grid(axis='y', alpha=.2)
    order = np.argsort(frequency, kind='stable')
    axes[1].barh(np.array(features)[order], frequency[order], color='#226080')
    axes[1].set_xlim(0,1.03)
    axes[1].set_xlabel('Selection frequency across 20 training refits')
    axes[1].set_ylabel('Predictor')
    axes[1].tick_params(axis='y', labelsize=8)
    axes[1].set_title('Pipeline pilot: fixed-penalty lasso\nSeven-day blocks; |coefficient| > 1e-8', fontsize=12)
    fig.suptitle('News popularity: lasso baseline checkpoint', fontsize=17)
    fig.text(.03,.025,'Source: UCI Online News Popularity, Fernandes et al. (2015), CC BY 4.0. '
             'Derived analysis; log target and exclusions applied.\n'
             f'Checkpoint: {date(check_days[0])} to {date(check_days[-1])}; '
             f'N = {len(check):,}. Final 145 observed publication days remain untouched.\n'
             'Intervals reflect checkpoint sampling conditional on the fitted model; '
             'selection frequencies are a 20-refit pilot, not known feature truth.',fontsize=9)
    fig.tight_layout(rect=[0,.09,1,.96])
    fig.savefig(ROOT / 'baseline.png', dpi=170)
    plt.close(fig)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
