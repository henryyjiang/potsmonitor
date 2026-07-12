#!/usr/bin/env python3
"""
Honest evaluation + hyperparameter sweep for the POTS flareup predictor.

Two things make the *old* numbers untrustworthy, both fixed here:

  1. Overlapping sliding windows (60 s wide, 10 s slide) share ~50 s of data
     with their neighbours. A random train/test split scatters near-duplicate
     windows across both sides, so the model is tested on rows it effectively
     trained on. We split by DAY instead (GroupKFold) so no day straddles the
     boundary. This also keeps each flareup episode intact on one side.

  2. Oversampling the minority class before splitting copies positives into
     both train and test. We oversample the TRAIN fold only.

The headline metric is F1 / recall on the flareup class under grouped CV — the
number that actually reflects "will this warn me before a real episode."
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")   # quiet SMOTE/LR numeric warnings
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, ParameterSampler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

FEATURES_CSV = Path("/Users/paff/Desktop/Projects/POTSMonitor/features.csv")
# Must stay in exact parity with FeatureEngine.swift / POTSPredictor.featureDict.
FEATURE_COLS = [
    "meanHR", "maxHR", "minHR", "hrDelta", "rmssd", "sdnn", "meanRR",
    "accMagMean", "accMagStd", "accVertDelta", "postureJerkPeak",
    "hrRiseFromBaseline", "rmssdPctChange", "hrSlope", "pNN50",
]
RANDOM_STATE = 42
N_ITER = 20
# Cap the SMOTE target so folds stay small enough to sweep quickly. A 1:4
# positive:negative training ratio is plenty of minority signal for trees.
OVERSAMPLE_TO = 8000


def load(path):
    df = pd.read_csv(path)
    X = df[FEATURE_COLS].values
    y = df["label"].values
    groups = df["date"].values
    return X, y, groups


def oversample_train(X, y):
    """SMOTE the training fold up to OVERSAMPLE_TO positives. Train-only."""
    pos = int((y == 1).sum())
    if pos < 2 or pos >= OVERSAMPLE_TO:
        return X, y
    k = min(5, pos - 1)
    sm = SMOTE(sampling_strategy={1: OVERSAMPLE_TO}, random_state=RANDOM_STATE, k_neighbors=k)
    return sm.fit_resample(X, y)


def grouped_cv_scores(make_model, X, y, groups, scale=False, n_splits=5):
    """Out-of-fold predictions under StratifiedGroupKFold; returns pooled metrics."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    y_true_all, y_pred_all, y_prob_all = [], [], []
    for tr, te in cv.split(X, y, groups):
        Xtr, ytr = oversample_train(X[tr], y[tr])
        Xte, yte = X[te], y[te]
        if scale:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        m = make_model()
        m.fit(Xtr, ytr)
        y_true_all.append(yte)
        y_pred_all.append(m.predict(Xte))
        if hasattr(m, "predict_proba"):
            y_prob_all.append(m.predict_proba(Xte)[:, 1])
    yt = np.concatenate(y_true_all)
    yp = np.concatenate(y_pred_all)
    prob = np.concatenate(y_prob_all) if y_prob_all else None
    return {
        "f1": f1_score(yt, yp, pos_label=1, zero_division=0),
        "precision": precision_score(yt, yp, pos_label=1, zero_division=0),
        "recall": recall_score(yt, yp, pos_label=1, zero_division=0),
        "auc": roc_auc_score(yt, prob) if prob is not None and len(np.unique(yt)) > 1 else float("nan"),
        "prob": prob, "yt": yt,
    }


def show(name, s):
    print(f"  {name:<34} F1={s['f1']:.3f}  P={s['precision']:.3f}  "
          f"R={s['recall']:.3f}  AUC={s['auc']:.3f}")


def leakage_demo(X, y, groups):
    """Same model, random split vs grouped split — shows the inflation."""
    from sklearn.model_selection import StratifiedKFold
    print(f"\n{'='*70}\n  Leakage demo: identical GBT, random split vs grouped-by-day\n{'='*70}")
    mk = lambda: GradientBoostingClassifier(max_depth=3, n_estimators=150,
                                            learning_rate=0.1, random_state=RANDOM_STATE)

    # Random split (the OLD, leaky way) — StratifiedKFold ignoring groups
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    yt_all, yp_all = [], []
    for tr, te in cv.split(X, y):
        Xtr, ytr = oversample_train(X[tr], y[tr])
        m = mk(); m.fit(Xtr, ytr)
        yt_all.append(y[te]); yp_all.append(m.predict(X[te]))
    yt, yp = np.concatenate(yt_all), np.concatenate(yp_all)
    print(f"  RANDOM split (leaky):   F1={f1_score(yt,yp,zero_division=0):.3f}  "
          f"P={precision_score(yt,yp,zero_division=0):.3f}  R={recall_score(yt,yp,zero_division=0):.3f}")

    s = grouped_cv_scores(mk, X, y, groups)
    print(f"  GROUPED split (honest): F1={s['f1']:.3f}  P={s['precision']:.3f}  R={s['recall']:.3f}")
    print("  → the gap between these two rows is pure leakage in the old metric.")


def sweep_gbt(X, y, groups):
    space = {
        "max_depth": [2, 3, 4],
        "n_estimators": [100, 150, 200, 300],
        "learning_rate": [0.03, 0.05, 0.1, 0.15],
        "min_samples_leaf": [5, 10, 20, 40],
        "subsample": [0.7, 0.8, 0.9],
    }
    best = None
    for params in ParameterSampler(space, n_iter=N_ITER, random_state=RANDOM_STATE):
        s = grouped_cv_scores(
            lambda p=params: GradientBoostingClassifier(random_state=RANDOM_STATE, **p),
            X, y, groups)
        if best is None or s["f1"] > best[1]["f1"]:
            best = (params, s)
    print(f"\n{'='*70}\n  Best Gradient Boosted Tree (grouped CV)\n{'='*70}")
    show("GBT (best)", best[1])
    print("  params:", best[0])
    return best


def feature_importance(model, names):
    if not hasattr(model, "feature_importances_"):
        return
    order = np.argsort(model.feature_importances_)[::-1]
    print("\n  Feature importances (final GBT on all data):")
    for i in order:
        imp = model.feature_importances_[i]
        print(f"    {names[i]:<22} {imp:.3f}  {'█' * int(imp * 40)}")


def main():
    X, y, groups = load(FEATURES_CSV)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    print(f"Loaded {len(y):,} windows — {pos} flareup / {neg} normal "
          f"across {len(np.unique(groups))} days ({np.unique(groups[y==1]).size} with flareups)")
    print("NOTE: only ~9 independent positive-days exist. Grouped-CV numbers are\n"
          "      noisy by nature; treat them as a realistic floor, not a point estimate.")

    leakage_demo(X, y, groups)

    print(f"\n{'='*70}\n  Model comparison (grouped-by-day CV, train-only oversampling)\n{'='*70}")
    show("Logistic Regression (balanced)",
         grouped_cv_scores(lambda: LogisticRegression(class_weight="balanced",
                           max_iter=1000, random_state=RANDOM_STATE), X, y, groups, scale=True))
    show("Random Forest (balanced)",
         grouped_cv_scores(lambda: RandomForestClassifier(n_estimators=300, max_depth=6,
                           min_samples_leaf=10, class_weight="balanced_subsample",
                           random_state=RANDOM_STATE), X, y, groups))
    show("GBT (shallow default)",
         grouped_cv_scores(lambda: GradientBoostingClassifier(max_depth=3, n_estimators=150,
                           learning_rate=0.1, random_state=RANDOM_STATE), X, y, groups))

    params, _ = sweep_gbt(X, y, groups)

    # Fit final GBT on all data for feature importances
    Xb, yb = oversample_train(X, y)
    final = GradientBoostingClassifier(random_state=RANDOM_STATE, **params).fit(Xb, yb)
    feature_importance(final, FEATURE_COLS)

    # CreateML params matched to the tuned sklearn GBT
    print(f"\n{'='*70}\n  Suggested train_model.swift params\n{'='*70}")
    print(f"""
  MLBoostedTreeClassifier.ModelParameters(
      maxDepth:         {params['max_depth']},
      maxIterations:    {params['n_estimators']},
      minLossReduction: 0.0,
      minChildWeight:   {float(params['min_samples_leaf'])},
      stepSize:         {params['learning_rate']}
  )
""")


if __name__ == "__main__":
    main()
