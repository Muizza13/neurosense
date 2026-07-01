"""Leakage-aware training + evaluation for EEG eye-state detection."""
import sys; sys.path.insert(0, ".")
from scipy.io import arff
import pandas as pd, numpy as np, json, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.inspection import permutation_importance
from src.features import clip_artifacts, make_windows, feature_names

RNG = 42

def load_features():
    data = arff.loadarff("data/raw/EEG Eye State.arff")
    df = pd.DataFrame(data[0]); df["eyeDetection"] = df["eyeDetection"].astype(int)
    ch = [c for c in df.columns if c != "eyeDetection"]
    X = clip_artifacts(df[ch].values); y = df["eyeDetection"].values
    Xw, yw = make_windows(X, y)
    return Xw, yw, feature_names(ch), ch

def pipe(model):
    return Pipeline([("scaler", StandardScaler()), ("model", model)])

def main():
    Xw, yw, names, ch = load_features()
    n = len(Xw); split = int(0.70 * n)
    Xtr, Xte, ytr, yte = Xw[:split], Xw[split:], yw[:split], yw[split:]
    print(f"windows={n}  train={len(ytr)} (open {int((ytr==0).sum())}/closed {int((ytr==1).sum())})"
          f"  test={len(yte)} (open {int((yte==0).sum())}/closed {int((yte==1).sum())})\n")

    models = {
        "LogReg": pipe(LogisticRegression(max_iter=2000, random_state=RNG)),
        "SVM-RBF": pipe(SVC(kernel="rbf", random_state=RNG)),
        "RandomForest": pipe(RandomForestClassifier(n_estimators=300, random_state=RNG)),
        "XGBoost": pipe(XGBClassifier(eval_metric="logloss", random_state=RNG, verbosity=0)),
    }

    # 1) Chronological holdout
    print("=== Chronological 70/30 holdout ===")
    print(f"{'model':14s}{'acc':>7s}{'prec':>7s}{'rec':>7s}{'f1':>7s}")
    holdout = {}
    for nm, m in models.items():
        m.fit(Xtr, ytr); p = m.predict(Xte)
        holdout[nm] = dict(
            acc=accuracy_score(yte,p), prec=precision_score(yte,p,zero_division=0),
            rec=recall_score(yte,p,zero_division=0), f1=f1_score(yte,p,zero_division=0))
        h = holdout[nm]
        print(f"{nm:14s}{h['acc']:7.3f}{h['prec']:7.3f}{h['rec']:7.3f}{h['f1']:7.3f}")

    # 2) Blocked TimeSeriesSplit (the leakage-aware headline)
    print("\n=== Blocked TimeSeriesSplit (5-fold) F1: mean +/- std ===")
    tscv = TimeSeriesSplit(n_splits=5)
    tss = {}
    for nm, m in models.items():
        sc = cross_val_score(m, Xw, yw, cv=tscv, scoring="f1")
        tss[nm] = (sc.mean(), sc.std())
        print(f"{nm:14s} F1 = {sc.mean():.3f} +/- {sc.std():.3f}")

    # 3) Leakage demo: SAME features, only the split changes (RandomForest)
    print("\n=== Leakage demo (RandomForest, identical features) ===")
    rf = pipe(RandomForestClassifier(n_estimators=300, random_state=RNG))
    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(Xw, yw, test_size=0.30,
                                                   random_state=RNG, shuffle=True, stratify=yw)
    rf.fit(Xs_tr, ys_tr); naive_f1 = f1_score(ys_te, rf.predict(Xs_te))
    print(f"naive shuffled split F1   = {naive_f1:.3f}")
    print(f"chronological holdout F1  = {holdout['RandomForest']['f1']:.3f}")
    print(f"blocked TimeSeriesSplit F1= {tss['RandomForest'][0]:.3f} +/- {tss['RandomForest'][1]:.3f}")

    # 4) Feature importance (permutation) on the chronological holdout
    best_name = max(tss, key=lambda k: tss[k][0])
    print(f"\n=== Permutation importance: {best_name} (best by TimeSeriesSplit) ===")
    best = models[best_name]
    imp = permutation_importance(best, Xte, yte, n_repeats=30, random_state=RNG, scoring="f1")
    order = imp.importances_mean.argsort()[::-1][:10]
    for i in order:
        print(f"  {names[i]:12s} {imp.importances_mean[i]:+.4f} +/- {imp.importances_std[i]:.4f}")

    # save artifacts + results
    best.fit(Xw, yw)
    joblib.dump(best, "models/model.pkl")
    json.dump({
        "best_model": best_name,
        "holdout": holdout,
        "timeseriessplit_f1": {k:{"mean":v[0],"std":v[1]} for k,v in tss.items()},
        "naive_shuffled_f1_rf": naive_f1,
        "top_features": [names[i] for i in order],
    }, open("reports/results.json","w"), indent=2)
    print("\nsaved models/model.pkl and reports/results.json")

if __name__ == "__main__":
    main()
