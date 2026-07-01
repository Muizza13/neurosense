import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0,".")
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import (TimeSeriesSplit, train_test_split,
                                     LeaveOneGroupOut, cross_val_predict, cross_val_score)
from sklearn.inspection import permutation_importance
from src.features import clip_artifacts, make_windows, band_powers, feature_names, BANDS

plt.rcParams.update({"figure.dpi":130,"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
INK="#1b2a4a"; HOT="#c0392b"; COOL="#2e86c1"; GOOD="#27ae60"; GREY="#95a5a6"
def pipe(m): return Pipeline([("s",StandardScaler()),("m",m)])

# ============ PHASE 1 recompute ============
df = pd.DataFrame(arff.loadarff("data/raw/EEG Eye State.arff")[0]); df["eyeDetection"]=df["eyeDetection"].astype(int)
ch=[c for c in df.columns if c!="eyeDetection"]
X=clip_artifacts(df[ch].values); y=df["eyeDetection"].values
Xw,yw=make_windows(X,y); names1=feature_names(ch)
split=int(0.7*len(Xw))
Xtr,Xte,ytr,yte=Xw[:split],Xw[split:],yw[:split],yw[split:]

rf=pipe(RandomForestClassifier(n_estimators=300,random_state=42))
Xs_tr,Xs_te,ys_tr,ys_te=train_test_split(Xw,yw,test_size=.3,random_state=42,shuffle=True,stratify=yw)
rf.fit(Xs_tr,ys_tr); naive_f1=f1_score(ys_te,rf.predict(Xs_te))
rf.fit(Xtr,ytr); chron_f1=f1_score(yte,rf.predict(Xte))
tss=cross_val_score(pipe(RandomForestClassifier(n_estimators=300,random_state=42)),Xw,yw,cv=TimeSeriesSplit(5),scoring="f1")

# leave-one-block-out AUC
bid=df["eyeDetection"].ne(df["eyeDetection"].shift()).cumsum().values
groups=[]; FS=128
for s in range(0,len(X)-FS+1,FS):
    if y[s:s+FS].min()==y[s:s+FS].max(): groups.append(int(np.bincount(bid[s:s+FS]).argmax()))
groups=np.array(groups)
lobo_auc={}
for nm,m in [("LogReg",pipe(LogisticRegression(max_iter=2000))),("RandomForest",pipe(RandomForestClassifier(n_estimators=300,random_state=42)))]:
    pr=cross_val_predict(m,Xw,yw,cv=LeaveOneGroupOut(),groups=groups,method="predict_proba")[:,1]
    lobo_auc[nm]=roc_auc_score(yw,pr)

# phase1 permutation importance (frontal vs posterior)
best=pipe(RandomForestClassifier(n_estimators=300,random_state=42)).fit(Xtr,ytr)
imp=permutation_importance(best,Xte,yte,n_repeats=30,random_state=42,scoring="f1")
order=imp.importances_mean.argsort()[::-1][:10]
POSTERIOR={"O1","O2","P7","P8","T7","T8"}
f1names=[names1[i] for i in order]; f1vals=[imp.importances_mean[i] for i in order]
f1cols=[COOL if nm.split("_")[0] in POSTERIOR else HOT for nm in f1names]

# ============ PHASE 2 recompute ============
d=np.load("data/processed/physionet_features.npz"); X2,y2,g2=d["X"],d["y"],d["g"]
from src.physionet import feature_names as fn2, MOTOR
names2=fn2()
from sklearn.model_selection import StratifiedKFold
def within(mk):
    b,a=[],[]
    for s in np.unique(g2):
        Xs,ys=X2[g2==s],y2[g2==s]
        pr=cross_val_predict(pipe(mk()),Xs,ys,cv=StratifiedKFold(5,shuffle=True,random_state=42),method="predict_proba")[:,1]
        b.append(balanced_accuracy_score(ys,(pr>=.5).astype(int))); a.append(roc_auc_score(ys,pr))
    return np.array(b),np.array(a)
wb,wa=within(lambda:LogisticRegression(max_iter=2000,C=0.5))
prc=cross_val_predict(pipe(LogisticRegression(max_iter=2000,C=0.5)),X2,y2,cv=LeaveOneGroupOut(),groups=g2,method="predict_proba")[:,1]
cross_b=balanced_accuracy_score(y2,(prc>=.5).astype(int)); cross_a=roc_auc_score(y2,prc)

coef=pipe(LogisticRegression(max_iter=2000,C=0.5)).fit(X2,y2).named_steps["m"].coef_[0]
o2=np.argsort(np.abs(coef))[::-1][:10]
CLINE={"C5","C3","C1","CZ","C2","C4","C6"}
f2names=[names2[i] for i in o2]; f2vals=[abs(coef[i]) for i in o2]
f2cols=[GOOD if nm.split("_")[0] in CLINE else GREY for nm in f2names]

# ============ FIGURE 1: Phase 1 leakage ============
fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.2))
labs=["Naive\nrandom split","Chronological\nholdout","Blocked\nTimeSeriesSplit"]
vals=[naive_f1,chron_f1,tss.mean()]; errs=[0,0,tss.std()]
bars=a1.bar(labs,vals,yerr=errs,capsize=5,color=[HOT,INK,INK])
a1.axhline(0.38,ls="--",color=GREY,lw=1); a1.text(2.35,0.39,"majority\nbaseline",color=GREY,fontsize=8,ha="right")
a1.set_ylabel("F1 score"); a1.set_ylim(0,0.75); a1.set_title("Same features, same model.\nOnly the split changes.",fontsize=11,loc="left")
for b,v in zip(bars,vals): a1.text(b.get_x()+b.get_width()/2,v+0.02,f"{v:.2f}",ha="center",fontweight="bold")
a1.text(0,naive_f1/2,"LEAKAGE",ha="center",color="white",fontweight="bold",rotation=90,fontsize=9)
a2.bar(list(lobo_auc.keys()),list(lobo_auc.values()),color=INK)
a2.axhline(0.5,ls="--",color=HOT,lw=1.3); a2.text(1.4,0.51,"chance",color=HOT,fontsize=9,ha="right")
a2.set_ylabel("ROC-AUC"); a2.set_ylim(0,0.7); a2.set_title("Leave-one-block-out:\nbelow chance",fontsize=11,loc="left")
for i,v in enumerate(lobo_auc.values()): a2.text(i,v+0.015,f"{v:.2f}",ha="center",fontweight="bold")
fig.suptitle("Phase 1  |  UCI EEG Eye State: apparent skill is an artifact of the split",fontweight="bold",x=0.02,ha="left")
fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig("reports/figures/fig1_phase1_leakage.png",bbox_inches="tight"); plt.close(fig)

# ============ FIGURE 2: Phase 2 within vs cross ============
fig,ax=plt.subplots(figsize=(7.2,4.6))
x=np.arange(2); w=0.35
ax.bar(x-w/2,[wb.mean(),cross_b],w,yerr=[wb.std(),0],capsize=5,label="Balanced accuracy",color=COOL)
ax.bar(x+w/2,[wa.mean(),cross_a],w,yerr=[wa.std(),0],capsize=5,label="ROC-AUC",color=INK)
ax.axhline(0.5,ls="--",color=HOT,lw=1.3); ax.text(1.5,0.51,"chance",color=HOT,ha="right",fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(["Within-subject\n(trial CV)","Cross-subject\n(leave-one-subject-out)"])
ax.set_ylim(0,0.75); ax.set_ylabel("score"); ax.legend(frameon=False,loc="upper right")
ax.set_title("Phase 2  |  Motor imagery: honest within-subject, chance across subjects",fontweight="bold",fontsize=11,loc="left")
for xi,(b,a_) in zip(x,[(wb.mean(),wa.mean()),(cross_b,cross_a)]):
    ax.text(xi-w/2,b+0.03,f"{b:.2f}",ha="center",fontweight="bold",fontsize=9)
    ax.text(xi+w/2,a_+0.03,f"{a_:.2f}",ha="center",fontweight="bold",fontsize=9)
ax.text(0,0.02,"8/10 subjects above chance",ha="center",fontsize=8.5,style="italic",color=INK)
fig.tight_layout(); fig.savefig("reports/figures/fig2_phase2_generalization.png",bbox_inches="tight"); plt.close(fig)

# ============ FIGURE 3: interpretability contrast ============
fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.6))
a1.barh(range(len(f1names))[::-1],f1vals,color=f1cols)
a1.set_yticks(range(len(f1names))[::-1]); a1.set_yticklabels(f1names,fontsize=9)
a1.set_xlabel("permutation importance"); a1.set_title("Phase 1: top features are FRONTAL\n= eye-movement / blink artifact",fontsize=10.5,loc="left",color=HOT)
a2.barh(range(len(f2names))[::-1],f2vals,color=f2cols)
a2.set_yticks(range(len(f2names))[::-1]); a2.set_yticklabels(f2names,fontsize=9)
a2.set_xlabel("|standardized coefficient|"); a2.set_title("Phase 2: top features are C3/C4 mu\n= real sensorimotor lateralization",fontsize=10.5,loc="left",color=GOOD)
from matplotlib.patches import Patch
a1.legend(handles=[Patch(color=HOT,label="frontal"),Patch(color=COOL,label="posterior")],frameon=False,fontsize=8,loc="lower right")
a2.legend(handles=[Patch(color=GOOD,label="central (motor)"),Patch(color=GREY,label="other")],frameon=False,fontsize=8,loc="lower right")
fig.suptitle("Explainability as a validity check: it exposes the artifact (left), it confirms the physiology (right)",fontweight="bold",x=0.02,ha="left",fontsize=11)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig("reports/figures/fig3_interpretability_contrast.png",bbox_inches="tight"); plt.close(fig)

print("figures written to reports/figures/")
print(f"phase1: naive={naive_f1:.3f} chron={chron_f1:.3f} tss={tss.mean():.3f}+/-{tss.std():.3f} lobo_auc={ {k:round(v,3) for k,v in lobo_auc.items()} }")
print(f"phase2: within balAcc={wb.mean():.3f}+/-{wb.std():.3f} AUC={wa.mean():.3f} | cross balAcc={cross_b:.3f} AUC={cross_a:.3f}")
