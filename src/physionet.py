"""Phase 2: PhysioNet motor-imagery (left vs right fist), leakage-aware evaluation.

Mirrors the Phase 1 methodology on data that can support honest generalization:
- discrete imagery trials (T1=left, T2=right) from runs R04/R08/R12
- mu (8-13 Hz) and beta (13-30 Hz) band power over a sensorimotor channel strip
- within-subject (trial-level CV) AND cross-subject (leave-one-subject-out)
"""
import warnings; warnings.filterwarnings("ignore")
import glob, numpy as np, mne
from scipy.signal import welch

FS = 160
BANDS = {"mu": (8, 13), "beta": (13, 30)}
MOTOR = ["FC3","FCZ","FC4","C5","C3","C1","CZ","C2","C4","C6","CP3","CPZ","CP4"]
_integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

def _clean(name): return name.replace(".", "").upper()

def load_subject(subj_dir, tmin=0.5, tmax=3.5):
    raws = []
    for f in sorted(glob.glob(f"{subj_dir}/*R*.edf")):
        r = mne.io.read_raw_edf(f, preload=True, verbose=False)
        r.rename_channels({c: _clean(c) for c in r.ch_names})
        raws.append(r)
    raw = mne.concatenate_raws(raws, verbose=False)
    raw.pick(MOTOR)
    events, eid = mne.events_from_annotations(raw, verbose=False)
    # T1/T2 are codes 2/3 (T0=1). Map to left=0, right=1.
    want = {k: v for k, v in eid.items() if k in ("T1", "T2")}
    ep = mne.Epochs(raw, events, event_id=want, tmin=tmin, tmax=tmax,
                    baseline=None, preload=True, verbose=False)
    y = (ep.events[:, 2] == want["T2"]).astype(int)  # 1 = right fist
    return ep.get_data(), y  # (n_epochs, n_ch, n_times), labels

def epoch_features(data):
    feats = []
    for ep in data:                      # ep: (n_ch, n_times)
        row = []
        for ch in range(ep.shape[0]):
            fr, psd = welch(ep[ch], fs=FS, nperseg=min(FS, ep.shape[1]))
            for lo, hi in BANDS.values():
                m = (fr >= lo) & (fr < hi)
                row.append(_integrate(psd[m], fr[m]))
        feats.append(row)
    return np.array(feats)

def feature_names():
    return [f"{ch}_{b}" for ch in MOTOR for b in BANDS]

def load_all(root="data/physionet"):
    X, y, g = [], [], []
    for i, d in enumerate(sorted(glob.glob(f"{root}/S0*"))):
        data, yy = load_subject(d)
        ff = epoch_features(data)
        X.append(ff); y.append(yy); g.append(np.full(len(yy), i))
        print(f"  {d.split('/')[-1]}: {len(yy)} trials (left {int((yy==0).sum())}/right {int((yy==1).sum())})")
    return np.vstack(X), np.concatenate(y), np.concatenate(g)
