"""Windowing + band-power feature extraction for EEG eye-state detection.

Design choices (all deliberate, see project writeup):
- delta/theta/alpha/beta only. No gamma: on consumer EMOTIV EPOC hardware the
  30-45 Hz range is dominated by EMG (muscle), not cortical activity.
- 1-second non-overlapping windows. Non-overlapping means window-then-split is
  leak-free, because each window is a disjoint block of time.
- Per-channel artifact clipping (winsorize) before feature extraction.
- Mixed-label windows (straddling a state change) are dropped to keep labels clean.
"""
import numpy as np
from scipy.signal import welch

FS = 128  # EMOTIV EPOC sampling rate (Hz)
BANDS = {
    "delta": (1, 4),     # start at 1 Hz: 128-sample window => ~1 Hz resolution
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
}

# numpy 2.x renamed trapz -> trapezoid
_integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def clip_artifacts(X, lo_q=0.001, hi_q=0.999):
    """Winsorize each channel to tame single-sample electrode pops."""
    X = X.copy().astype(float)
    for j in range(X.shape[1]):
        lo, hi = np.quantile(X[:, j], [lo_q, hi_q])
        X[:, j] = np.clip(X[:, j], lo, hi)
    return X


def band_powers(segment, fs=FS):
    """segment: (n_samples, n_channels) -> (n_channels * n_bands,) band power."""
    nperseg = min(fs, segment.shape[0])
    freqs, psd = welch(segment, fs=fs, nperseg=nperseg, axis=0)
    feats = []
    for ch in range(segment.shape[1]):
        for (lo, hi) in BANDS.values():
            mask = (freqs >= lo) & (freqs < hi)
            feats.append(_integrate(psd[mask, ch], freqs[mask]))
    return np.array(feats)


def make_windows(X_raw, y_raw, fs=FS, win_sec=1.0):
    """Non-overlapping windows. Drops windows that straddle a label change."""
    win = int(win_sec * fs)
    feats, labels = [], []
    for start in range(0, len(X_raw) - win + 1, win):
        seg_y = y_raw[start:start + win]
        if seg_y.min() == seg_y.max():            # pure-label window only
            feats.append(band_powers(X_raw[start:start + win], fs))
            labels.append(int(seg_y[0]))
    return np.array(feats), np.array(labels)


def feature_names(channels):
    return [f"{ch}_{band}" for ch in channels for band in BANDS]
