"""Shared windowing and spectral feature extraction for all NeuroSense phases.

Design rules enforced here:

1. No module-level sampling frequency. Every caller passes ``sfreq`` explicitly.
   Phase 1 is 128 Hz, Phase 2 is 160 Hz, Phase 3 (EEGMAT) is 500 Hz, and a
   silently wrong constant would corrupt every band boundary downstream.

2. Windowing is separate from feature extraction. The three phases have three
   different label structures (per-sample, per-trial, per-recording), so window
   construction is phase-specific while the spectral estimator is shared.

3. Artifact clipping is a fitted transformer. Quantiles are learned on training
   data only and applied to held-out data. Computing them over a whole recording
   leaks the distribution of future samples into past ones.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

# Standard band definitions used for the harmonised (Track B) comparison.
# Delta starts at 1 Hz, not 0.5 Hz: with a 1-second Welch segment the frequency
# resolution is 1 Hz, so a 0.5 Hz edge is not resolvable.
STANDARD_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

# Phase 2 native (Track A) bands for sensorimotor rhythms.
MOTOR_BANDS = {
    "mu": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

_integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------
def make_continuous_windows(x, y_per_sample, sfreq, win_sec=1.0, drop_mixed=True):
    """Split a continuous recording with per-sample labels into windows.

    Used by Phase 1, where the eye-state label changes partway through the
    recording. Windows are non-overlapping, so a window-level split is a split
    over disjoint blocks of time.

    Parameters
    ----------
    x : ndarray, shape (n_samples, n_channels)
    y_per_sample : ndarray, shape (n_samples,)
    sfreq : float
    win_sec : float
    drop_mixed : bool
        Discard windows that straddle a label change.

    Returns
    -------
    epochs : ndarray, shape (n_windows, n_channels, n_times)
    labels : ndarray, shape (n_windows,)
    starts : ndarray, shape (n_windows,)
        Start sample index of each window, retained so that chronological and
        block-wise splits remain possible after windowing.
    """
    x = np.asarray(x, dtype=float)
    y_per_sample = np.asarray(y_per_sample)
    win = int(round(win_sec * sfreq))
    if win <= 0:
        raise ValueError("win_sec * sfreq must be at least 1 sample")

    epochs, labels, starts = [], [], []
    for start in range(0, len(x) - win + 1, win):
        seg_y = y_per_sample[start:start + win]
        if drop_mixed and seg_y.min() != seg_y.max():
            continue
        epochs.append(x[start:start + win].T)  # -> (n_channels, n_times)
        labels.append(int(seg_y[0]))
        starts.append(start)

    if not epochs:
        raise ValueError("no windows produced; check win_sec and label array")
    return np.stack(epochs), np.asarray(labels), np.asarray(starts)


def make_recording_epochs(x, sfreq, label, epoch_sec=4.0, overlap=0.0):
    """Split a single-condition recording into fixed-length epochs.

    Used by Phase 3, where the condition label applies to the whole recording
    (one rest file, one arithmetic file per subject).

    Returns
    -------
    epochs : ndarray, shape (n_epochs, n_channels, n_times)
    labels : ndarray, shape (n_epochs,)
        Constant, equal to ``label``.
    starts : ndarray, shape (n_epochs,)
        Start sample index, retained for the early-versus-late drift control.
    """
    x = np.asarray(x, dtype=float)
    win = int(round(epoch_sec * sfreq))
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    step = max(1, int(round(win * (1.0 - overlap))))

    epochs, starts = [], []
    for start in range(0, len(x) - win + 1, step):
        epochs.append(x[start:start + win].T)
        starts.append(start)

    if not epochs:
        raise ValueError("recording shorter than one epoch")
    return (
        np.stack(epochs),
        np.full(len(epochs), int(label)),
        np.asarray(starts),
    )


# --------------------------------------------------------------------------
# Artifact clipping (fitted, fold-safe)
# --------------------------------------------------------------------------
class ArtifactClipper:
    """Per-channel winsorisation with thresholds fitted on training data only.

    Either fit the quantiles on a training split, or pass fixed absolute limits
    prespecified before seeing the data. What is not allowed is computing
    quantiles over a full recording that later supplies held-out samples.
    """

    def __init__(self, lo_q=0.001, hi_q=0.999, fixed_limits=None):
        self.lo_q = lo_q
        self.hi_q = hi_q
        self.fixed_limits = fixed_limits
        self.lo_ = None
        self.hi_ = None

    def fit(self, epochs):
        """epochs: (n_epochs, n_channels, n_times)"""
        if self.fixed_limits is not None:
            lo, hi = self.fixed_limits
            n_ch = epochs.shape[1]
            self.lo_ = np.full(n_ch, float(lo))
            self.hi_ = np.full(n_ch, float(hi))
            return self
        flat = np.moveaxis(epochs, 1, 0).reshape(epochs.shape[1], -1)
        self.lo_ = np.quantile(flat, self.lo_q, axis=1)
        self.hi_ = np.quantile(flat, self.hi_q, axis=1)
        return self

    def transform(self, epochs):
        if self.lo_ is None:
            raise RuntimeError("ArtifactClipper must be fitted before transform")
        lo = self.lo_[None, :, None]
        hi = self.hi_[None, :, None]
        return np.clip(np.asarray(epochs, dtype=float), lo, hi)

    def fit_transform(self, epochs):
        return self.fit(epochs).transform(epochs)


# --------------------------------------------------------------------------
# Spectral features
# --------------------------------------------------------------------------
def band_power(
    epochs,
    sfreq,
    bands,
    channel_names,
    selected_channels=None,
    relative=False,
    nperseg_sec=1.0,
):
    """Welch band power per channel per epoch.

    Parameters
    ----------
    epochs : ndarray, shape (n_epochs, n_channels, n_times)
    sfreq : float
        Sampling frequency in Hz. Required, never assumed.
    bands : dict
        Ordered mapping of band name to (low, high) in Hz, high exclusive.
    channel_names : sequence of str
        Names for all channels present in ``epochs``, in order.
    selected_channels : sequence of str, optional
        Subset to keep. Order of this list defines output column order.
        A name not present in ``channel_names`` raises, rather than being
        silently dropped or imputed.
    relative : bool
        If True, divide each band by the summed power across all requested
        bands within the same channel and epoch.
    nperseg_sec : float
        Welch segment length in seconds. Sets frequency resolution to
        1 / nperseg_sec Hz. The default of 1.0 s gives 1 Hz resolution, which
        is why the lowest band edge is 1 Hz rather than 0.5 Hz.

    Returns
    -------
    X : ndarray, shape (n_epochs, n_selected_channels * n_bands)
    names : list of str
        Column names as "<channel>_<band>".
    """
    epochs = np.asarray(epochs, dtype=float)
    if epochs.ndim != 3:
        raise ValueError("epochs must be (n_epochs, n_channels, n_times)")
    if len(channel_names) != epochs.shape[1]:
        raise ValueError(
            f"channel_names has {len(channel_names)} entries but epochs has "
            f"{epochs.shape[1]} channels"
        )

    if selected_channels is None:
        idx = list(range(epochs.shape[1]))
        used = list(channel_names)
    else:
        lookup = {name: i for i, name in enumerate(channel_names)}
        missing = [c for c in selected_channels if c not in lookup]
        if missing:
            raise KeyError(f"channels not present in this montage: {missing}")
        idx = [lookup[c] for c in selected_channels]
        used = list(selected_channels)

    nperseg = min(epochs.shape[2], int(round(nperseg_sec * sfreq)))
    sel = epochs[:, idx, :]

    freqs, psd = welch(sel, fs=sfreq, nperseg=nperseg, axis=-1)
    # psd: (n_epochs, n_channels, n_freqs)

    nyquist = sfreq / 2.0
    powers = []
    for lo, hi in bands.values():
        if hi > nyquist:
            raise ValueError(
                f"band edge {hi} Hz exceeds Nyquist ({nyquist} Hz) at "
                f"sfreq={sfreq}"
            )
        mask = (freqs >= lo) & (freqs < hi)
        if not mask.any():
            raise ValueError(
                f"no FFT bins fall in band ({lo}, {hi}) Hz at sfreq={sfreq} "
                f"with nperseg_sec={nperseg_sec}"
            )
        powers.append(_integrate(psd[:, :, mask], freqs[mask], axis=-1))

    stacked = np.stack(powers, axis=-1)  # (n_epochs, n_channels, n_bands)

    if relative:
        total = stacked.sum(axis=-1, keepdims=True)
        total = np.where(total <= 0, np.nan, total)
        stacked = stacked / total

    x = stacked.reshape(stacked.shape[0], -1)
    names = [f"{ch}_{band}" for ch in used for band in bands]
    return x, names


class EpochBandPower:
    """Sklearn-compatible transformer: raw epochs in, band power out.

    Exists so that artifact clipping and spectral extraction sit *inside* a
    Pipeline, and therefore inside the cross-validation fold. Anything fitted
    here (clipping quantiles) is learned from training epochs only.

    Input X is flattened epochs, shape (n_epochs, n_channels * n_times), because
    scikit-learn requires 2-D input. The original shape is restored internally.
    """

    def __init__(self, sfreq, bands, channel_names, n_times,
                 selected_channels=None, relative=False, nperseg_sec=1.0,
                 clip=True, lo_q=0.001, hi_q=0.999):
        self.sfreq = sfreq
        self.bands = bands
        self.channel_names = channel_names  # stored unmodified: sklearn clone requires it
        self.n_times = n_times
        self.selected_channels = selected_channels
        self.relative = relative
        self.nperseg_sec = nperseg_sec
        self.clip = clip
        self.lo_q = lo_q
        self.hi_q = hi_q
        self.clipper_ = None
        self.feature_names_ = None

    def get_params(self, deep=True):
        return {k: getattr(self, k) for k in (
            "sfreq", "bands", "channel_names", "n_times", "selected_channels",
            "relative", "nperseg_sec", "clip", "lo_q", "hi_q")}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _unflatten(self, x):
        return np.asarray(x, dtype=float).reshape(
            len(x), len(list(self.channel_names)), self.n_times
        )

    def fit(self, X, y=None):
        epochs = self._unflatten(X)
        if self.clip:
            self.clipper_ = ArtifactClipper(self.lo_q, self.hi_q).fit(epochs)
        return self

    def transform(self, X):
        epochs = self._unflatten(X)
        if self.clip:
            if self.clipper_ is None:
                raise RuntimeError("EpochBandPower must be fitted first")
            epochs = self.clipper_.transform(epochs)
        feats, names = band_power(
            epochs, self.sfreq, self.bands, self.channel_names,
            selected_channels=self.selected_channels,
            relative=self.relative, nperseg_sec=self.nperseg_sec,
        )
        self.feature_names_ = names
        return feats

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def flatten_epochs(epochs):
    """(n_epochs, n_channels, n_times) -> (n_epochs, n_channels * n_times)."""
    epochs = np.asarray(epochs, dtype=float)
    return epochs.reshape(len(epochs), -1)


def label_blocks(y_per_sample):
    """Contiguous runs of a constant label, as a block id per sample.

    Phase 1 has a single subject, so the grouping unit for leakage-aware
    evaluation is the contiguous label block rather than the person.
    """
    y_per_sample = np.asarray(y_per_sample)
    change = np.concatenate([[0], np.diff(y_per_sample) != 0])
    return np.cumsum(change)


REGIONS = {
    "frontal": ("FP1", "FP2", "AF3", "AF4", "F3", "F4", "F7", "F8", "FZ"),
    "central": ("C3", "C4", "CZ", "FC5", "FC6"),
    "temporal": ("T7", "T8", "T9", "T10", "FT7", "FT8", "TP7", "TP8"),
    "parietal": ("P3", "P4", "PZ", "P7", "P8", "CP3", "CP4"),
    "occipital": ("O1", "O2", "OZ"),
}


def regional_summary(x, names, bands, regions=REGIONS):
    """Average band power within scalp regions, for the Track B comparison.

    Returns the mean over whichever electrodes of a region the montage actually
    contains, plus a coverage dict recording how many electrodes contributed.
    A region with zero available electrodes yields NaN and coverage 0, so it is
    visibly absent rather than silently imputed to equivalence.
    """
    col = {n: i for i, n in enumerate(names)}
    out, out_names, coverage = [], [], {}
    for region, electrodes in regions.items():
        for band in bands:
            present = [e for e in electrodes if f"{e}_{band}" in col]
            coverage[f"{region}_{band}"] = len(present)
            if present:
                out.append(x[:, [col[f"{e}_{band}"] for e in present]].mean(axis=1))
            else:
                out.append(np.full(x.shape[0], np.nan))
            out_names.append(f"{region}_{band}")
    return np.stack(out, axis=1), out_names, coverage
