import numpy as np
import pytest

from src.core.features import (
    STANDARD_BANDS,
    ArtifactClipper,
    band_power,
    make_continuous_windows,
    make_recording_epochs,
    regional_summary,
)


def _sine_epochs(freq, sfreq, n_epochs=8, n_times=None, n_channels=2, seed=0):
    n_times = n_times or int(sfreq * 4)
    rng = np.random.default_rng(seed)
    t = np.arange(n_times) / sfreq
    base = np.sin(2 * np.pi * freq * t)
    return base + 0.01 * rng.standard_normal((n_epochs, n_channels, n_times))


def test_band_power_finds_the_right_band_at_128hz():
    epochs = _sine_epochs(10.0, 128)  # 10 Hz -> alpha
    x, names = band_power(epochs, 128, STANDARD_BANDS, ["C3", "C4"])
    assert names == [
        "C3_delta", "C3_theta", "C3_alpha", "C3_beta",
        "C4_delta", "C4_theta", "C4_alpha", "C4_beta",
    ]
    per_band = x[:, :4].mean(axis=0)
    assert np.argmax(per_band) == 2  # alpha wins


def test_same_signal_same_answer_at_a_different_sfreq():
    """The whole point of removing the module-level FS constant.

    A 10 Hz sine is a 10 Hz sine whether sampled at 128 or 500 Hz. The band
    ranking must not depend on the sampling rate.
    """
    for sfreq in (128, 160, 500):
        epochs = _sine_epochs(6.0, sfreq)  # 6 Hz -> theta
        x, _ = band_power(epochs, sfreq, STANDARD_BANDS, ["C3", "C4"])
        assert np.argmax(x[:, :4].mean(axis=0)) == 1


def test_wrong_sfreq_misplaces_the_peak():
    """Guards the regression the old hardcoded FS=128 would have caused."""
    epochs = _sine_epochs(6.0, 500)
    x_right, _ = band_power(epochs, 500, STANDARD_BANDS, ["C3", "C4"])
    x_wrong, _ = band_power(epochs, 128, STANDARD_BANDS, ["C3", "C4"])
    assert np.argmax(x_right[:, :4].mean(axis=0)) != np.argmax(
        x_wrong[:, :4].mean(axis=0)
    )


def test_missing_channel_raises_instead_of_silently_dropping():
    epochs = _sine_epochs(10.0, 128)
    with pytest.raises(KeyError):
        band_power(epochs, 128, STANDARD_BANDS, ["C3", "C4"],
                   selected_channels=["C3", "O1"])


def test_channel_name_count_must_match_montage():
    epochs = _sine_epochs(10.0, 128, n_channels=2)
    with pytest.raises(ValueError):
        band_power(epochs, 128, STANDARD_BANDS, ["C3", "C4", "CZ"])


def test_band_above_nyquist_raises():
    epochs = _sine_epochs(10.0, 50)
    with pytest.raises(ValueError):
        band_power(epochs, 50, {"beta": (13.0, 30.0)}, ["C3", "C4"])


def test_relative_power_sums_to_one_per_channel():
    epochs = _sine_epochs(10.0, 128)
    x, _ = band_power(epochs, 128, STANDARD_BANDS, ["C3", "C4"], relative=True)
    assert np.allclose(x[:, :4].sum(axis=1), 1.0)
    assert np.allclose(x[:, 4:].sum(axis=1), 1.0)


def test_selected_channel_order_defines_column_order():
    epochs = _sine_epochs(10.0, 128, n_channels=3)
    _, names = band_power(epochs, 128, {"alpha": (8.0, 13.0)},
                          ["C3", "C4", "CZ"], selected_channels=["CZ", "C3"])
    assert names == ["CZ_alpha", "C3_alpha"]


def test_continuous_windows_drop_mixed_labels():
    sfreq = 128
    x = np.zeros((sfreq * 4, 2))
    y = np.zeros(sfreq * 4, dtype=int)
    y[sfreq + 30:] = 1  # label flips mid-window
    epochs, labels, starts = make_continuous_windows(x, y, sfreq, win_sec=1.0)
    assert len(epochs) == 3  # the straddling window is dropped
    assert starts.tolist() == [0, 256, 384]
    assert labels.tolist() == [0, 1, 1]


def test_continuous_windows_are_non_overlapping():
    sfreq = 128
    x = np.zeros((sfreq * 5, 2))
    y = np.zeros(sfreq * 5, dtype=int)
    _, _, starts = make_continuous_windows(x, y, sfreq, win_sec=1.0)
    assert np.all(np.diff(starts) >= sfreq)


def test_recording_epochs_default_to_no_overlap():
    x = np.zeros((500 * 60, 3))
    epochs, labels, starts = make_recording_epochs(x, 500, label=1, epoch_sec=4.0)
    assert epochs.shape == (15, 3, 2000)
    assert set(labels) == {1}
    assert np.all(np.diff(starts) == 2000)


def test_recording_shorter_than_one_epoch_raises():
    with pytest.raises(ValueError):
        make_recording_epochs(np.zeros((100, 2)), 500, label=0, epoch_sec=4.0)


def test_clipper_uses_training_thresholds_on_unseen_data():
    rng = np.random.default_rng(0)
    train = rng.standard_normal((20, 2, 128))
    test = train.copy()
    test[0, 0, 0] = 500.0  # electrode pop in held-out data
    clipper = ArtifactClipper(lo_q=0.01, hi_q=0.99).fit(train)
    out = clipper.transform(test)
    assert out[0, 0, 0] < 10.0
    assert np.allclose(clipper.hi_, clipper.transform(train).max(axis=(0, 2)))


def test_clipper_must_be_fitted_first():
    with pytest.raises(RuntimeError):
        ArtifactClipper().transform(np.zeros((2, 2, 10)))


def test_regional_summary_marks_absent_regions_rather_than_imputing():
    names = ["F3_alpha", "F4_alpha", "C3_alpha"]
    x = np.array([[1.0, 3.0, 5.0]])
    out, out_names, coverage = regional_summary(x, names, {"alpha": (8.0, 13.0)})
    col = dict(zip(out_names, out[0]))
    assert col["frontal_alpha"] == 2.0          # mean of F3 and F4
    assert coverage["frontal_alpha"] == 2
    assert coverage["occipital_alpha"] == 0
    assert np.isnan(col["occipital_alpha"])     # absent, not zero
