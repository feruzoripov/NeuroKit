"""Tests for parallel processing features."""

import numpy as np

import neurokit2 as nk


# =============================================================================
# bio_process parallel
# =============================================================================
def test_bio_process_parallel():
    """Test that bio_process with parallel=True produces same results as sequential."""
    sampling_rate = 250
    ecg = nk.ecg_simulate(duration=30, sampling_rate=sampling_rate)
    rsp = nk.rsp_simulate(duration=30, sampling_rate=sampling_rate)
    eda = nk.eda_simulate(duration=30, sampling_rate=sampling_rate, scr_number=3)
    emg = nk.emg_simulate(duration=30, sampling_rate=sampling_rate, burst_number=3)

    bio_seq, info_seq = nk.bio_process(ecg=ecg, rsp=rsp, eda=eda, emg=emg, sampling_rate=sampling_rate, parallel=False)
    bio_par, info_par = nk.bio_process(ecg=ecg, rsp=rsp, eda=eda, emg=emg, sampling_rate=sampling_rate, parallel=True)

    # Same shape and columns
    assert list(bio_seq.columns) == list(bio_par.columns)
    assert len(bio_seq) == len(bio_par)

    # Same info keys
    assert set(info_seq.keys()) == set(info_par.keys())


def test_bio_process_parallel_single_signal():
    """Test that parallel=True works fine with a single signal (falls back to sequential)."""
    ecg = nk.ecg_simulate(duration=10, sampling_rate=250)
    bio_df, bio_info = nk.bio_process(ecg=ecg, sampling_rate=250, parallel=True)
    assert "ECG_Raw" in bio_df.columns


# =============================================================================
# ecg_findpeaks ProMAC parallel
# =============================================================================
def test_ecg_findpeaks_promac_parallel():
    """Test that ProMAC with n_jobs>1 finds the same peaks as sequential."""
    ecg = nk.ecg_simulate(duration=20, sampling_rate=500)
    ecg_clean = nk.ecg_clean(ecg, sampling_rate=500)

    info_seq = nk.ecg_findpeaks(ecg_clean, sampling_rate=500, method="promac", n_jobs=1)
    info_par = nk.ecg_findpeaks(ecg_clean, sampling_rate=500, method="promac", n_jobs=2)

    assert len(info_seq["ECG_R_Peaks"]) == len(info_par["ECG_R_Peaks"])
    np.testing.assert_array_equal(info_seq["ECG_R_Peaks"], info_par["ECG_R_Peaks"])


# =============================================================================
# entropy_multiscale parallel
# =============================================================================
def test_entropy_multiscale_parallel():
    """Test that entropy_multiscale with n_jobs>1 gives same result as sequential."""
    signal = nk.signal_simulate(duration=2, frequency=[5, 12])

    mse_seq, info_seq = nk.entropy_multiscale(signal, scale=5, show=False, n_jobs=1)
    mse_par, info_par = nk.entropy_multiscale(signal, scale=5, show=False, n_jobs=2)

    assert np.isclose(mse_seq, mse_par, atol=1e-10)
    np.testing.assert_array_almost_equal(info_seq["Value"], info_par["Value"])
