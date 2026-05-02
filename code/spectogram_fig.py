#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# ─── your existing paths ─────────────────────────────────────────────────
sequences_directory = "/home/ubuntu/dds_paper/DDS_Paper/data_revision/csvs/sequences"
file_name = "kaggle_30_0_train_scaled_sequences.npy"
signal_path = os.path.join(sequences_directory, file_name)
# ────────────────────────────────────────────────────────────────────────────

def load_all_signals(path: str) -> np.ndarray:
    """
    Load a NumPy array of shape (N_seqs, N_chans, seq_len) or (N_seqs, seq_len).
    Returns an array of shape (N_seqs, N_chans, seq_len).
    """
    data = np.load(path)
    if data.ndim == 2:
        # assume (N_seqs, seq_len) → make channel dimension = 1
        data = data[:, np.newaxis, :]
    return data  # shape (N_seqs, N_chans, seq_len)

def choose_best_segment(data: np.ndarray) -> (int, int):
    """
    Pick the (seq_idx, chan_idx) with the highest variance over time.
    """
    # compute variance along the time axis
    # result shape: (N_seqs, N_chans)
    varmap = data.var(axis=2)
    seq_idx, chan_idx = np.unravel_index(np.argmax(varmap), varmap.shape)
    return int(seq_idx), int(chan_idx)

def choose_window_params(sig_len: int, max_nperseg: int = 256) -> (int, int):
    cap = min(sig_len, max_nperseg)
    p = 2 ** int(np.floor(np.log2(cap)))
    nperseg = p if p >= 2 else sig_len
    noverlap = nperseg // 2
    return nperseg, noverlap

def plot_signal_spectrogram(
    signal: np.ndarray,
    fs: float = 20000.0,
    cmap: str = 'viridis',
    save_path: str = None
):
    L = signal.shape[0]
    nperseg, noverlap = choose_window_params(L)
    print(f"[Debug] plotting seq_len={L}, nperseg={nperseg}, noverlap={noverlap}")

    f, t, Sxx = spectrogram(
        signal, fs=fs,
        nperseg=nperseg, noverlap=noverlap,
        scaling='density', mode='psd'
    )
    print(f"[Debug] spectrogram grid → freqs: {len(f)}, times: {len(t)}")

    plt.figure(figsize=(6, 4))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12),
                   shading='auto', cmap=cmap)
    plt.colorbar(label='Power (dB)')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(f'Spectrogram (nperseg={nperseg}, noverlap={noverlap})')
    plt.tight_layout()
    #if save_path:
    plt.savefig("specto.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    # 1) load everything
    data = load_all_signals(signal_path)

    # 2) pick the single (sequence, channel) with highest variance
    seq_idx, chan_idx = choose_best_segment(data)
    print(f"[Info] Selected sequence #{seq_idx}, channel #{chan_idx} (max variance)")

    # 3) extract that 1-D signal and plot
    signal = data[seq_idx, chan_idx, :]
    plot_signal_spectrogram(signal, fs=20000.0)
