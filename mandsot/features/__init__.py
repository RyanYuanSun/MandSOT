import librosa
import numpy as np
import noisereduce as nr
import matplotlib.pyplot as plt


def get_features(wav_path, denoise=False, ms1_w=1, ms2_w=1, ms3_w=1):
    y, sr = librosa.load(wav_path, sr=None)

    # get noise window
    if denoise:
        noise_start, noise_stop = get_noise_window(y, sr, win_len=0.5)
        noise_start = librosa.time_to_samples(noise_start, sr=sr)
        noise_stop = librosa.time_to_samples(noise_stop, sr=sr)
        noise = y[noise_start:noise_stop]
        y = nr.reduce_noise(y=y, sr=int(sr), n_jobs=-1, stationary=False, prop_decrease=1, y_noise=noise)

    # resample
    target_sr = 48000
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    # padding/trimming
    target_dur = 3
    target_sample = target_dur * target_sr
    if len(y) < target_sample:
        pad_width = target_sample - len(y)
        y = np.pad(y, pad_width=(0, pad_width))
    else:
        y = y[:target_sample]

    # pre-emphasis
    y = librosa.effects.preemphasis(y, coef=0.97)

    # multi hop spectrogram
    f_max = 10000
    ms1 = librosa.feature.melspectrogram(y=y, sr=target_sr, n_fft=1024, n_mels=80, fmax=f_max, hop_length=128, window='hamming')
    ms2 = librosa.feature.melspectrogram(y=y, sr=target_sr, n_fft=512, n_mels=40, fmax=f_max, hop_length=128, window='hamming')
    ms3 = librosa.feature.melspectrogram(y=y, sr=target_sr, n_fft=256, n_mels=20, fmax=f_max, hop_length=128, window='hamming')
    ms1 = ms1[:, :6000]
    ms2 = np.repeat(ms2, 2, axis=0)[:, :6000]
    ms3 = np.repeat(ms3, 4, axis=0)[:, :6000]
    ms1 = ms1 * ms1_w
    ms2 = ms2 * ms2_w
    ms3 = ms3 * ms3_w
    ms_total = (ms1 + ms2 + ms3) / (ms1_w + ms2_w + ms3_w)  # weighted blended spectrum
    ms1 = librosa.core.power_to_db(ms1, ref=np.max)
    ms2 = librosa.core.power_to_db(ms2, ref=np.max)
    ms3 = librosa.core.power_to_db(ms3, ref=np.max)
    ms_total = librosa.core.power_to_db(ms_total, ref=np.max)

    # normalize
    data_min = np.min(ms_total)
    data_max = np.max(ms_total)
    ms_total = (ms_total - data_min) / (data_max - data_min)

    return ms1, ms2, ms3, ms_total


def get_noise_window(y, sr, win_len=0.5, nfft=2048):
    dur = librosa.get_duration(y=y, sr=sr)
    s = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=nfft, n_mels=80, hop_length=int(nfft/2), window='hamming')
    win_npts = round((win_len / dur) * s.shape[1])

    if win_len > dur:
        return -1

    for k in range(1, (s.shape[1] - win_npts)):  # loop time points
        amp_std = 0

        for kk in range(s.shape[0]):  # loop frequencies
            amp_std_f = np.std(np.abs(s[kk, k:k + win_npts]) ** 2)
            amp_std += amp_std_f

        if k == 1:
            min_std_idx = 1
            cur_min_std = amp_std
        else:
            if amp_std < cur_min_std:
                cur_min_std = amp_std
                min_std_idx = k

    n_start_t = min_std_idx / s.shape[1] * dur
    n_stop_t = (min_std_idx + win_npts) / s.shape[1] * dur

    return n_start_t, n_stop_t


def view_features(audio, denoise):
    spec_1, spec_2, spec_3, spec_t = get_features(wav_path=audio, denoise=denoise)
    fig, axes = plt.subplots(4, 1)
    librosa.display.specshow(spec_1, ax=axes[0])
    librosa.display.specshow(spec_2, ax=axes[1])
    librosa.display.specshow(spec_3, ax=axes[2])
    librosa.display.specshow(spec_t, ax=axes[3])
    plt.show()

