import numpy as np
import librosa
import soundfile as sf
from sklearn.decomposition import NMF
from scipy.ndimage import median_filter
import time

# ---------------------------------------------------------
# 1. Build noise dictionary from noise files
# ---------------------------------------------------------
def build_noise_dict(noise_files, n_bases_list, n_fft=2048, hop=512):
    W_list = []
    for nf, n_bases in zip(noise_files, n_bases_list):
        y, sr = librosa.load(nf, sr=None, mono=True)
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)).astype(np.float32)
        nmf = NMF(n_components=n_bases, init="nndsvda", solver="mu",
                  beta_loss="kullback-leibler", max_iter=200, random_state=0)
        W = nmf.fit_transform(S)
        W_list.append(W)
    return np.hstack(W_list), sr, n_fft, hop

# ---------------------------------------------------------
# 2. Semi-supervised NMF with cricket & low-frequency noise removal and animal enhancement
# ---------------------------------------------------------
def denoise_with_nmf_component_mask(mixture_file, W_noise, sr, n_animal=15,
                                    n_iter=250, n_fft=2048, hop=512,
                                    alpha_noise=0.9, n_cricket_bases=20,
                                    low_freq_cutoff=20, animal_gain=1.5):
    # Load mixture
    y, _ = librosa.load(mixture_file, sr=sr, mono=True)
    S_complex = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    S = np.abs(S_complex).astype(np.float32)

    F, T = S.shape
    n_noise = W_noise.shape[1]

    # Initialize animal and noise NMF components
    W_init = np.hstack([W_noise, np.abs(np.random.rand(F, n_animal)).astype(np.float32)])
    H_init = np.abs(np.random.rand(W_init.shape[1], T)).astype(np.float32)

    nmf = NMF(n_components=W_init.shape[1], init="custom", solver="mu",
              beta_loss="kullback-leibler", max_iter=n_iter, random_state=0)
    W = nmf.fit_transform(S, W=W_init, H=H_init)
    H = nmf.components_

    # Separate noise and animal components
    W_noise_fit, H_noise = W[:, :n_noise], H[:n_noise, :]
    W_animal, H_animal = W[:, n_noise:], H[n_noise:, :]
    S_animal = W_animal @ H_animal
    S_noise = W_noise_fit @ H_noise

    # -----------------------------
    # Step 1: Low-frequency noise removal
    # -----------------------------
    freqs = np.linspace(0, sr / 2, F)
    low_idx = np.where(freqs <= low_freq_cutoff)[0]
    S_animal[low_idx, :] = 0.0
    S_noise[low_idx, :] = 0.0

    # -----------------------------
    # Step 2: Cricket noise soft mask
    # -----------------------------
    W_cricket, H_cricket = W_noise_fit[:, :n_cricket_bases], H_noise[:n_cricket_bases, :]
    S_cricket = W_cricket @ H_cricket
    mask = alpha_noise * (S_cricket / (S_cricket + S_animal + 1e-12))
    mask = median_filter(mask, size=(1,3))
    mask = np.clip(mask, 0.0, 1.0)
    S_cricket_attenuated = S_cricket * (1 - mask)

    # -----------------------------
    # Step 3: Enhance animal sounds only
    # -----------------------------
    S_animal_enhanced = S_animal * animal_gain

    # -----------------------------
    # Step 4: Combine enhanced animal + attenuated cricket
    # -----------------------------
    S_clean = S_animal_enhanced + S_cricket_attenuated


    # -----------------------------
    # Reconstruct audio with original phase and preserve length
    # -----------------------------
    Y_hat = S_clean * np.exp(1j * np.angle(S_complex))
    y_hat = librosa.istft(Y_hat, hop_length=hop, length=len(y))

    # Normalize
    y_hat /= np.max(np.abs(y_hat)) + 1e-12
    return y_hat, sr

# ---------------------------------------------------------
# Example usage
# ---------------------------------------------------------
if __name__ == "__main__":
    noise_files = [
        "my_audio/noise/cricket_only_speaker.wav",
        "my_audio/noise/Motor.wav",
        "my_audio/noise/running_water_speaker.wav",
    ]
    n_bases_list = [20, 6, 20]  # bases for cricket & water

    # Build noise dictionary
    W_noise, sr, n_fft, hop = build_noise_dict(noise_files, n_bases_list)

    mixture_file = "my_audio/10_seconds_speaker/Possum_TrackA_chunk1.wav"
    start_time = time.time()
    denoised, sr = denoise_with_nmf_component_mask(
        mixture_file,
        W_noise,
        sr,
        n_animal=15,
        n_iter=250,
        n_fft=2048,
        hop=512,
        alpha_noise=0.9,       # attenuation strength for cricket
        n_cricket_bases=20,    # first N bases of W_noise reserved for cricket
        low_freq_cutoff=20,    # remove low-frequency noise below 20 Hz
        animal_gain=2          # amplify animal sounds
    )
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")

    sf.write("denoised_component_mask_enhanced.wav", denoised, sr)
    print("Denoising completed. Output saved to 'denoised_component_mask_enhanced.wav'")
