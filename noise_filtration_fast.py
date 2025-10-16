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
def denoise_with_nmf_component_mask(
    mixture_file,
    W_noise,
    sr,
    n_animal=15,
    n_iter=300,
    n_fft=2048,
    hop=512,
    alpha_cricket=0.9,
    alpha_wind=0.8,
    n_cricket_bases=20,
    n_wind_bases=6,
    low_freq_cutoff=20,
    animal_gain=1.4
):
    # Load mixture
    y, _ = librosa.load(mixture_file, sr=sr, mono=True)
    S_complex = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    S = np.abs(S_complex).astype(np.float32)

    F, T = S.shape
    n_noise = W_noise.shape[1]

    # Initialize noise + animal components
    W_init = np.hstack([
        W_noise,
        np.abs(np.random.rand(F, n_animal)).astype(np.float32)
    ])
    H_init = np.abs(np.random.rand(W_init.shape[1], T)).astype(np.float32)

    # Semi-supervised NMF
    nmf = NMF(
        n_components=W_init.shape[1],
        init="custom",
        solver="mu",
        beta_loss="kullback-leibler",
        max_iter=n_iter,
        random_state=0
    )
    W = nmf.fit_transform(S, W=W_init, H=H_init)
    H = nmf.components_

    # Separate noise and animal
    W_noise_fit, H_noise = W[:, :n_noise], H[:n_noise, :]
    W_animal, H_animal = W[:, n_noise:], H[n_noise:, :]
    S_animal = W_animal @ H_animal
    S_noise = W_noise_fit @ H_noise

    # -----------------------------
    # Step 1: Low-frequency suppression (<20 Hz)
    # -----------------------------
    freqs = np.linspace(0, sr / 2, F)
    low_idx = np.where(freqs <= low_freq_cutoff)[0]
    S_animal[low_idx, :] = 0.0
    S_noise[low_idx, :] = 0.0

    # -----------------------------
    # Step 2: Identify cricket and wind bases
    # -----------------------------
    W_cricket = W_noise_fit[:, :n_cricket_bases]
    H_cricket = H_noise[:n_cricket_bases, :]
    S_cricket = W_cricket @ H_cricket

    W_wind = W_noise_fit[:, n_cricket_bases:n_cricket_bases + n_wind_bases]
    H_wind = H_noise[n_cricket_bases:n_cricket_bases + n_wind_bases, :]
    S_wind = W_wind @ H_wind

    # -----------------------------
    # Step 3: Soft masks for noise components
    # -----------------------------
    mask_cricket = alpha_cricket * (S_cricket / (S_cricket + S_animal + 1e-12))
    mask_cricket = median_filter(mask_cricket, size=(1, 5))
    mask_cricket = np.clip(mask_cricket, 0.0, 1.0)

    mask_wind = alpha_wind * (S_wind / (S_wind + S_animal + 1e-12))
    mask_wind = median_filter(mask_wind, size=(1, 7))
    mask_wind = np.clip(mask_wind, 0.0, 1.0)

    # -----------------------------
    # Step 4: Apply noise suppression
    # -----------------------------
    S_suppressed = S * (1 - mask_cricket) * (1 - mask_wind)
    S_suppressed = np.clip(S_suppressed, 0, None) + 1e-12  # prevent negative values

    # -----------------------------
    # Step 5: Refinement NMF on residual (animal-focused)
    # -----------------------------
    nmf2 = NMF(
        n_components=n_animal,
        init="nndsvda",
        solver="mu",
        beta_loss="kullback-leibler",
        max_iter=400,
        random_state=0
    )
    W_animal_refined = nmf2.fit_transform(S_suppressed)
    H_animal_refined = nmf2.components_
    S_animal_refined = W_animal_refined @ H_animal_refined

    # -----------------------------
    # Step 6: Combine and enhance animal sounds
    # -----------------------------
    S_clean = S_animal_refined * animal_gain
    S_clean /= np.max(S_clean) + 1e-12

    # -----------------------------
    # Step 7: Reconstruct audio
    # -----------------------------
    Y_hat = S_clean * np.exp(1j * np.angle(S_complex))
    y_hat = librosa.istft(Y_hat, hop_length=hop, length=len(y))
    y_hat /= np.max(np.abs(y_hat)) + 1e-12

    return y_hat, sr


# ---------------------------------------------------------
# 3. Example usage
# ---------------------------------------------------------
if __name__ == "__main__":
    # if no motor sounds( processing audio enhancement quicker)
    # noise_files = [
    #     "my_audio/noise/cricket_only_speaker.wav",
    #     "my_audio/noise/running_water_speaker.wav",
    #     "my_audio/noise/wind_n_leave.wav"
    # ]
    # n_bases_list = [20, 20, 6]  # more bases for complex noise

    # # Build noise dictionary
    # W_noise, sr, n_fft, hop = build_noise_dict(noise_files, n_bases_list)

    # # Input audio
    # mixture_file = "my_audio/10_seconds_speaker/Wombat_TrackA_chunk1.wav"

    # start_time = time.time()
    # denoised, sr = denoise_with_nmf_component_mask(
    #     mixture_file,
    #     W_noise,
    #     sr,
    #     n_animal=20,
    #     n_iter=300,
    #     n_fft=2048,
    #     hop=512,
    #     alpha_cricket=0.9,
    #     alpha_wind=0.8,
    #     n_cricket_bases=20,
    #     n_wind_bases=6,
    #     low_freq_cutoff=20,
    #     animal_gain=1.4
    # )
    # end_time = time.time()

    # if noise : cricket,motor, running water, wind n leaves rusttling
    noise_files = [
        "my_audio/noise/cricket_only_speaker.wav",
        "my_audio/noise/Motor.wav",
        "my_audio/noise/running_water_speaker.wav",
        "my_audio/noise/wind_n_leave.wav" # contain wind, leave rustling, some human sounds
    ]
    n_bases_list = [20, 6, 20, 6]  # more bases for complex noise, increasing the bases wouldn't improve the performance, optimal for 20 atm 

    # Build noise dictionary
    W_noise, sr, n_fft, hop = build_noise_dict(noise_files, n_bases_list)

    # Input audio
    mixture_file = "my_audio/10_seconds_speaker/Wombat_TrackA_chunk1.wav"

    start_time = time.time()
    denoised, sr = denoise_with_nmf_component_mask(
        mixture_file,
        W_noise,
        sr,
        n_animal=20,
        n_iter=300,
        n_fft=2048, # chnage n_fft & hop if want better sound quality, but increasing process time
        hop=512,
        alpha_cricket=0.9, # Adjust alpha (range from 0-1) for certain noise for soft masking
        alpha_wind=0.8, 
        n_cricket_bases=20,
        n_wind_bases=6,
        low_freq_cutoff=20, # for high-pass filter
        animal_gain=1.4 # can be increased for amplifying animal sounds but the gain is required to be less than 2 to prevent sounds from clipping
    )
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")

    sf.write("denoised_component_mask_enhanced_Frog.wav", denoised, sr)
    print("Denoising completed. Output saved to 'denoised_component_mask_enhanced.wav'")



