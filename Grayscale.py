import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import pinv
from scipy.signal import butter, filtfilt, find_peaks
from scipy.signal import welch

# Define the bandpass filter for PPG signals
def bandpass_filter_ppg(signal, lowcut=0.6, highcut=3.0, fs=30.0, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal, axis=0)

def apply_dis_method(intensity_changes, motion_signals, signature):
    min_len = min(intensity_changes.shape[0], motion_signals.shape[0])
    intensity_changes = intensity_changes[:min_len]
    motion_signals = motion_signals[:min_len]

    S = np.vstack([intensity_changes, motion_signals])
    S_mean = np.mean(S, axis=1, keepdims=True)
    S_norm = S - S_mean
    S_filtered = bandpass_filter_ppg(S_norm.T).T
    e_bar = np.concatenate([signature, np.zeros(motion_signals.shape[0])])

    w = np.dot(e_bar[:S_filtered.shape[0]], pinv(np.dot(S_filtered, S_filtered.T)))
    ppg_signal = np.dot(w, S_filtered)
    return ppg_signal

def extract_intensity_changes(patches):
    intensity_changes = []
    for patch in patches:
        mean_intensity = np.mean(patch)
        intensity_changes.append(mean_intensity)
    return np.array(intensity_changes).reshape(-1, 1)

def normalize_patch(patch):
    mean_intensity = np.mean(patch)
    std_intensity = np.std(patch)
    normalized_patch = (patch - mean_intensity) / std_intensity
    return normalized_patch

def compute_optical_flow(prev_gray, curr_gray):
    Ix = cv2.Sobel(prev_gray, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(prev_gray, cv2.CV_64F, 0, 1, ksize=5)
    It = curr_gray.astype(np.float64) - prev_gray.astype(np.float64)
    return Ix, Iy, It

def calculate_motion_vectors(Ix, Iy, It, window_size=5):
    half_window = window_size // 2
    u = np.zeros(Ix.shape)
    v = np.zeros(Ix.shape)
    
    for i in range(half_window, Ix.shape[0] - half_window):
        for j in range(half_window, Ix.shape[1] - half_window):
            Ix_window = Ix[i-half_window:i+half_window+1, j-half_window:j+half_window+1].flatten()
            Iy_window = Iy[i-half_window:i+half_window+1, j-half_window:j+half_window+1].flatten()
            It_window = It[i-half_window:i+half_window+1, j-half_window:j+half_window+1].flatten()

            if Ix_window.shape[0] == Iy_window.shape[0] == It_window.shape[0]:
                A = np.vstack((Ix_window, Iy_window)).T
                b = -It_window

                nu = np.linalg.pinv(A.T @ A) @ A.T @ b

                u[i, j] = nu[0]
                v[i, j] = nu[1]

    return u, v

def generate_parallel_ppg_signals(segments, u, v):
    normalized_segments = [normalize_patch(segment) for segment in segments]
    color_changes = extract_intensity_changes(normalized_segments)

    signature = np.array([1.0])

    disturbance_signals = v.flatten()

    ppg_signals = apply_dis_method(color_changes.flatten(), disturbance_signals, signature)
    
    if ppg_signals.ndim == 1:
        time_axis = np.arange(ppg_signals.shape[0]) / 30.0  # Assuming 30 FPS
    else:
        time_axis = np.arange(ppg_signals.shape[1]) / 30.0  # Assuming 30 FPS
    
    return ppg_signals, time_axis

def binning_method(image, target_size):
    bin_size = 10
    (h, w) = image.shape[:2]
    h_bins = h // bin_size
    w_bins = w // bin_size
    small_image = cv2.resize(image, (w_bins, h_bins), interpolation=cv2.INTER_LINEAR)
    binned_image = cv2.resize(small_image, (w, h), interpolation=cv2.INTER_NEAREST)
    resized_image = cv2.resize(binned_image, target_size, interpolation=cv2.INTER_LINEAR)
    
    cv2.imshow("binned_image", resized_image)
    cv2.waitKey(1)
    return resized_image

def segment_image(image, block_size, stride):
    (h, w) = image.shape[:2]
    segments = []
    for y in range(0, h - block_size + 1, stride):
        for x in range(0, w - block_size + 1, stride):
            segment = image[y:y + block_size, x:x + block_size]
            segments.append(segment)
            cv2.imshow(f'Block Size {block_size} Segment', segment)
            cv2.waitKey(1)
    return segments

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_idx = 0
    prev_frame = None

    snr_values = []

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break

        print(f"Processing frame {frame_idx}")

        # Extract the green channel and treat it as grayscale
        gray_frame = frame[:, :, 1]  # Extract green channel
        green_frame = cv2.merge([np.zeros_like(gray_frame), gray_frame, np.zeros_like(gray_frame)])
    
        target_size = (320, 240)
        binned_image = binning_method(green_frame, target_size)
        
        if prev_frame is not None:
            Ix, Iy, It = compute_optical_flow(prev_frame, binned_image)
            u, v = calculate_motion_vectors(Ix, Iy, It)

            scales = [28, 48, 70, 100]
            all_segments = []
            for scale in scales:
                stride = scale // 2
                segments = segment_image(binned_image, scale, stride)
                all_segments.extend(segments)
            
            ppg_signals, time_axis = generate_parallel_ppg_signals(all_segments, u, v)
            
            for i, ppg_signal in enumerate(ppg_signals):
                print(f"Frame {frame_idx}, PPG Signal {i+1}: {ppg_signal}")

            heart_rate = calculate_heart_rate(ppg_signals, 30)
            print(f"Heart Rate: {heart_rate}")
            
            plot_ppg_signals(ppg_signals, time_axis, frame_idx)

            # Calculate SNR
            snr = calculate_snr(ppg_signals)
            snr_values.append(snr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
        prev_frame = binned_image
        frame_idx += 1


    cap.release()
    print("Video processing completed.")

def calculate_heart_rate(ppg_signals, fs):
    filtered_ppg = bandpass_filter_ppg(ppg_signals, 0.6, 3.0, fs)
    peaks, _ = find_peaks(filtered_ppg, distance=fs/2)
    rr_intervals = np.diff(peaks) / fs 
    heart_rate = 60.0 / rr_intervals  
    return heart_rate

def plot_ppg_signals(ppg_signals, time_axis, frame_idx):
    plt.figure(figsize=(10, 4))
    if len(ppg_signals.shape) > 1:
        for i in range(ppg_signals.shape[0]):
            plt.plot(time_axis, ppg_signals[i, :], label=f"PPG Signal {i+1}")
    else:
        plt.plot(time_axis, ppg_signals, label=f"PPG Signal")
    plt.title(f"PPG Signals - Frame {frame_idx}")
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def calculate_snr(ppg_signal):
    # Estimate the signal power
    signal_power = np.mean(np.square(ppg_signal))
    
    # Estimate the noise power
    f, Pxx_den = welch(ppg_signal, fs=30.0, nperseg=256)
    noise_power = np.mean(Pxx_den[(f < 0.6) | (f > 3.0)])
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

if __name__ == "__main__":
    video_path = "karan.mp4"  # Replace with your video file path
    process_video(video_path)
