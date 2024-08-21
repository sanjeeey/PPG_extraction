import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import pinv
from scipy.signal import butter, filtfilt, find_peaks
import sys

# Define the bandpass filter for PPG signals
def bandpass_filter_ppg(signal, lowcut=0.6, highcut=3.0, fs=30.0, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal, axis=0)

def apply_dis_method(color_changes, motion_signals, signature):
    min_len = min(color_changes.shape[1], motion_signals.shape[0])
    color_changes = color_changes[:, :min_len]
    motion_signals = motion_signals[:min_len]

    S = np.vstack([color_changes, motion_signals[np.newaxis, :]])
    S_mean = np.mean(S, axis=1, keepdims=True)
    S_norm = S - S_mean
    S_filtered = bandpass_filter_ppg(S_norm.T).T
    e_bar = np.concatenate([signature, np.zeros(motion_signals.shape[0])])

    w = np.dot(e_bar[:S_filtered.shape[0]], pinv(np.dot(S_filtered, S_filtered.T)))
    ppg_signal = np.dot(w, S_filtered)
    return ppg_signal

def extract_color_signals(patches):
    color_signals = []
    for patch in patches:
        mean_color = np.mean(patch.reshape(-1, 3), axis=0)
        color_signals.append(mean_color)
    return np.array(color_signals).T

def normalize_patch(patch):
    mean_intensity = np.mean(patch)
    std_intensity = np.std(patch)
    if std_intensity != 0:
        normalized_patch = (patch - mean_intensity) / std_intensity
    else:
        normalized_patch = patch - mean_intensity
    return normalized_patch

def compute_optical_flow(prev_frame, curr_frame):
    Ix = cv2.Sobel(prev_frame, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(prev_frame, cv2.CV_64F, 0, 1, ksize=5)
    It = curr_frame.astype(np.float64) - prev_frame.astype(np.float64)
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
    color_changes = extract_color_signals(normalized_segments)

    print(f"color_changes shape: {color_changes.shape}")
    
    signature = np.array([0.3, 0.8, 0.5])
    disturbance_signals = v.flatten()
    print(f"motion_signals shape: {disturbance_signals.shape}")

    ppg_signals = apply_dis_method(color_changes, disturbance_signals, signature)
    
    print(f"ppg_signals shape: {ppg_signals.shape}")
    
    if ppg_signals.ndim == 1:
        time_axis = np.arange(ppg_signals.shape[0]) / 30.0
    else:
        time_axis = np.arange(ppg_signals.shape[1]) / 30.0
    
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

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break

        print(f"Processing frame {frame_idx}")

        target_size = (320, 240)
        binned_image = binning_method(frame, target_size)
        
        gray_frame = cv2.cvtColor(binned_image, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is not None:
            Ix, Iy, It = compute_optical_flow(prev_frame, gray_frame)
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
            
            plot_ppg_signals(ppg_signals, time_axis, frame_idx)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit("Video processing terminated by user.")

        prev_frame = gray_frame
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Video processing completed.")

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

if __name__ == "__main__":
    video_path = "video.mp4"
    process_video(video_path)
