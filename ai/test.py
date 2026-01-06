import numpy as np
import sys
import librosa
import torch
from dataset import MelConfig, generate_melspectrogram
from model import INSTRUMENT_MAP, MusiCNN

instrument_list = list(INSTRUMENT_MAP.keys())

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def load_model(model_file):
    state = torch.load(model_file, map_location=device, weights_only=False)
    mel_config: MelConfig = state["mel_config"]
    frames_per_window = state["frames_per_window"]
    optimal_thresholds = state["optimal_thresholds"]
    model = MusiCNN(num_classes=len(INSTRUMENT_MAP), num_mels=mel_config.n_mels)
    model.load_state_dict(state["model"])
    model.to(device)
    return model, mel_config, frames_per_window, optimal_thresholds

def post_process_thresholds(thresholds):
    thresholds["drums"] *= 0.8
    thresholds["bass"] *= 0.7
    return thresholds

def analyze(model, mel_config, frames_per_window, audio, audio_sample_rate):
    model.eval()

    # 2. Process Audio
    audio = librosa.resample(audio, orig_sr=audio_sample_rate, target_sr=mel_config.sample_rate)

    mel = generate_melspectrogram(audio, mel_config)

    # 3. Create Batches (Much Faster)
    chunks = []
    total_frames = mel.shape[1]

    # Slice the mel spectrogram into 512-frame chunks
    for offset in range(0, total_frames, frames_per_window):
        chunk = mel[:, offset:offset + frames_per_window]

        if chunk.shape[1] < frames_per_window:
            pad_width = frames_per_window - chunk.shape[1]
            # Pad only the CHUNK, not 'mel'
            chunk = np.pad(chunk, ((0, 0), (0, pad_width)), mode='constant')

        chunks.append(chunk)

    if len(chunks) > 0:
        batch_tensor = np.stack(chunks)
        batch_tensor = torch.from_numpy(batch_tensor).float().unsqueeze(1).to(device)

        with torch.no_grad():
            # Run all chunks at once (or usually fits in memory for one song)
            logits = model(batch_tensor)
            probs = logits.cpu().numpy()

        # 5. Aggregate (Max Pooling over the song)
        # This aligns with your training strategy: "If it appeared once, it's there."
        song_prediction = np.max(probs, axis=0)

        return song_prediction

    return None


if __name__ == '__main__':
    audio_file = sys.argv[1]
    model_file = sys.argv[2]

    model, mel_config, frames_per_window, optimal_thresholds = load_model(model_file)
    audio, audio_sample_rate = librosa.load(audio_file, sr=None)

    song_prediction = analyze(model, mel_config, frames_per_window, audio, audio_sample_rate)
    if song_prediction is None:
        raise Exception("No data")

    optimal_thresholds = post_process_thresholds(optimal_thresholds)

    # Print Results using your computed thresholds
    print(f"\nResults for {audio_file}:")
    for inst, score in zip(instrument_list, song_prediction):
        print(f"{inst}: {score:.4f} ({"YES" if score >= optimal_thresholds[inst] else "NO"})")
