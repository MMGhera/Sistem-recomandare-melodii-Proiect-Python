import os
import random
import numpy as np
import librosa
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset

class MTG_Jamendo_Dataset(Dataset):
    def __init__(self, mel_dir, label_file, marker_name, class_map, frames_per_window, num_crops, mel_config, is_train=True, spec_aug="adaptive"):
        self.dataset_dir = mel_dir
        self.name_to_idx = {name: i for i, name in enumerate(class_map.keys())}

        self.frames_per_window = frames_per_window
        self.mel_config = mel_config
        self.samples_per_window = (frames_per_window - 1) * self.mel_config.hop_size

        self.num_crops = num_crops
        self.is_train = is_train
        self.spec_aug = spec_aug
        self.items = []

        with open(label_file, "r") as f:
            _ = f.readline()
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 5: continue

                mp3_path = os.path.join(self.dataset_dir, parts[3].replace(".mp3", ".low.mp3"))
                # if not os.path.exists(mp3_path): continue

                tags = parts[5:]
                vec = np.zeros(len(class_map), dtype=np.float32)
                has_label = False

                for tag in tags:
                    if tag.startswith(marker_name):
                        name = tag[len(marker_name):]
                        for final, aliases in class_map.items():
                            if name == final or name in aliases:
                                vec[self.name_to_idx[final]] = 1.0
                                has_label = True

                if has_label:
                    self.items.append((mp3_path, vec))

        print(f"Loaded {len(self.items)} tracks.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        mp3_path, target = self.items[idx]

        try:
            audio, sr = librosa.load(mp3_path, sr=self.mel_config.sample_rate)
        except Exception as e:
            print(f"Error loading {mp3_path}: {e}")
            audio = np.zeros(self.samples_per_window, dtype=np.float32)

        if len(audio) < self.samples_per_window:
            padding = self.samples_per_window - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')

        crops = []

        for _ in range(self.num_crops):
            # Random crop for each glimpse
            start_sample = random.randint(0, max(0, len(audio) - self.samples_per_window))
            audio_crop = audio[start_sample : start_sample + self.samples_per_window]

            # Generate Mel
            mel = generate_melspectrogram(audio_crop, self.mel_config)

            # Augment (Apply DIFFERENT masks to each crop!)
            if self.is_train and self.spec_aug:
                mel = spec_augment_adaptive(mel, replace_with_zero=True)

            crops.append(mel)

        # Stack them: [Num_Crops, 96, Time]
        crops_stack = np.stack(crops)

        # Convert to Tensor: [Num_Crops, 1, 96, Time]
        mel_tensor = torch.from_numpy(crops_stack).float().unsqueeze(1)
        target_tensor = torch.from_numpy(target).float()

        return mel_tensor, target_tensor

@dataclass
class MelConfig:
    sample_rate: int = 12000
    n_fft: int = 512
    hop_size: int = 256
    n_mels: int = 96
    fmin: int = 0
    fmax: int = None
    top_db: float = 80.0
    power: float = 2.0

def generate_melspectrogram(audio, config: MelConfig):
    # Calculate fmax if not provided
    fmax = config.fmax if config.fmax is not None else config.sample_rate / 2

    mel_power = librosa.feature.melspectrogram(
        y=audio,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_size,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=fmax,
        norm='slaney', # 'slaney' is usually preferred for magnitude accuracy
        power=config.power
    )

    # 1. Log scale (dB)
    # ref=np.max ensures the loudest pixel is 0 dB
    log_mel = librosa.power_to_db(mel_power, ref=np.max, top_db=config.top_db)

    # 2. Normalize based on the configured top_db
    # This assumes power_to_db returns values in range [-top_db, 0]
    log_mel = (log_mel + config.top_db) / config.top_db

    # Clip to ensure numerical stability [0, 1]
    log_mel = np.clip(log_mel, 0.0, 1.0)

    return log_mel

def spec_augment_adaptive(mel, freq_mask_ratio=0.15, time_mask_ratio=0.15,
                          num_freq_masks=2, num_time_masks=2, replace_with_zero=True):

    mel = mel.copy()
    n_mels, T = mel.shape

    # Fill value: 0.0 is best for [0,1] normalized data (represents silence)
    fill_value = 0.0 if replace_with_zero else mel.mean()

    # Frequency Masking
    max_freq_width = max(1, int(n_mels * freq_mask_ratio))
    for _ in range(num_freq_masks):
        f = random.randint(0, max_freq_width)
        if f == 0: continue
        f0 = random.randint(0, max(0, n_mels - f))
        mel[f0:f0+f, :] = fill_value

    # Time Masking
    max_time_width = max(1, int(T * time_mask_ratio))
    for _ in range(num_time_masks):
        t = random.randint(0, max_time_width)
        if t == 0: continue
        t0 = random.randint(0, max(0, T - t))
        mel[:, t0:t0+t] = fill_value

    return mel

def get_weights_from_dataset(dataset):
    weights = []

    counts = np.zeros(len(dataset.items[0][1]), dtype=np.float32)
    for _, targets in dataset.items:
        counts += targets
    counts = np.maximum(counts, 1.0)
    print(counts)
    max_freq = max(counts)

    for _, targets in dataset.items:
        weight = 0.0
        indices = np.where(np.array(targets) > 0.1)[0]
        if len(indices) > 0:
            weight = np.max(max_freq / counts[indices])
        weight = max(weight, 1e-4)
        weights.append(weight)

    return torch.DoubleTensor(weights)
