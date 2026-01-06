# Cleanup unused songs from the dataset to save space

import sys, os

label_file = sys.argv[1]
dataset_dir = sys.argv[2]

mp3s = []

with open(label_file, "r") as f:
    f.readline() # read table header

    for line in f:
        parts = line.strip().split("\t")
        if (len(parts) == 0):
            continue
        mp3_name = os.path.basename(parts[3].replace(".mp3", ".low.mp3"))
        mp3s.append(mp3_name)

size_bytes = 0
for dir_name in os.listdir(dataset_dir):
    dir_path = os.path.join(dataset_dir, dir_name)
    if not os.path.isdir(dir_path):
        continue

    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        if file_name not in mp3s:
            size_bytes += os.path.getsize(file_path)
            print(f"removing {file_path}")
            os.remove(file_path)

print(f"Freed {(size_bytes / (1024 * 1024 * 1024)):.4f} GB")
