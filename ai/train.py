import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from tqdm import tqdm

# Ensure these match your actual file names
from dataset import MTG_Jamendo_Dataset, MelConfig, get_weights_from_dataset
from model import MusiCNN, INSTRUMENT_MAP

# --- Configuration ---
INSTRUMENT_MARKER = "instrument---"
DATASET_DIR = "DATASET_DIR"
LABEL_FILE = "autotagging_instrument.tsv"
LOG_DIR = "logs/test"
CHECKPOINT_DIR = "checkpoints/test"

FRAMES_PER_WINDOW = 256  # approx 5-10 seconds depending on hop size
BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-6
SPEC_AUG = True
MODEL_SIZE = "small"
GRAD_ACCUM_STEPS = 8     # Effective batch size = 64
NUM_CROPS = 5
MEL_CONFIG = MelConfig(
    sample_rate=22050,
    n_fft=2048,
    hop_size=512,
    n_mels=128,
)

# Setup Device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Mixed Precision Setup
USE_AMP = torch.cuda.is_available()
if USE_AMP:
    scaler = torch.amp.grad_scaler.GradScaler(device.type)
else:
    scaler = None

writer = SummaryWriter(LOG_DIR)

def train(model, train_loader, val_loader, epochs, optimizer, criterion,
          scheduler, instrument_list, checkpoint_dir="checkpoints", device=None,
          resume=False, grad_accum_steps=1):

    model.to(device)
    start_epoch = 0
    best_map = -1.0

    # -----------------------------
    # Resume checkpoint if exists
    # -----------------------------
    latest_ckpt = os.path.join(checkpoint_dir, "latest.pt")
    if resume and os.path.exists(latest_ckpt):
        print(f"Loading checkpoint from {latest_ckpt}")
        ck = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        if "scheduler" in ck and scheduler is not None:
            scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        best_map = ck.get("best_map", best_map)
        print(f"Resumed from epoch {start_epoch}, best_map={best_map:.4f}")

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{epochs}")

        optimizer.zero_grad()

        for step, (x, y) in enumerate(pbar):
            x = x.to(device) # Shape: [B, Crops, 1, F, T]
            y = y.float().to(device)

            # 1. FLATTEN: Combine Batch and Crops
            b, num_crops, c, f, t = x.shape
            x = x.view(b * num_crops, c, f, t)

            # 2. Forward Pass
            if scaler is not None:
                with torch.amp.autocast_mode.autocast(device.type):
                    preds = model(x) # [B*Crops, Num_Classes]

                    # 3. UN-FLATTEN & AGGREGATE
                    preds = preds.view(b, num_crops, -1)

                    # Max Pool across crops ("Did the instrument appear anywhere?")
                    preds_max, _ = torch.max(preds, dim=1) # [B, Num_Classes]

                    loss = criterion(preds_max, y)

                scaler.scale(loss / grad_accum_steps).backward()
            else:
                preds = model(x)
                preds = preds.view(b, num_crops, -1)
                preds_max, _ = torch.max(preds, dim=1)

                loss = criterion(preds_max, y)
                (loss / grad_accum_steps).backward()

            # Gradient Accumulation
            if (step + 1) % grad_accum_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()

            running_loss += loss.item()

            # Inside the training loop:
            current_lr = optimizer.param_groups[0]['lr']

            pbar.set_postfix({
                'loss': running_loss / (step + 1),
                'lr': f"{current_lr:.2e}"
            })

            if step % 50 == 0:
                writer.add_scalar('Learning Rate', current_lr, epoch * len(pbar) + step)
                writer.add_scalar('Training Loss', running_loss / (step + 1), epoch * len(pbar) + step)

        # -----------------------------
        # Validation
        # -----------------------------
        if val_loader is not None:
            model.eval()
            all_y, all_p = [], []
            with torch.no_grad():
                for x_val, y_val in tqdm(val_loader, desc="Validation"):
                    x_val = x_val.to(device)
                    y_val = y_val.float().to(device)

                    b, num_crops, c, f, t = x_val.shape
                    x_val = x_val.view(b * num_crops, c, f, t)

                    preds = model(x_val)

                    # Aggregate same as training
                    preds = preds.view(b, num_crops, -1)
                    preds_max, _ = torch.max(preds, dim=1) # [B, Num_Classes]

                    # No sigmoid here either
                    probs = preds_max.cpu().numpy()
                    all_p.append(probs)
                    all_y.append(y_val.cpu().numpy())

            all_y = np.vstack(all_y)
            all_p = np.vstack(all_p)

            ap_per_class = []
            for i in range(all_y.shape[1]):
                try:
                    ap = average_precision_score(all_y[:, i], all_p[:, i])
                except ValueError:
                    ap = 0.0
                ap_per_class.append(ap)

            map_score = np.mean(ap_per_class)
            print(f"Epoch {epoch+1} val mAP: {map_score:.4f}")
            writer.add_scalar("Mean mAP", map_score, epoch)

            for name, instrument_map_score in zip(instrument_list, ap_per_class):
                writer.add_scalar(f"{name} mAP", instrument_map_score, epoch)

            if map_score > best_map:
                best_map = map_score
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save({
                    "instrument_list": instrument_list,
                    "mel_config": MEL_CONFIG,
                    "frames_per_window": FRAMES_PER_WINDOW,
                    "model": model.state_dict(),
                }, os.path.join(checkpoint_dir, "best.pt"))
                print(f"New best model saved! mAP: {best_map:.4f}")

        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "instrument_list": instrument_list,
            "mel_config": MEL_CONFIG,
            "frames_per_window": FRAMES_PER_WINDOW,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_map": best_map
        }, latest_ckpt)

    return model

def find_optimal_thresholds(model, val_loader, device, instrument_list):
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for x_val, y_val in tqdm(val_loader, desc="Validation"):
            x_val = x_val.to(device)
            y_val = y_val.float().to(device)

            b, num_crops, c, f, t = x_val.shape
            x_val = x_val.view(b * num_crops, c, f, t)

            preds = model(x_val)

            # Aggregate same as training
            preds = preds.view(b, num_crops, -1)
            preds_max, _ = torch.max(preds, dim=1) # [B, Num_Classes]

            # No sigmoid here either
            probs = preds_max.cpu().numpy()
            all_p.append(probs)
            all_y.append(y_val.cpu().numpy())

    all_y = np.vstack(all_y)
    all_p = np.vstack(all_p)

    optimal_thresholds = {}

    # 2. Iterate through each instrument
    print("\n--- Optimal Thresholds ---")
    for i, instrument in enumerate(instrument_list):
        y_true = all_y[:, i]
        y_score = all_p[:, i]

        best_f1 = 0
        best_thresh = 0.5

        # Test thresholds from 0.1 to 0.9
        for thresh in np.arange(0.1, 0.95, 0.05):
            y_pred = (y_score >= thresh).astype(int)
            f1 = f1_score(y_true, y_pred)

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        optimal_thresholds[instrument] = best_thresh
        print(f"{instrument}: {best_thresh:.2f} (F1: {best_f1:.3f})")

    return optimal_thresholds

def main():
    instrument_list = list(INSTRUMENT_MAP.keys())

    # Ensure Dataset accepts num_crops
    train_dataset = MTG_Jamendo_Dataset(DATASET_DIR, LABEL_FILE, INSTRUMENT_MARKER, INSTRUMENT_MAP,
                                        FRAMES_PER_WINDOW, NUM_CROPS, MEL_CONFIG, is_train=True)
    val_dataset = MTG_Jamendo_Dataset(DATASET_DIR, LABEL_FILE, INSTRUMENT_MARKER, INSTRUMENT_MAP,
                                      FRAMES_PER_WINDOW, NUM_CROPS, MEL_CONFIG, is_train=False)

    all_indices = list(range(len(train_dataset)))
    random.seed(42)
    random.shuffle(all_indices)
    split = int(0.9 * len(all_indices))
    train_indices = all_indices[:split]
    val_indices = all_indices[split:]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    print("Calculating sampler weights...")

    train_weights = get_weights_from_dataset(train_dataset)[train_indices]

    train_sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        sampler=train_sampler,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"Creating MusiCNN model...")

    model = MusiCNN(num_classes=len(instrument_list), num_mels=MEL_CONFIG.n_mels)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    criterion = nn.BCELoss()

    steps_per_epoch = len(train_loader) // GRAD_ACCUM_STEPS
    total_steps = steps_per_epoch * EPOCHS

    print(f"Total training steps: {total_steps}")

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        total_steps=total_steps,
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000
    )

    trained = train(
        model, train_loader, val_loader,
        epochs=EPOCHS,
        optimizer=optimizer,
        device=device,
        criterion=criterion,
        resume=True,
        checkpoint_dir=CHECKPOINT_DIR,
        scheduler=scheduler,
        instrument_list=instrument_list,
        grad_accum_steps=GRAD_ACCUM_STEPS
    )

    optimal_thresholds = find_optimal_thresholds(model, val_loader, device, instrument_list)

    torch.save({
        "instrument_list": instrument_list,
        "mel_config": MEL_CONFIG,
        "frames_per_window": FRAMES_PER_WINDOW,
        "optimal_thresholds": optimal_thresholds,
        "model": trained.state_dict(),
    }, os.path.join(CHECKPOINT_DIR, "final.pt"))


if __name__ == "__main__":
    main()
