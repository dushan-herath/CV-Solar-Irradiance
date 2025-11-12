import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import random
import os

class IrradianceForecastDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        split: str = "train",
        val_ratio: float = 0.25,
        img_seq_len: int = 5,
        ts_seq_len: int = 30,
        horizon: int = 25,
        feature_cols=None,
        target_cols=None,
        transform=None,
        img_size: int = 224,
        time_col: str = "timestamp",
        normalization_stats: dict = None,
    ):
        full_df = pd.read_csv(csv_path)
        n = len(full_df)
        split_idx = int(n * (1 - val_ratio))

        # --- Split dataset
        if split == "train":
            self.df = full_df.iloc[:split_idx].reset_index(drop=True)
        elif split == "val":
            self.df = full_df.iloc[split_idx:].reset_index(drop=True)
        else:
            raise ValueError("split must be 'train' or 'val'")

        self.split = split
        self.img_seq_len = img_seq_len
        self.ts_seq_len = ts_seq_len
        self.horizon = horizon
        self.img_size = img_size
        self.time_col = time_col
        #self.feature_cols = feature_cols or ["ghi", "dni", "dhi", "temp", "pressure"]
        self.feature_cols = feature_cols or ["ghi", "dni", "dhi"]
        self.target_cols = target_cols or ["ghi", "dni", "dhi"]
        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        self.max_lookback = max(img_seq_len, ts_seq_len)

        # Parse timestamps
        if self.time_col in self.df.columns:
            self.df[self.time_col] = pd.to_datetime(self.df[self.time_col])

        # --- Normalization ---
        if split == "train":
            mean = self.df[self.feature_cols].mean()
            std = self.df[self.feature_cols].std()
            self.normalization_stats = {"mean": mean, "std": std}
        else:
            if normalization_stats is None:
                raise ValueError("Validation split requires normalization_stats from training set")
            self.normalization_stats = normalization_stats
            mean = normalization_stats["mean"]
            std = normalization_stats["std"]

        # Apply normalization using training stats
        # Print the mean and std values used for normalization
        # print("\nNormalization statistics applied:")
        # print("Mean values:")
        # print(mean)
        # print("\nStandard deviation values:")
        # print(std)
        self.df[self.feature_cols] = (self.df[self.feature_cols] - mean) / std

        # --- Summary ---
        print(f"\nDataset initialized ({split.upper()}):")
        print(f"Total samples available: {len(self)}")
        print(f"Image seq length: {self.img_seq_len}")
        print(f"Time-series seq length: {self.ts_seq_len}")
        print(f"Forecast horizon: {self.horizon}")
        print(f"Features normalized using: {'train set' if split=='train' else 'provided stats'}")
        print(f"Mean (first 3): {mean.values[:3]}")
        print(f"Std (first 3): {std.values[:3]}")
        print(f"Feature columns: {self.feature_cols}")
        print(f"Target columns: {self.target_cols}\n")

    def __len__(self):
        return len(self.df) - self.max_lookback - self.horizon

    def __getitem__(self, idx):
        img_window = self.df.iloc[idx + self.ts_seq_len - self.img_seq_len : idx + self.ts_seq_len]
        ts_window = self.df.iloc[idx : idx + self.ts_seq_len]
        target_window = self.df.iloc[idx + self.ts_seq_len : idx + self.ts_seq_len + self.horizon]

        # --- Image sequence ---
        img_seq = []
        for path in img_window["image_path"].values:
            image = Image.open(path).convert("RGB")
            image = self.transform(image)
            img_seq.append(image)
        img_seq = torch.stack(img_seq)

        # --- Time-series sequence ---
        ts_seq = torch.tensor(ts_window[self.feature_cols].values, dtype=torch.float32)

        # --- Target sequence ---
        target_seq = torch.tensor(target_window[self.target_cols].values, dtype=torch.float32)

        # --- Timestamps ---
        if self.time_col in ts_window.columns:
            ts_times = [str(t) for t in ts_window[self.time_col].tolist()]
        else:
            ts_times = list(range(len(ts_window)))

        if self.time_col in target_window.columns:
            target_times = [str(t) for t in target_window[self.time_col].tolist()]
        else:
            target_times = list(range(len(ts_window), len(ts_window) + len(target_window)))

        # --- Image names ---
        img_names = [os.path.basename(p) for p in img_window["image_path"].values]

        # --- Return everything ---
        return img_seq, ts_seq, target_seq, ts_times, target_times, img_names


    # Visualization helper
    def show_sample(self, idx=None):
        """Show one dataset sample: images, time-series, and target curves."""
        if idx is None:
            idx = random.randint(0, len(self) - 1)

        img_seq, ts_seq, target_seq, ts_times, target_times, img_names = self[idx]

        print(f"\nSample index: {idx} ({self.split.upper()} set)")
        print(f"Image sequence: {img_seq.shape}")
        print(f"Time-series sequence: {ts_seq.shape}")
        print(f"Target sequence: {target_seq.shape}")

        # --- Plot images ---
        num_images = self.img_seq_len
        fig, axes = plt.subplots(1, num_images, figsize=(3*num_images, 3))
        if num_images == 1:
            axes = [axes]
        for i in range(num_images):
            img = img_seq[i].permute(1, 2, 0).numpy()
            axes[i].imshow(img)
            axes[i].axis("off")
            axes[i].set_title(img_names[i], fontsize=8)
        plt.suptitle("Image Sequence", fontsize=12)
        plt.tight_layout()
        plt.show()

        # --- Print sample data ---
        print("\nFirst 5 time-series samples:")
        print(pd.DataFrame(ts_seq[:5].numpy(), columns=self.feature_cols))

        print("\nFirst 5 target values:")
        print(pd.DataFrame(target_seq[:5].numpy(), columns=self.target_cols))

        # --- Plot irradiance time evolution ---
        plt.figure(figsize=(10, 4))
        past_vals = ts_seq[:, :3].numpy()
        future_vals = target_seq.numpy()
        for i, col in enumerate(self.target_cols):
            plt.plot(ts_times, past_vals[:, i], '--o', label=f"Past {col.upper()}")
            plt.plot(target_times, future_vals[:, i], '-x', label=f"Future {col.upper()}")
        plt.xlabel("Time")
        plt.ylabel("Irradiance (W/m²)")
        plt.title("GHI/DNI/DHI Forecast Visualization")
        plt.xticks(rotation=90)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
