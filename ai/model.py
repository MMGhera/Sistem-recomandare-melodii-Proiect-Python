import torch
import torch.nn.functional as F
from torch import nn

INSTRUMENT_MAP = {
    # Target Class      # Source Tags in your file
    "piano":            ["piano", "pianosolo", "pianoforte", "grandpiano"],
    "synthesizer":      ["synthesizer", "synth", "synths", "synthetizer", "synthesizers"],
    "drums":            ["drum", "drums", "drummachine", "batterie", "percussion"],
    "bass":             ["bass", "doublebass", "bassguitar", "basso", "contrabass"],
    "electric_guitar":  ["electricguitar", "electricguitars", "distortedguitar"],
    "acoustic_guitar":  ["acousticguitar", "classicalguitar", "guitar", "guitars", "guitare"],
    "voice":            ["voice", "vocals", "choir", "femalevoice", "malevoice"],
    "violin":           ["violin", "violins", "fiddle", "violon"],
    "cello":            ["cello", "violoncello"],
    "flute":            ["flute", "flutes"],
    "saxophone":        ["saxophone", "sax"],
    "trumpet":          ["trumpet", "brass"],
}

class Attention1D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.att_conv = nn.Conv1d(in_channels, num_classes, kernel_size=1)
        self.cla_conv = nn.Conv1d(in_channels, num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=2) # Softmax over time

    def forward(self, x):
        # x: [Batch, Channels, Time]

        # Attention Score
        att = self.softmax(self.att_conv(x))

        # Classification Score
        cla = torch.sigmoid(self.cla_conv(x))

        # Weighted Sum
        x = torch.sum(cla * att, dim=2)
        return x

class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=p)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class MusiCNN(nn.Module):
    def __init__(self, num_classes, num_mels=96):
        super().__init__()

        # --- 1. The Harmonic (Timbre) Front-End ---
        # Vertical filters that look at the spectral shape.
        # We explicitly define heights relative to the input (96 bands).
        # These capture the "vertical lines" of harmonics.
        self.timbre_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(int(0.9 * num_mels), 1), bias=False),
                nn.BatchNorm2d(32), nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(int(0.7 * num_mels), 1), bias=False),
                nn.BatchNorm2d(32), nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(int(0.4 * num_mels), 1), bias=False),
                nn.BatchNorm2d(32), nn.ReLU()
            )
        ])

        # --- 2. The Temporal (Rhythm) Front-End ---
        # Horizontal filters that look at onsets and decay.
        # These capture the "horizontal lines" of sustained notes or sharp hits.
        self.temporal_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(1, 3), padding=(0, 1), bias=False),
                nn.BatchNorm2d(32), nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(1, 5), padding=(0, 2), bias=False),
                nn.BatchNorm2d(32), nn.ReLU()
            )
        ])

        # Calculate output channels after concatenating all branches
        # 3 timbre branches * 32 + 2 temporal branches * 32 = 160 channels
        self.frontend_channels = (3 * 32) + (2 * 32)

        # --- 3. The 1D Backbone ---
        # Once we extract timbre/rhythm, we flatten frequency and process as a time-series
        self.backbone = nn.Sequential(
            ConvBlock1D(self.frontend_channels, 128),
            nn.MaxPool1d(2),
            ConvBlock1D(128, 256),
            nn.MaxPool1d(2),
            ConvBlock1D(256, 512),
            nn.MaxPool1d(2),
            ConvBlock1D(512, 512),
        )

        # --- 4. Output Head (Attention) ---
        self.head = Attention1D(512, num_classes)

    def forward(self, x):
        # x: [Batch, 1, Freq(96), Time]

        features = []

        # Run Timbre Branches (Vertical)
        for block in self.timbre_convs:
            out = block(x) # [Batch, 32, H', Time]
            # Max Pool frequency to 1 (we only care IF the timbre exists, not where in pitch)
            out = torch.max(out, dim=2)[0] # [Batch, 32, Time]
            features.append(out)

        # Run Temporal Branches (Horizontal)
        for block in self.temporal_convs:
            out = block(x) # [Batch, 32, Freq, Time]
            # Max Pool frequency to 1
            out = torch.max(out, dim=2)[0] # [Batch, 32, Time]
            features.append(out)

        # Concatenate all features: [Batch, 160, Time]
        x = torch.cat(features, dim=1)

        # Run Backbone (1D Convs)
        x = self.backbone(x)

        # Run Attention Head
        x = self.head(x)

        return x
