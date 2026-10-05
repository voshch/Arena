"""The slice of the DCASE 2023 SELD baseline the front-end needs, for the 4-mic planar array.

Adapted from https://github.com/sharathadavanne/seld-dcase2023 (MIT License,
Copyright (c) 2023 Sharath Adavanne, Archontis Politis, Parthasaarathy Sudarsanam):
the SALSA-Lite feature extraction of ``cls_feature_class.FeatureClass``, the
``SeldModel`` architecture, and the multi-ACCDOA decode helpers of
``train_seldnet``. Numerics are unchanged so the published checkpoint loads
and infers exactly as under the original code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

PARAMS: dict = {
    "fs": 16000,
    "hop_len_s": 0.02,
    "label_hop_len_s": 0.1,
    "fmin_doa_salsalite": 50,
    "fmax_doa_salsalite": 2000,
    "fmax_spectra_salsalite": 7500,
    "thresh_unify": 15,
    "label_sequence_length": 50,
    "dropout_rate": 0.05,
    "nb_cnn2d_filt": 64,
    "f_pool_size": [4, 4, 2],
    "nb_heads": 8,
    "nb_self_attn_layers": 2,
    "nb_rnn_layers": 2,
    "rnn_size": 128,
    "nb_fnn_layers": 1,
    "fnn_size": 128,
    "unique_classes": 2,
}
_resolution = int(PARAMS["label_hop_len_s"] // PARAMS["hop_len_s"])
PARAMS["feature_sequence_length"] = PARAMS["label_sequence_length"] * _resolution
PARAMS["t_pool_size"] = [_resolution, 1, 1]


class SalsaLiteFeatures:
    """STFT plus SALSA-Lite spatial features for a 4-channel microphone array."""

    def __init__(self, params: dict = PARAMS) -> None:
        self.fs = int(params["fs"])
        self.hop_len = int(self.fs * params["hop_len_s"])
        self.label_hop_len = int(self.fs * params["label_hop_len_s"])
        self.win_len = 2 * self.hop_len
        self.nfft = 2 ** (self.win_len - 1).bit_length()
        self.nb_channels = 4

        self.lower_bin = max(1, int(np.floor(params["fmin_doa_salsalite"] * self.nfft / float(self.fs))))
        self.upper_bin = int(np.floor(min(params["fmax_doa_salsalite"], self.fs // 2) * self.nfft / float(self.fs)))
        c = 343
        self.delta = 2 * np.pi * self.fs / (self.nfft * c)
        freq_vector = np.arange(self.nfft // 2 + 1)
        freq_vector[0] = 1
        self.freq_vector = freq_vector[None, :, None]
        self.cutoff_bin = int(np.floor(params["fmax_spectra_salsalite"] * self.nfft / float(self.fs)))
        if self.upper_bin > self.cutoff_bin:
            raise ValueError(f"doa upper bin {self.upper_bin} exceeds spectrogram cutoff bin {self.cutoff_bin}")
        self.nb_mel_bins = self.cutoff_bin - self.lower_bin

    def spectrogram(self, audio: np.ndarray, nb_frames: int) -> np.ndarray:
        import librosa

        spectra = []
        for ch in range(audio.shape[1]):
            stft = librosa.core.stft(np.asfortranarray(audio[:, ch]), n_fft=self.nfft, hop_length=self.hop_len, win_length=self.win_len, window="hann")
            spectra.append(stft[:, :nb_frames])
        return np.array(spectra).T

    def salsalite(self, linear_spectra: np.ndarray) -> np.ndarray:
        import librosa

        phase = np.angle(linear_spectra[:, :, 1:] * np.conj(linear_spectra[:, :, 0, None]))
        phase = phase / (self.delta * self.freq_vector)
        phase = phase[:, self.lower_bin : self.cutoff_bin, :]
        phase[:, self.upper_bin :, :] = 0
        phase = phase.transpose((0, 2, 1)).reshape((phase.shape[0], -1))

        power = np.abs(linear_spectra) ** 2
        for ch in range(power.shape[-1]):
            power[:, :, ch] = librosa.power_to_db(power[:, :, ch], ref=1.0, amin=1e-10, top_db=None)
        power = power[:, self.lower_bin : self.cutoff_bin, :]
        power = power.transpose((0, 2, 1)).reshape((power.shape[0], -1))
        return np.concatenate((power, phase), axis=-1)


def build_model(in_shape: tuple[int, ...], out_shape: tuple[int, ...], params: dict = PARAMS) -> torch.nn.Module:
    """The SELDnet CRNN with self-attention, built lazily so torch is only imported when used."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    class ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.bn = nn.BatchNorm2d(out_channels)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.relu(self.bn(self.conv(x)))

    class SeldModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv_block_list = nn.ModuleList()
            for conv_cnt in range(len(params["f_pool_size"])):
                self.conv_block_list.append(ConvBlock(params["nb_cnn2d_filt"] if conv_cnt else in_shape[1], params["nb_cnn2d_filt"]))
                self.conv_block_list.append(nn.MaxPool2d((params["t_pool_size"][conv_cnt], params["f_pool_size"][conv_cnt])))
                self.conv_block_list.append(nn.Dropout2d(p=params["dropout_rate"]))

            gru_input_dim = params["nb_cnn2d_filt"] * int(np.floor(in_shape[-1] / np.prod(params["f_pool_size"])))
            self.gru = nn.GRU(input_size=gru_input_dim, hidden_size=params["rnn_size"], num_layers=params["nb_rnn_layers"], batch_first=True, dropout=params["dropout_rate"], bidirectional=True)

            self.mhsa_block_list = nn.ModuleList()
            self.layer_norm_list = nn.ModuleList()
            for _ in range(params["nb_self_attn_layers"]):
                self.mhsa_block_list.append(nn.MultiheadAttention(embed_dim=params["rnn_size"], num_heads=params["nb_heads"], dropout=params["dropout_rate"], batch_first=True))
                self.layer_norm_list.append(nn.LayerNorm(params["rnn_size"]))

            self.fnn_list = nn.ModuleList()
            for fc_cnt in range(params["nb_fnn_layers"]):
                self.fnn_list.append(nn.Linear(params["fnn_size"] if fc_cnt else params["rnn_size"], params["fnn_size"], bias=True))
            self.fnn_list.append(nn.Linear(params["fnn_size"] if params["nb_fnn_layers"] else params["rnn_size"], out_shape[-1], bias=True))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for block in self.conv_block_list:
                x = block(x)
            x = x.transpose(1, 2).contiguous()
            x = x.view(x.shape[0], x.shape[1], -1).contiguous()
            (x, _) = self.gru(x)
            x = torch.tanh(x)
            x = x[:, :, x.shape[-1] // 2 :] * x[:, :, : x.shape[-1] // 2]
            for mhsa, norm in zip(self.mhsa_block_list, self.layer_norm_list, strict=True):
                x_in = x
                x, _ = mhsa(x_in, x_in, x_in)
                x = norm(x + x_in)
            for fnn in self.fnn_list[:-1]:
                x = fnn(x)
            return torch.tanh(self.fnn_list[-1](x))

    return SeldModel()


def get_multi_accdoa_labels(accdoa_in: np.ndarray, nb_classes: int) -> tuple[np.ndarray, ...]:
    """(batch, frames, 3 tracks * 3 axes * classes) -> per-track (sed, doa) pairs."""
    out = []
    for track in range(3):
        doa = accdoa_in[:, :, 3 * track * nb_classes : 3 * (track + 1) * nb_classes]
        x, y, z = doa[:, :, :nb_classes], doa[:, :, nb_classes : 2 * nb_classes], doa[:, :, 2 * nb_classes :]
        out.append(np.sqrt(x**2 + y**2 + z**2) > 0.5)
        out.append(doa)
    return tuple(out)


def distance_between_cartesian_coordinates(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> float:
    """Angular distance in degrees between two direction vectors."""
    n1 = np.sqrt(x1**2 + y1**2 + z1**2 + 1e-10)
    n2 = np.sqrt(x2**2 + y2**2 + z2**2 + 1e-10)
    x1, y1, z1, x2, y2, z2 = x1 / n1, y1 / n1, z1 / n1, x2 / n2, y2 / n2, z2 / n2
    dist = x1 * x2 + y1 * y2 + z1 * z2
    dist = np.clip(dist, -1, 1)
    return float(np.arccos(dist) * 180 / np.pi)


def determine_similar_location(sed0: bool, sed1: bool, doa0: np.ndarray, doa1: np.ndarray, cls: int, thresh_unify: float, nb_classes: int) -> int:
    if sed0 == 1 and sed1 == 1:
        close = distance_between_cartesian_coordinates(doa0[cls], doa0[cls + nb_classes], doa0[cls + 2 * nb_classes], doa1[cls], doa1[cls + nb_classes], doa1[cls + 2 * nb_classes]) < thresh_unify
        return 1 if close else 0
    return 0
