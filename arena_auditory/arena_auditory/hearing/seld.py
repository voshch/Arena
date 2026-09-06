"""SELDnet front-end (multi-ACCDOA, SALSA-Lite) for the 4-mic planar array.

Pure numpy/torch, no ROS.  Two entry points share one feature and decode path:

* ``events_from_audio``: whole-clip inference, the DCASE per-file path
  (feature extraction, the fitted scaler, the generator's per-file reshape,
  the repo's multi-ACCDOA decode),
* ``SeldStream``: a sliding window of ``feature_sequence_length`` feature
  frames re-run once per label frame, emitting the detections of the label
  frame ``lookahead`` frames before the window end.  The model is not causal
  (self-attention + bidirectional GRU), so a small lookahead trades latency for
  future context.

The model, feature extraction and decode live in ``arena_auditory.hearing.dcase``.
The scaler is an ``.npz`` with ``mean`` and ``scale`` arrays, one per feature
bin, fitted on the checkpoint's training split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from arena_auditory.hearing import dcase

CLASS_NAMES: tuple[str, ...] = ("footstep", "speech")
EPS = 1e-8
FILLER = 1e-6


@dataclass(frozen=True)
class Detection:
    frame: int
    cls: int
    azimuth_rad: float
    elevation_rad: float
    activity: float

    @property
    def sound_type(self) -> str:
        return CLASS_NAMES[self.cls] if self.cls < len(CLASS_NAMES) else str(self.cls)


class SeldFrontend:
    """Loads scaler and checkpoint once; owns feature extraction and decode."""

    def __init__(
        self,
        checkpoint: str | Path,
        scaler: str | Path,
        device: str = "cuda",
        det_thresh: float = 0.5,
    ) -> None:
        import torch

        params = dcase.PARAMS
        self.params = params
        self.feat = dcase.SalsaLiteFeatures(params)
        with np.load(str(scaler)) as npz:
            self.scaler_mean = np.asarray(npz["mean"], dtype=np.float64)
            self.scaler_scale = np.asarray(npz["scale"], dtype=np.float64)
        self.det_thresh = float(det_thresh)
        self.thresh_unify = float(params["thresh_unify"])
        self.nb_classes = int(params["unique_classes"])
        self.nb_mel_bins = int(self.feat.nb_mel_bins)
        self.nb_raw_ch = int(self.feat.nb_channels)
        self.nb_ch = self.nb_raw_ch + (self.nb_raw_ch - 1)
        self.fs = int(params["fs"])
        self.hop_len = int(self.feat.hop_len)
        self.label_hop_len = int(self.feat.label_hop_len)
        self.feature_sequence_length = int(params["feature_sequence_length"])
        self.label_sequence_length = int(params["label_sequence_length"])
        self.window_samples = self.feature_sequence_length * self.hop_len
        if self.scaler_mean.shape != (self.nb_ch * self.nb_mel_bins,):
            raise ValueError(f"scaler has {self.scaler_mean.shape[0]} bins, features have {self.nb_ch * self.nb_mel_bins}")

        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        in_shape = (1, self.nb_ch, self.feature_sequence_length, self.nb_mel_bins)
        out_shape = (1, self.label_sequence_length, self.nb_classes * 3 * 3)
        self.model = dcase.build_model(in_shape, out_shape, params).to(self.device)
        self.model.load_state_dict(torch.load(str(checkpoint), map_location="cpu"))
        self.model.eval()
        self._torch = torch

    def features(self, audio: np.ndarray) -> np.ndarray:
        """``audio`` float (n, 4) in [-1, 1]; returns scaled SALSA-Lite (frames, 7*bins)."""
        audio = np.asarray(audio, dtype=np.float64)[:, : self.nb_raw_ch] + EPS
        nb_feat_frames = int(len(audio) / float(self.hop_len))
        spect = self.feat.spectrogram(audio, nb_feat_frames)
        feat = self.feat.salsalite(spect)
        return (feat - self.scaler_mean) / self.scaler_scale

    def _to_sequences(self, feat: np.ndarray) -> np.ndarray:
        seq = self.feature_sequence_length
        total = feat.shape[0]
        batch = int(np.ceil(total / float(seq)))
        pad = batch * seq - total
        if pad > 0:
            feat = np.concatenate([feat, np.ones((pad, feat.shape[1])) * FILLER], axis=0)
        feat = feat.reshape(batch * seq, self.nb_ch, self.nb_mel_bins)
        feat = feat.reshape(batch, seq, self.nb_ch, self.nb_mel_bins)
        return np.transpose(feat, (0, 2, 1, 3))

    def forward(self, feat: np.ndarray) -> np.ndarray:
        """Scaled features (frames, 7*bins) -> ACCDOA rows (label frames, 3*3*classes)."""
        torch = self._torch
        x = torch.tensor(self._to_sequences(feat)).float().to(self.device)
        with torch.no_grad():
            out = self.model(x).detach().cpu().numpy()
        return out.reshape(-1, out.shape[-1])

    def decode(self, accdoa: np.ndarray, frames: range | None = None) -> list[Detection]:
        """Port of ``train_seldnet.test_epoch``'s per-frame multi-ACCDOA merge loop."""
        nb = self.nb_classes
        sed0, doa0, sed1, doa1, sed2, doa2 = dcase.get_multi_accdoa_labels(accdoa[None, ...], nb)
        sed0, doa0, sed1, doa1, sed2, doa2 = sed0[0], doa0[0], sed1[0], doa1[0], sed2[0], doa2[0]
        if self.det_thresh != 0.5:

            def resed(doa: np.ndarray) -> np.ndarray:
                x, y, z = doa[:, :nb], doa[:, nb : 2 * nb], doa[:, 2 * nb :]
                return np.sqrt(x**2 + y**2 + z**2) > self.det_thresh

            sed0, sed1, sed2 = resed(doa0), resed(doa1), resed(doa2)

        out: list[Detection] = []
        if frames is None:
            frames = range(sed0.shape[0])

        def emit(frame: int, cls: int, row: np.ndarray) -> None:
            xx, yy, zz = float(row[cls]), float(row[cls + nb]), float(row[cls + 2 * nb])
            out.append(
                Detection(
                    frame=frame,
                    cls=cls,
                    azimuth_rad=float(np.arctan2(yy, xx)),
                    elevation_rad=float(np.arctan2(zz, np.hypot(xx, yy))),
                    activity=float(np.sqrt(xx * xx + yy * yy + zz * zz)),
                )
            )

        sim = dcase.determine_similar_location
        tu = self.thresh_unify
        for f in frames:
            for c in range(nb):
                s01 = sim(sed0[f][c], sed1[f][c], doa0[f], doa1[f], c, tu, nb)
                s12 = sim(sed1[f][c], sed2[f][c], doa1[f], doa2[f], c, tu, nb)
                s20 = sim(sed2[f][c], sed0[f][c], doa2[f], doa0[f], c, tu, nb)
                nsim = s01 + s12 + s20
                if nsim == 0:
                    for sed_x, doa_x in ((sed0, doa0), (sed1, doa1), (sed2, doa2)):
                        if sed_x[f][c]:
                            emit(f, c, doa_x[f])
                elif nsim == 1:
                    if s01 and sed2[f][c]:
                        emit(f, c, doa2[f])
                    elif s12 and sed0[f][c]:
                        emit(f, c, doa0[f])
                    elif s20 and sed1[f][c]:
                        emit(f, c, doa1[f])
                    if s01:
                        emit(f, c, (doa0[f] + doa1[f]) / 2)
                    elif s12:
                        emit(f, c, (doa1[f] + doa2[f]) / 2)
                    else:
                        emit(f, c, (doa2[f] + doa0[f]) / 2)
                else:
                    emit(f, c, (doa0[f] + doa1[f] + doa2[f]) / 3)
        return out

    def events_from_audio(self, audio: np.ndarray) -> list[Detection]:
        """Per-file path: one non-overlapping window per 5 s, tail padded with the filler."""
        nb_label_frames = int(len(audio) / float(self.label_hop_len))
        accdoa = self.forward(self.features(audio))
        return [d for d in self.decode(accdoa) if d.frame < nb_label_frames]

    def events_from_wav(self, wav_path: str | Path) -> list[Detection]:
        audio, fs = load_wav(wav_path)
        if fs != self.fs:
            raise ValueError(f"wav sample rate {fs} != expected {self.fs}")
        return self.events_from_audio(audio)


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """int16 wav -> float (n, ch) scaled by 32768 as the DCASE loader does (without eps)."""
    import scipy.io.wavfile as wav

    fs, audio = wav.read(str(path))
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.dtype == np.int16:
        audio = audio / 32768.0
    return np.asarray(audio, dtype=np.float64), int(fs)


class SeldStream:
    """Sliding-window front-end over a live 4-channel float stream.

    Feed blocks with ``push``; ``step`` runs the model on the newest
    ``window_samples`` and returns the detections of label frame
    ``label_sequence_length - 1 - lookahead`` together with the sample index of
    that frame's end (relative to the stream start).  Before a full window has
    arrived the buffer is padded at the front with silence.
    """

    def __init__(self, frontend: SeldFrontend, lookahead: int = 0) -> None:
        if not 0 <= lookahead < frontend.label_sequence_length:
            raise ValueError("lookahead must be within one label sequence")
        self.fe = frontend
        self.lookahead = int(lookahead)
        self._buf = np.zeros((frontend.window_samples, frontend.nb_raw_ch), dtype=np.float64)
        self.samples_seen = 0
        self._last_step_at = -1

    def push(self, block: np.ndarray) -> None:
        block = np.asarray(block, dtype=np.float64)
        n = block.shape[0]
        if n >= self._buf.shape[0]:
            self._buf[:] = block[-self._buf.shape[0] :, : self._buf.shape[1]]
        elif n > 0:
            self._buf = np.roll(self._buf, -n, axis=0)
            self._buf[-n:] = block[:, : self._buf.shape[1]]
        self.samples_seen += n

    def ready(self) -> bool:
        return self.samples_seen > self._last_step_at

    def step(self) -> tuple[list[Detection], int, np.ndarray]:
        """Returns (detections, end sample index of the emitted frame, that frame's samples plus the one before)."""
        fe = self.fe
        self._last_step_at = self.samples_seen
        accdoa = fe.forward(fe.features(self._buf))
        frame = fe.label_sequence_length - 1 - self.lookahead
        dets = fe.decode(accdoa, frames=range(frame, frame + 1))
        end = self.samples_seen - self.lookahead * fe.label_hop_len
        seg = self._buf[max(frame - 1, 0) * fe.label_hop_len : (frame + 1) * fe.label_hop_len]
        return dets, end, seg
