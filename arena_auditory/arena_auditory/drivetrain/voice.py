"""One Arena-embedded drivetrain, rendered incrementally.

A `DrivetrainVoice` is a single drivetrain — one motor, one gearbox, one side of
the robot. You feed it that side's wheel velocity and it hands back audio. Open
one per drivetrain and give each its own velocity; the left/right beating then
comes from the real velocity difference instead of from `lr_mismatch`, which is
the one number in the calibration that was assumed rather than measured.

Everything is incremental and exact. The phase clock is an integer fixed-point
accumulator and every filter carries its state, so the audio does not depend on
how the caller chops up its calls: pushing 4096 frames or 37 frames at a time
gives the identical sample sequence. That property is what makes this safe to
drive from a control loop whose tick rate wobbles.

Requires numpy. Nothing else — `stream.py` is standard library only and can be
lifted out on its own.
"""

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from .spec import JACKAL, DrivetrainSpec

__all__ = ["DrivetrainVoice", "TransferFilter", "design_transfer_fir", "prewarm", "clear_cache", "cache_bytes", "OUTPUT_GAIN"]

#: Default output scaling. The model works in its own units, where a two-voice
#: mix at the Jackal's 2 m/s top speed runs an rms of 3.3 and a crest factor of
#: 14.9 dB. That crest is a real maximum rather than a tail estimate: the source
#: is periodic in distance, so once a run is long enough to traverse the whole
#: noise field — 12 s at 2 m/s — the largest peak has already happened and the
#: measurement stops growing. This gain puts it at -2.7 dBFS. It is the same for
#: every entry point, so levels do not change moving between `render_offline`
#: and a live stream; pass `gain=` to override, and `gain=1/count` recovers the
#: reference renderer's raw units.
OUTPUT_GAIN = 0.04

TWO_PI = 2.0 * math.pi
Q32 = 1 << 32
ORDER_DEN = 1024  # orders are carried as integer m / ORDER_DEN
_MOD = ORDER_DEN * Q32  # phase modulus; see the note on exactness below
_CHUNK = 4096  # internal subdivision, bounds peak memory only


# ----------------------------------------------------------------------------
# transfer H(f): a linear-phase FIR, designed here so scipy is not needed
# ----------------------------------------------------------------------------
def _firwin2(numtaps: int, freq: ArrayLike, gain: ArrayLike) -> NDArray[np.float64]:
    """Frequency-sampling FIR design. Matches scipy.signal.firwin2 to ~1e-16.

    `freq` is normalised to Nyquist = 1 and must start at 0 and end at 1.
    """
    nfreqs = 1 + 2 ** int(math.ceil(math.log2(numtaps)))
    x = np.linspace(0.0, 1.0, nfreqs)
    fx = np.interp(x, freq, gain)
    shift = np.exp(-(numtaps - 1) / 2.0 * 1j * np.pi * x)
    return np.fft.irfft(fx * shift)[:numtaps] * np.hamming(numtaps)


def design_transfer_fir(spec: DrivetrainSpec) -> NDArray[np.float64] | None:
    """FIR taps for the fitted chassis + room + mic transfer of `spec`."""
    if len(spec.transfer_hz) < 2:
        return None
    ny = spec.sample_rate / 2.0
    fh = np.asarray(spec.transfer_hz, float)
    hd = np.asarray(spec.transfer_db, float)
    grid = np.concatenate([[0.0], np.geomspace(20.0, ny * 0.999, 400), [ny]])
    g = np.interp(np.log(np.maximum(grid, 1e-9)), np.log(fh), hd, left=hd[0], right=hd[-1])
    amp = 10.0 ** ((g - g.max()) / 20.0)
    amp[0] = amp[1]
    return _firwin2(spec.transfer_taps, grid / ny, amp)


class TransferFilter:
    """Stateful FIR. Use one per voice, or one on the mix bus.

    H(f) is a linear filter, so filtering each voice and summing gives the same
    result as summing and filtering once. Per-voice is the default because it
    keeps voices independent; a shared bus filter is cheaper for many voices.
    """

    def __init__(self, taps: ArrayLike) -> None:
        self.taps = np.asarray(taps, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        self._tail = np.zeros(len(self.taps) - 1)

    @property
    def latency(self) -> int:
        """Group delay in samples (the FIR is linear phase)."""
        return (len(self.taps) - 1) // 2

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        z = np.concatenate([self._tail, x])
        self._tail = z[len(z) - (len(self.taps) - 1) :].copy()
        return np.convolve(z, self.taps, mode="valid")


# ----------------------------------------------------------------------------
# generated tables, shared between voices
# ----------------------------------------------------------------------------
class _Tables:
    __slots__ = ("m", "a", "ph", "order", "levels", "dx0", "field_m", "nbytes")


def _smooth_len(n: int) -> int:
    """Largest even 5-smooth number <= n.

    The field length falls where it falls, and at the shipped settings that is
    2 * 3 * 692761 — one large prime, so numpy's FFT drops to Bluestein and the
    build takes seconds. Rounding to a 5-smooth length costs 0.23% of the loop
    period and buys a ~10x faster build.
    """
    best, p = 2, 1
    while p <= n // 2:
        q = p
        while q <= n // 2:
            r = q
            while r <= n // 2:
                best = max(best, r)
                r *= 5
            q *= 3
        p *= 2
    return 2 * best


_CACHE: dict[tuple[DrivetrainSpec, int, str], _Tables] = {}


def _build(spec: DrivetrainSpec, seed: int, dtype: np.dtype) -> _Tables:
    """Comb table and mipmapped distance-domain noise field.

    The random draws happen in the same order as the reference renderer, so a
    given (spec, seed) reproduces its output exactly.
    """
    t = _Tables()
    rng = np.random.default_rng(seed + 2027)

    m_list, a_list, ph_list = [], [], []
    for k, db in enumerate(spec.partials_db, start=1):
        m_list.append(k * ORDER_DEN)
        a_list.append(10.0 ** ((db + spec.tonal_gain_db) / 20.0))
        ph_list.append(rng.uniform(0, TWO_PI))
    t.m = np.array(m_list, dtype=np.int64)
    t.a = np.array(a_list, dtype=np.float64)
    t.ph = np.array(ph_list, dtype=np.float64)
    t.order = t.m / ORDER_DEN

    t.levels, t.dx0, t.field_m = [], 0.0, spec.field_metres
    if len(spec.order_grid) >= 2:
        og = np.asarray(spec.order_grid, float)
        qd = np.asarray(spec.order_db, float)
        lo = np.log(og)
        sl_hi = (qd[-1] - qd[-2]) / (lo[-1] - lo[-2])
        sl_lo = (qd[1] - qd[0]) / (lo[1] - lo[0])
        nu_max = spec.order_max * spec.K / TWO_PI  # cycles per metre
        t.dx0 = 1.0 / (2.0 * nu_max)
        n = int(spec.field_metres / t.dx0) // 2 * 2
        if spec.field_smooth:
            # take the field length from the table length rather than the other
            # way round, so the read position wraps exactly on the table and the
            # loop point is seamless
            n = _smooth_len(n)
            t.field_m = n * t.dx0
        w = rng.standard_normal(n)
        nu = np.fft.rfftfreq(n, t.dx0)
        lc = np.log(np.maximum(TWO_PI * nu / max(spec.K, 1e-9), 1e-9))
        q = np.interp(lc, lo, qd)
        # extrapolate on the terminal log-log slope; holding it flat past the
        # fitted range dumps energy into orders that are in band at low speed
        q = np.where(lc > lo[-1], qd[-1] + sl_hi * (lc - lo[-1]), q)
        q = np.where(lc < lo[0], qd[0] + sl_lo * (lc - lo[0]), q)
        amp = 10.0 ** (q / 20.0)
        amp[0] = 0.0
        base = np.fft.irfft(np.fft.rfft(w) * amp, n=n)
        base = base / (np.std(base) + 1e-20)
        # mipmap: every level halves the pitch count and the bandwidth, so the
        # level selected at a given speed has already removed whatever would
        # have aliased at that read rate
        cur = base
        for _ in range(spec.field_levels):
            t.levels.append(np.asarray(cur, dtype=dtype))
            if len(cur) < 64:
                break
            cf = np.fft.rfft(cur)
            cf[len(cf) // 2 :] = 0.0
            cur = np.fft.irfft(cf, n=len(cur))[::2].copy()
    t.nbytes = sum(lv.nbytes for lv in t.levels)
    return t


def _tables(spec: DrivetrainSpec, seed: int, dtype: DTypeLike) -> _Tables:
    key = (spec, int(seed), np.dtype(dtype).str)
    t = _CACHE.get(key)
    if t is None:
        t = _CACHE[key] = _build(spec, seed, np.dtype(dtype))
    return t


def prewarm(spec: DrivetrainSpec = JACKAL, seed: int = 0, field_dtype: DTypeLike = "float32") -> int:
    """Build and cache the noise field ahead of time.

    The field is a few million samples; generating it takes on the order of a
    second. Call this before entering a real-time loop so the first render does
    not stall. Returns the number of bytes now held.
    """
    return _tables(spec, seed, field_dtype).nbytes


def clear_cache() -> None:
    """Drop all cached noise fields."""
    _CACHE.clear()


def cache_bytes() -> int:
    """Total memory held by cached noise fields."""
    return sum(t.nbytes for t in _CACHE.values())


# ----------------------------------------------------------------------------
# the voice
# ----------------------------------------------------------------------------
class DrivetrainVoice:
    """A single drivetrain rendered incrementally.

    Args:
        spec:   a `DrivetrainSpec`; defaults to the shipped Jackal calibration.
        index:  which drivetrain this is. Sets the starting phase, the starting
                position in the noise field, and the sign of the `lr_mismatch`
                offset applied to K, so two voices never sit exactly on top of
                one another.
        count:  how many drivetrains are being mixed. Only sets the default
                gain (1/count). Defaults to `spec.n_drivetrains`.
        seed:   picks the noise field and the comb phases. Voices sharing a
                (spec, seed) share one field, so give the sides the same seed
                unless you want two independent fields and the memory cost.
        k:      override the phase-clock constant for this side, in rad/m. Pass
                a per-side measured value if you have one.
        transfer: apply H(f) here. Set False on every voice and run one
                `TransferFilter` on the mix instead if you prefer.
        gain:   linear output gain; defaults to `OUTPUT_GAIN / count`, which
                keeps the loudest speed the robot can reach inside full scale.
                Pass `1 / count` for the reference renderer's raw units.
        field_dtype: storage for the noise field. float32 halves the memory for
                an error 140 dB down; float64 reproduces the reference exactly.

    Not thread-safe: one producer per voice. `PcmStream` handles the handoff to
    a consumer thread.
    """

    def __init__(self, spec: DrivetrainSpec | Mapping[str, Any] = JACKAL, index: int = 0, count: int | None = None, seed: int = 0, k: float | None = None, transfer: bool = True, gain: float | None = None, field_dtype: DTypeLike = "float32") -> None:
        if not isinstance(spec, DrivetrainSpec):
            spec = DrivetrainSpec.from_dict(dict(spec))
        self.spec = spec
        self.seed = int(seed)
        self.index = int(index)
        self.count = int(spec.n_drivetrains if count is None else count)
        self.sample_rate = int(spec.sample_rate)
        self.k = float(self._default_k() if k is None else k)
        self.gain = float(OUTPUT_GAIN / max(self.count, 1) if gain is None else gain)

        t = _tables(spec, seed, field_dtype)
        self._m, self._a, self._ph, self._order = t.m, t.a, t.ph, t.order
        self._levels, self._dx0 = t.levels, t.dx0
        #: effective noise-field length in metres, and so its loop period
        self.field_metres = t.field_m

        taps = design_transfer_fir(spec) if transfer else None
        self.transfer = TransferFilter(taps) if taps is not None else None
        self.reset()

    def _default_k(self) -> float:
        s = self.spec
        if self.count < 2 or s.lr_mismatch == 0.0:
            return s.K
        return s.K * (1.0 + (-0.5 if self.index % 2 == 0 else 0.5) * s.lr_mismatch)

    # -- state -------------------------------------------------------------
    def reset(self) -> None:
        """Rewind to the start: phase, field position, gate and filter state."""
        i = self.index
        self._psi_q = int((0.31 * i % 1.0) * Q32) % _MOD
        self._x_q = int((0.41 * i % 1.0) * Q32) % Q32
        self._gate = 0.0
        self._resid = 0.0
        self.frames_rendered = 0
        if self.transfer is not None:
            self.transfer.reset()

    @property
    def distance(self) -> float:
        """Metres of wheel travel accumulated, modulo the noise field length."""
        return self._x_q * self.field_metres / Q32

    def loop_seconds(self, v: float) -> float:
        """How long the noise field runs before repeating, at speed `v`."""
        return self.field_metres / max(abs(v), 1e-12)

    # -- rendering ----------------------------------------------------------
    def render(
        self,
        v: ArrayLike,
        frames: int | None = None,
        tau: ArrayLike = 0.0,
        *,
        frequency_scale: float = 1.0,
        tonal_gain_db: float = 0.0,
        broadband_gain_db: float = 0.0,
        speed_exponent: float | None = None,
    ) -> NDArray[np.float64]:
        """Render the next block and return it as float64 in [-1, 1]-ish.

        Args:
            v:      speed in m/s. A scalar is held for the whole block; a
                    sequence gives a per-sample ramp. Negative means reverse,
                    which sounds the same and runs the clock backwards.
            frames: block length. Required when `v` is a scalar.
            tau:    normalised torque, scalar or per-sample, used only when
                    `spec.load_depth` is non-zero.
            frequency_scale: pitch multiplier without changing loudness.
            tonal_gain_db: runtime trim for the periodic gear-mesh layer.
            broadband_gain_db: runtime trim for the noise layer.
            speed_exponent: runtime override for velocity-to-level response.
        """
        v = np.asarray(v, dtype=np.float64)
        if v.ndim == 0:
            if frames is None:
                raise ValueError("frames is required when v is a scalar")
            v = np.full(int(frames), float(v))
        elif v.ndim != 1:
            raise ValueError("v must be a scalar or a 1-D sequence")
        elif frames is not None and len(v) != int(frames):
            raise ValueError(f"got {len(v)} speeds but frames={frames}")
        n = len(v)
        if n == 0:
            return np.zeros(0)
        tau = np.broadcast_to(np.asarray(tau, dtype=np.float64), (n,))
        frequency_scale = float(frequency_scale)
        tonal_scale = 10.0 ** (float(tonal_gain_db) / 20.0)
        broadband_scale = 10.0 ** (float(broadband_gain_db) / 20.0)
        velocity_exponent = self.spec.speed_exponent if speed_exponent is None else float(speed_exponent)

        dt = 1.0 / self.sample_rate
        out = np.empty(n)
        for a in range(0, n, _CHUNK):
            b = min(a + _CHUNK, n)
            out[a:b] = self._chunk(
                v[a:b],
                tau[a:b],
                dt,
                frequency_scale,
                tonal_scale,
                broadband_scale,
                velocity_exponent,
            )
        if self.transfer is not None:
            out = self.transfer(out)
        self.frames_rendered += n
        return out * self.gain

    def render_seconds(self, v: ArrayLike, seconds: float, tau: ArrayLike = 0.0, **render_options: float | None) -> NDArray[np.float64]:
        """Render `seconds` worth of audio, carrying the fractional remainder.

        Use this when the control loop ticks at a rate that is not a divisor of
        the sample rate; the leftover fraction of a frame is carried over so the
        audio clock never drifts from the sim clock.
        """
        want = seconds * self.sample_rate + self._resid
        frames = int(want)
        self._resid = want - frames
        return self.render(v, frames, tau, **render_options)

    # -- internals ----------------------------------------------------------
    def _chunk(
        self,
        v: NDArray[np.float64],
        tau: NDArray[np.float64],
        dt: float,
        frequency_scale: float,
        tonal_scale: float,
        broadband_scale: float,
        speed_exponent: float,
    ) -> NDArray[np.float64]:
        s = self.spec
        n = len(v)
        av = np.abs(v)
        pitch_v = v * frequency_scale
        pitch_av = np.abs(pitch_v)
        lim = s.alias_limit * self.sample_rate
        acc = np.zeros(n)

        # phase clock. Integer addition is exact and associative, so this is
        # identical however the caller blocks its calls. Orders are rationals
        # m/ORDER_DEN and the accumulator is kept modulo ORDER_DEN*Q32, so
        # multiplying by m stays exact and wraps correctly; wrapping to [0, 2pi)
        # would be wrong for non-integer orders.
        dq = np.rint(pitch_v * dt * self.k / TWO_PI * Q32).astype(np.int64)
        psi_q = (self._psi_q + np.cumsum(dq)) % _MOD
        self._psi_q = int(psi_q[-1])

        # ---- comb: one explicit sinusoid per integer order ----
        if len(self._m):
            f_mesh = self.k * pitch_av / TWO_PI
            # pre-select on the slowest sample in the chunk so anything live
            # anywhere in the chunk is kept; the per-sample taper below then
            # silences it wherever it would exceed the limit. Components dropped
            # here would have tapered to zero across the whole chunk anyway,
            # which is why the choice of chunk size cannot change the output.
            cand = self._order * float(f_mesh.min()) < lim
            if np.any(cand):
                m = self._m[cand]
                ang = (m[:, None] * psi_q[None, :]) % _MOD
                side = self._a[cand][:, None] * np.sin(ang * (TWO_PI / _MOD) + self._ph[cand][:, None])
                # taper into the limit rather than switching: a hard switch
                # clicks, and the click is broadband
                fk = self._order[cand][:, None] * f_mesh[None, :]
                side *= np.clip((lim - fk) / (0.06 * lim), 0.0, 1.0)
                acc += side.sum(axis=0) * tonal_scale

        # ---- broadband: mipmapped distance field, read at dx/dt ----
        if self._levels:
            dxq = np.rint(pitch_v * dt / self.field_metres * Q32).astype(np.int64)
            x_q = (self._x_q + np.cumsum(dxq)) % Q32
            self._x_q = int(x_q[-1])
            pos0 = x_q * (self.field_metres / Q32) / self._dx0  # level-0 samples
            # choose the level so the read rate stays below 1/headroom, i.e.
            # always interpolating, never decimating
            lf = np.log2(np.maximum(pitch_av * dt / self._dx0, 1e-12) * s.field_headroom)
            lf = np.clip(lf, 0.0, len(self._levels) - 1.000001)
            l0 = np.floor(lf).astype(np.int64)
            fr = lf - l0
            bb = np.zeros(n)
            for lv in np.unique(np.concatenate([l0, l0 + 1])):
                if lv >= len(self._levels):
                    continue
                sel = (l0 == lv) | (l0 + 1 == lv)
                if not np.any(sel):
                    continue
                wgt = np.where(l0[sel] == lv, 1.0 - fr[sel], fr[sel])
                bb[sel] += wgt * self._read(int(lv), pos0[sel])
            acc += bb * 10.0 ** (s.broadband_gain_db / 20.0) * broadband_scale

        # the 1/count mix scaling lives in self.gain, applied once in render()
        g = (av / max(s.v_ref, 1e-9)) ** speed_exponent
        g = g * (1.0 + s.load_depth * tau)
        return acc * g * self._gate_env(av, dt)

    def _read(self, level: int, pos0: NDArray[np.float64]) -> NDArray[np.float64]:
        """Catmull-Rom read of one mipmap level. `pos0` is in level-0 samples."""
        tab = self._levels[level]
        n = len(tab)
        p = pos0 / (1 << level)
        i0 = np.floor(p).astype(np.int64)
        fr = p - i0
        i0 %= n
        a = tab[(i0 - 1) % n].astype(np.float64, copy=False)
        b = tab[i0].astype(np.float64, copy=False)
        c = tab[(i0 + 1) % n].astype(np.float64, copy=False)
        d = tab[(i0 + 2) % n].astype(np.float64, copy=False)
        a0 = -0.5 * a + 1.5 * b - 1.5 * c + 0.5 * d
        a1 = a - 2.5 * b + 2.0 * c - 0.5 * d
        a2 = -0.5 * a + 0.5 * c
        return ((a0 * fr + a1) * fr + a2) * fr + b

    def _gate_env(self, av: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
        """Static/driving crossfade, one-pole, state carried across calls.

        Evaluated in closed form per run of constant input rather than by
        recursion: the gate input is binary and changes rarely, so this is both
        exact and vectorised.
        """
        s = self.spec
        on = (av > s.v_static * (1.0 + s.hysteresis)).astype(np.float64)
        a = math.exp(-dt / max(s.crossfade_s, 1e-6))
        y = np.empty(len(on))
        prev = self._gate
        edges = np.flatnonzero(np.diff(on)) + 1
        starts = np.concatenate([[0], edges])
        ends = np.concatenate([edges, [len(on)]])
        for i, j in zip(starts, ends, strict=True):
            u = on[i]
            y[i:j] = u + (prev - u) * a ** np.arange(1, j - i + 1)
            prev = y[j - 1]
        self._gate = float(prev)
        return y
