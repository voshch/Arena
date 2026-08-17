"""The Arena-embedded parameter set for one drivetrain.

`DrivetrainSpec` is frozen and hashable so that voices sharing a spec also share
their (large) generated noise field — see `voice._tables`.
"""
import json
from collections.abc import Iterable, Mapping
from typing import Any

import attrs

from ._jackal import JACKAL_PARAMS

__all__ = ["DrivetrainSpec", "JACKAL"]


def _t(x: Iterable[float]) -> tuple[float, ...]:
    return tuple(float(v) for v in x)


@attrs.frozen
class DrivetrainSpec:
    """Calibrated parameters for the order-domain drivetrain model.

    The model is: a source defined on the ORDER axis (so it translates with
    speed), read off a phase clock driven by distance travelled, passed once
    through a transfer H(f) that is fixed in FREQUENCY.

        psi(t) = K * x(t)                      phase clock, radians
        source = comb(partials_db) + broadband(order_grid, order_db)
        out    = H(f) * source * (|v| / v_ref) ** speed_exponent

    `field_smooth` rounds the noise field down to a 5-smooth number of samples
    (0.23% shorter at the shipped settings). That makes it build about ten times
    faster, because the natural length has a large prime factor and the FFT falls
    back to Bluestein, and it makes the field loop seamlessly, because the read
    position then wraps exactly on the table length. Set it False to reproduce
    the reference renderer's field sample for sample.
    """

    # --- phase clock ---------------------------------------------------
    K: float = 2720.0                 # rad/m; lumped gear teeth * ratio / wheel radius
    # --- order-domain source: the gear-mesh comb -----------------------
    partials_db: tuple[float, ...] = attrs.field(converter=_t, default=())  # S_k at integer orders k = 1, 2, 3, ...
    tonal_gain_db: float = 0.0
    # --- order-domain source: the broadband ----------------------------
    order_grid: tuple[float, ...] = attrs.field(converter=_t, default=())  # order axis for the broadband PSD
    order_db: tuple[float, ...] = attrs.field(converter=_t, default=())  # broadband PSD, dB, at those orders
    order_max: float = 200.0          # highest order the noise field resolves
    field_metres: float = 24.0        # noise field length; also its loop period
    field_levels: int = 6             # mipmap depth
    field_headroom: float = 2.0       # keep the field read rate under 1/this
    field_smooth: bool = True         # see note below
    broadband_gain_db: float = 0.0
    # --- fixed transfer -------------------------------------------------
    transfer_hz: tuple[float, ...] = attrs.field(converter=_t, default=())
    transfer_db: tuple[float, ...] = attrs.field(converter=_t, default=())
    transfer_taps: int = 513          # linear-phase FIR length
    # --- level ----------------------------------------------------------
    speed_exponent: float = 1.0
    v_ref: float = 1.0
    load_depth: float = 0.0           # fractional level change per unit torque
    # --- two drivetrains ------------------------------------------------
    n_drivetrains: int = 2            # only sets the default per-voice gain and K
    lr_mismatch: float = 0.006        # fractional K spread between sides
    # --- regimes ---------------------------------------------------------
    v_static: float = 0.02            # below this the drivetrain is silent
    hysteresis: float = 0.15
    crossfade_s: float = 0.040        # static <-> driving fade time constant
    # --- housekeeping -----------------------------------------------------
    sample_rate: int = 44100
    alias_limit: float = 0.45         # comb partials are faded out below this * fs

    def __attrs_post_init__(self) -> None:
        if len(self.order_grid) != len(self.order_db):
            raise ValueError("order_grid and order_db must be the same length")
        if len(self.transfer_hz) != len(self.transfer_db):
            raise ValueError("transfer_hz and transfer_db must be the same length")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.transfer_taps < 1 or self.transfer_taps % 2 == 0:
            raise ValueError("transfer_taps must be a positive odd number")
        if not 0.0 < self.alias_limit < 0.5:
            raise ValueError("alias_limit must be in (0, 0.5)")
        if self.field_levels < 1:
            raise ValueError("field_levels must be >= 1")

    # -- conversions ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return attrs.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DrivetrainSpec":
        """Build from a dict, ignoring keys this version does not use."""
        known = {f.name for f in attrs.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_json(self, **kw: Any) -> str:
        kw.setdefault("indent", 2)
        return json.dumps(self.to_dict(), **kw)

    @classmethod
    def from_json(cls, text: str | bytes) -> "DrivetrainSpec":
        return cls.from_dict(json.loads(text))

    def replace(self, **kw: Any) -> "DrivetrainSpec":
        return attrs.evolve(self, **kw)

    # -- derived ----------------------------------------------------------
    def f_mesh(self, v: float) -> float:
        """Fundamental gear-mesh frequency, Hz, at speed v (m/s)."""
        return self.K * v / (2.0 * 3.141592653589793)

    def field_seconds(self, v: float) -> float:
        """Nominal loop period of the noise field, seconds, at speed v.

        `field_smooth` shortens the field by 0.23%; `DrivetrainVoice.loop_seconds`
        gives the effective figure.
        """
        return self.field_metres / max(abs(v), 1e-12)


#: The shipped Clearpath Jackal calibration (see output/A16_v2_spec.json).
JACKAL = DrivetrainSpec(**JACKAL_PARAMS)
