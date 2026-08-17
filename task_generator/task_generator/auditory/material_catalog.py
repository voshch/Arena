from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import attrs
import yaml

DEFAULT_OCTAVE_BANDS_HZ = (
    125,
    250,
    500,
    1000,
    2000,
    4000,
)


@attrs.frozen
class AcousticMaterial:
    """Frequency-dependent acoustic properties for one material.

    Absorption and scattering values are energy coefficients in the
    inclusive range [0, 1].

    Transmission loss is expressed in decibels and must be non-negative.
    """

    material_id: str
    canonical_name: str
    center_frequencies_hz: tuple[int, ...]
    absorption: tuple[float, ...]
    transmission_loss_db: tuple[float, ...]
    scattering: tuple[float, ...]
    used_default: bool = False

    @property
    def band_count(self) -> int:
        return len(self.center_frequencies_hz)

    @property
    def mean_absorption(self) -> float:
        if not self.absorption:
            return 0.0
        return sum(self.absorption) / len(self.absorption)

    @property
    def mean_transmission_loss_db(self) -> float:
        if not self.transmission_loss_db:
            return 0.0
        return (
            sum(self.transmission_loss_db)
            / len(self.transmission_loss_db)
        )

    @property
    def mean_scattering(self) -> float:
        if not self.scattering:
            return 0.0
        return sum(self.scattering) / len(self.scattering)


class AcousticMaterialCatalog:
    """Loads and validates acoustic materials from YAML.

    Expected YAML structure:

        octave_bands_hz: [125, 250, 500, 1000, 2000, 4000]

        defaults:
          canonical_name: default
          absorption: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
          transmission_loss_db: [20, 20, 20, 20, 20, 20]
          scattering: 0.1

        materials:
          Concrete_Smooth:
            canonical_name: concrete
            absorption: [...]
            transmission_loss_db: [...]
            scattering: 0.05

    ``scattering`` may be either a scalar or one value per octave band.
    Absorption and transmission loss may also be scalar values; scalar
    values are expanded across every configured octave band.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

        raw = self._load_yaml(self._path)

        self._center_frequencies_hz = (
            self._parse_center_frequencies(
                raw.get(
                    "octave_bands_hz",
                    DEFAULT_OCTAVE_BANDS_HZ,
                )
            )
        )

        default_entry = raw.get("defaults")
        if not isinstance(default_entry, Mapping):
            raise ValueError(
                f"{self._path}: expected 'defaults' to be a mapping"
            )

        self._default_material = self._parse_material(
            material_id="default",
            entry=default_entry,
            used_default=True,
        )

        raw_materials = raw.get("materials", {})
        if raw_materials is None:
            raw_materials = {}

        if not isinstance(raw_materials, Mapping):
            raise ValueError(
                f"{self._path}: expected 'materials' to be a mapping"
            )

        materials: dict[str, AcousticMaterial] = {}

        for raw_material_id, raw_entry in raw_materials.items():
            material_id = str(raw_material_id).strip()

            if not material_id:
                raise ValueError(
                    f"{self._path}: material ID cannot be empty"
                )

            if material_id in materials:
                raise ValueError(
                    f"{self._path}: duplicate material "
                    f"{material_id!r}"
                )

            if not isinstance(raw_entry, Mapping):
                raise ValueError(
                    f"{self._path}: material {material_id!r} "
                    "must be a mapping"
                )

            materials[material_id] = self._parse_material(
                material_id=material_id,
                entry=raw_entry,
                used_default=False,
            )

        self._materials = materials

    @property
    def path(self) -> Path:
        return self._path

    @property
    def center_frequencies_hz(self) -> tuple[int, ...]:
        return self._center_frequencies_hz

    @property
    def default_material(self) -> AcousticMaterial:
        return self._default_material

    def contains(self, material_id: str) -> bool:
        return str(material_id).strip() in self._materials

    def known_material_ids(self) -> frozenset[str]:
        return frozenset(self._materials)

    def get(self, material_id: str) -> AcousticMaterial:
        """Return a material or a marked default-material fallback.

        Unknown material IDs preserve the requested ID so callers can
        report which Arena material was missing from the acoustic catalog.
        The returned material has ``used_default=True``.
        """

        requested_id = str(material_id).strip()

        if not requested_id:
            requested_id = "default"

        material = self._materials.get(requested_id)
        if material is not None:
            return material

        return attrs.evolve(
            self._default_material,
            material_id=requested_id,
            canonical_name=requested_id,
            used_default=True,
        )

    def require(self, material_id: str) -> AcousticMaterial:
        """Return a known material and reject unknown IDs."""

        requested_id = str(material_id).strip()

        if not requested_id:
            raise KeyError("acoustic material ID cannot be empty")

        try:
            return self._materials[requested_id]
        except KeyError:
            known = ", ".join(sorted(self._materials))
            raise KeyError(
                f"unknown acoustic material {requested_id!r}; "
                f"known materials: {known or '<none>'}"
            ) from None

    def surface_damping_db(
        self,
        material_id: str,
        surface_role: str,
    ) -> float:
        """Return a material-specific damping term in dB.

        The acoustic catalog already encodes realistic, per-band absorption and
        transmission-loss values. We use those values here instead of inventing
        synthetic per-surface constants. The floor path maps absorption to a
        small attenuation term; walls map directly to the profile's transmission
        loss. This keeps the damping cheap while honoring the material model.
        """
        material = self.get(str(material_id).strip() or "default")
        role = str(surface_role).strip().lower()

        if role == "wall":
            return float(material.mean_transmission_loss_db)

        if role in {"floor", "ceiling"}:
            absorption = material.mean_absorption
            if absorption <= 0.0:
                return 0.0

            # Floor and ceiling absorption is a same-room interaction term, not a
            # full barrier transmission loss. Keep the response physically
            # plausible, but scale it down to a modest dB range so contrasting
            # materials still matter without over-attenuating footsteps.
            floor_gain_db = 10.0 * absorption
            floor_gain_db = min(floor_gain_db, 6.0)
            return float(max(0.0, floor_gain_db))

        return 0.0

    @staticmethod
    def _load_yaml(path: Path) -> Mapping[str, Any]:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise OSError(
                f"failed to read acoustic material catalog {path}"
            ) from exc

        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(
                f"failed to parse acoustic material catalog {path}"
            ) from exc

        if raw is None:
            raise ValueError(
                f"acoustic material catalog {path} is empty"
            )

        if not isinstance(raw, Mapping):
            raise ValueError(
                f"{path}: catalog root must be a mapping"
            )

        return raw

    def _parse_center_frequencies(
        self,
        raw_frequencies: object,
    ) -> tuple[int, ...]:
        if not self._is_sequence(raw_frequencies):
            raise ValueError(
                f"{self._path}: 'octave_bands_hz' must be a sequence"
            )

        frequencies: list[int] = []

        for index, raw_value in enumerate(raw_frequencies):
            if isinstance(raw_value, bool):
                raise ValueError(
                    f"{self._path}: octave-band frequency at "
                    f"index {index} cannot be boolean"
                )

            try:
                frequency = int(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{self._path}: invalid octave-band frequency "
                    f"at index {index}: {raw_value!r}"
                ) from exc

            if frequency <= 0:
                raise ValueError(
                    f"{self._path}: octave-band frequencies "
                    "must be positive"
                )

            frequencies.append(frequency)

        if not frequencies:
            raise ValueError(
                f"{self._path}: at least one octave-band "
                "frequency is required"
            )

        if len(set(frequencies)) != len(frequencies):
            raise ValueError(
                f"{self._path}: octave-band frequencies "
                "must be unique"
            )

        if frequencies != sorted(frequencies):
            raise ValueError(
                f"{self._path}: octave-band frequencies "
                "must be strictly increasing"
            )

        return tuple(frequencies)

    def _parse_material(
        self,
        *,
        material_id: str,
        entry: Mapping[str, Any],
        used_default: bool,
    ) -> AcousticMaterial:
        canonical_name = str(
            entry.get("canonical_name", material_id)
        ).strip()

        if not canonical_name:
            canonical_name = material_id

        absorption = self._parse_band_values(
            entry.get("absorption"),
            field_name="absorption",
            material_id=material_id,
            minimum=0.0,
            maximum=1.0,
        )

        transmission_loss_db = self._parse_band_values(
            entry.get("transmission_loss_db"),
            field_name="transmission_loss_db",
            material_id=material_id,
            minimum=0.0,
            maximum=None,
        )

        scattering = self._parse_band_values(
            entry.get("scattering", 0.1),
            field_name="scattering",
            material_id=material_id,
            minimum=0.0,
            maximum=1.0,
        )

        material = AcousticMaterial(
            material_id=material_id,
            canonical_name=canonical_name,
            center_frequencies_hz=self._center_frequencies_hz,
            absorption=absorption,
            transmission_loss_db=transmission_loss_db,
            scattering=scattering,
            used_default=used_default,
        )

        self._validate_material(material)
        return material

    def _parse_band_values(
        self,
        raw_values: object,
        *,
        field_name: str,
        material_id: str,
        minimum: float | None,
        maximum: float | None,
    ) -> tuple[float, ...]:
        if raw_values is None:
            raise ValueError(
                f"{self._path}: material {material_id!r} "
                f"is missing {field_name!r}"
            )

        if self._is_number(raw_values):
            values = [
                float(raw_values)
                for _ in self._center_frequencies_hz
            ]
        elif self._is_sequence(raw_values):
            values = []

            for index, raw_value in enumerate(raw_values):
                if not self._is_number(raw_value):
                    raise ValueError(
                        f"{self._path}: material {material_id!r} "
                        f"has a non-numeric {field_name} value "
                        f"at index {index}: {raw_value!r}"
                    )

                values.append(float(raw_value))
        else:
            raise ValueError(
                f"{self._path}: material {material_id!r} "
                f"field {field_name!r} must be a number "
                "or a sequence of numbers"
            )

        expected_count = len(self._center_frequencies_hz)

        if len(values) != expected_count:
            raise ValueError(
                f"{self._path}: material {material_id!r} "
                f"field {field_name!r} has {len(values)} values; "
                f"expected {expected_count}"
            )

        for index, value in enumerate(values):
            if not self._is_finite(value):
                raise ValueError(
                    f"{self._path}: material {material_id!r} "
                    f"field {field_name!r} contains a non-finite "
                    f"value at index {index}"
                )

            if minimum is not None and value < minimum:
                raise ValueError(
                    f"{self._path}: material {material_id!r} "
                    f"field {field_name!r} contains {value} below "
                    f"the minimum {minimum}"
                )

            if maximum is not None and value > maximum:
                raise ValueError(
                    f"{self._path}: material {material_id!r} "
                    f"field {field_name!r} contains {value} above "
                    f"the maximum {maximum}"
                )

        return tuple(values)

    def _validate_material(
        self,
        material: AcousticMaterial,
    ) -> None:
        expected_count = len(self._center_frequencies_hz)

        fields = {
            "absorption": material.absorption,
            "transmission_loss_db": (
                material.transmission_loss_db
            ),
            "scattering": material.scattering,
        }

        for field_name, values in fields.items():
            if len(values) != expected_count:
                raise ValueError(
                    f"{self._path}: material "
                    f"{material.material_id!r} field "
                    f"{field_name!r} has {len(values)} values; "
                    f"expected {expected_count}"
                )

    @staticmethod
    def _is_sequence(value: object) -> bool:
        return isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        )

    @staticmethod
    def _is_number(value: object) -> bool:
        return isinstance(value, (int, float)) and not isinstance(
            value,
            bool,
        )

    @staticmethod
    def _is_finite(value: float) -> bool:
        return value == value and value not in (
            float("inf"),
            float("-inf"),
        )
