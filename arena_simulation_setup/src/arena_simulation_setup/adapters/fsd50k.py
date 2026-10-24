"""FSD50K importer: Freesound clips under CC0 or CC BY, built into Common/Sound items by the sound database builder."""

from __future__ import annotations

import csv
import json
import re
import typing
from collections.abc import Iterable, Iterator, Mapping
from functools import cached_property
from pathlib import Path

import attrs
import yaml

from arena_assets.impl import AssetType
from arena_assets.importers import IMPORTERS, Attribution
from arena_assets.importers.providers import HFHub, Provider
from arena_assets.semantic.taxonomy import load_spec
from arena_simulation_setup import ASS_DIR
from arena_simulation_setup.adapters.sound import SoundImporter

if typing.TYPE_CHECKING:
    from arena_simulation_setup.tree.assets.sound_catalog import SoundAsset

REPO = "Fhrozen/FSD50k"
SPLITS = ("dev", "eval")
LABELS_FILE = "labels/{split}.csv"
CLIP_INFO_FILE = "metadata/{split}_clips_info_FSD50K.json"
CLIP_FILE = "clips/{split}/{item}.wav"
KINDS_FILE = ASS_DIR / "configs" / "sounds" / "fsd50k_kinds.yaml"
ORIGIN = "origin::fsd50k"
NAME = "fsd50k_{item}"
FREESOUND_URL = "https://freesound.org/s/{item}/"
CITATION = 'Eduardo Fonseca, Xavier Favory, Jordi Pons, Frederic Font, Xavier Serra. "FSD50K: an Open Dataset of Human-Labeled Sound Events", arXiv 2020.'

_LICENSES = (
    (re.compile(r"creativecommons\.org/publicdomain/zero/(\d+\.\d+)/?$"), "CC0-{}"),
    (re.compile(r"creativecommons\.org/licenses/by/(\d+\.\d+)/?$"), "CC-BY-{}"),
)
_TITLE_SUFFIX = re.compile(r"\.(wav|aif|aiff|mp3|flac|ogg|m4a)$", re.IGNORECASE)
_NOT_A_TAG_CHARACTER = re.compile(r"[^a-z0-9]+")


def spdx_license(url: str) -> str | None:
    """SPDX id of a Freesound CC0 or CC BY license URL, None for every other license."""
    for pattern, template in _LICENSES:
        match = pattern.search(url.strip())
        if match:
            return template.format(match.group(1))
    return None


@attrs.frozen
class Clip:
    item: str
    split: str
    labels: tuple[str, ...]
    title: str
    description: str
    tags: tuple[str, ...]
    license_url: str
    uploader: str

    @property
    def license(self) -> str | None:
        return spdx_license(self.license_url)


def load_kinds(path: Path = KINDS_FILE) -> dict[str, str]:
    """AudioSet label to ASA sound kind. Raises ValueError on a kind outside the ASA vocabulary or a label listed under two kinds."""
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: kinds file must map kind to a list of labels")
    vocabulary = load_spec().facets["kind"].values_for(AssetType.SOUND.value)
    kinds: dict[str, str] = {}
    for kind, labels in data.items():
        if kind not in vocabulary:
            raise ValueError(f"{path}: {kind!r} is not an ASA sound kind, expected one of {vocabulary}")
        if not isinstance(labels, list) or not labels:
            raise ValueError(f"{path}: kind {kind!r} needs a non-empty list of labels")
        for label in labels:
            if label in kinds:
                raise ValueError(f"{path}: label {label!r} is listed under {kinds[label]!r} and {kind!r}")
            kinds[str(label)] = str(kind)
    return kinds


def kind_of(labels: Iterable[str], kinds: Mapping[str, str]) -> str | None:
    """The kind of the first mapped label, None when no label maps."""
    for label in labels:
        kind = kinds.get(label)
        if kind is not None:
            return kind
    return None


def asset_name(item: str) -> str:
    return NAME.format(item=item)


def asset_tags(clip: Clip) -> list[str]:
    """The origin facet, then the AudioSet labels and the uploader's Freesound tags as bare tags."""
    words = (_NOT_A_TAG_CHARACTER.sub("_", word.lower()).strip("_") for word in (*clip.labels, *clip.tags))
    return list(dict.fromkeys([ORIGIN, *(word for word in words if word)]))


class FSD50K(SoundImporter):
    meta = IMPORTERS.meta("sound", "fsd50k")

    @classmethod
    def default_provider(cls) -> Provider:
        return HFHub(REPO)

    def _file(self, relpath: str) -> Path:
        return self.provider.get(relpath, self.cache / relpath)

    @cached_property
    def kinds(self) -> dict[str, str]:
        return load_kinds()

    @cached_property
    def clips(self) -> dict[str, Clip]:
        clips: dict[str, Clip] = {}
        for split in SPLITS:
            with open(self._file(CLIP_INFO_FILE.format(split=split))) as f:
                info = json.load(f)
            with open(self._file(LABELS_FILE.format(split=split)), newline="") as f:
                for row in csv.DictReader(f):
                    item = row["fname"]
                    meta = info[item]
                    clips[item] = Clip(
                        item=item,
                        split=split,
                        labels=tuple(row["labels"].split(",")),
                        title=str(meta["title"]),
                        description=str(meta["description"]),
                        tags=tuple(str(tag) for tag in meta["tags"]),
                        license_url=str(meta["license"]),
                        uploader=str(meta["uploader"]),
                    )
        return clips

    def clip(self, item: str) -> Clip:
        try:
            return self.clips[str(item)]
        except KeyError:
            raise KeyError(f"fsd50k has no clip {item!r}, items are Freesound ids such as {next(iter(self.clips), '')!r}") from None

    def _classified(self, item: str) -> tuple[str, SoundAsset]:
        clip = self.clip(item)
        kind = kind_of(clip.labels, self.kinds)
        if kind is None:
            raise ValueError(f"fsd50k clip {item!r} has no label mapped in {KINDS_FILE.name}: {', '.join(clip.labels)}")
        reference = self.reference(kind)
        if reference is None:
            raise ValueError(f"fsd50k clip {item!r} is of kind {kind!r}, which has no default asset in the kinds table to take the emission level from")
        return kind, reference

    def kind(self, item: str) -> str:
        return self._classified(item)[0]

    def items(self) -> Iterator[str]:
        for item, clip in self.clips.items():
            kind = kind_of(clip.labels, self.kinds)
            if clip.license is not None and kind is not None and self.library.kind(kind).default_asset:
                yield item

    def fetch(self, item: str) -> Path:
        clip = self.clip(item)
        if clip.license is None:
            raise ValueError(f"fsd50k clip {item!r} is not redistributable: {clip.license_url}")
        return self._file(CLIP_FILE.format(split=clip.split, item=item))

    def asset_name(self, item: str) -> str:
        return asset_name(item)

    def convert(self, item: str, src: Path, out: Path) -> str:
        kind, reference = self._classified(item)
        clip = self.clip(item)
        return self.build(item, kind, reference, _TITLE_SUFFIX.sub("", clip.title).strip(), asset_tags(clip), src, out)

    def attribution(self, item: str) -> Attribution:
        clip = self.clip(item)
        license = clip.license
        if license is None:
            raise ValueError(f"fsd50k clip {item!r} is not redistributable: {clip.license_url}")
        return Attribution(source=self.meta.name, url=FREESOUND_URL.format(item=item), license=license, authors=clip.uploader, citation=CITATION)
