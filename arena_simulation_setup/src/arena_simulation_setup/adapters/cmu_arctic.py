"""CMU ARCTIC importer: read-speech utterances of seven speakers, built into Common/Sound items of kind speech by the sound database builder."""

from __future__ import annotations

import re
import urllib.error
from collections.abc import Iterator
from pathlib import Path

import attrs

from arena_assets.importers import IMPORTERS, Attribution
from arena_assets.importers.providers import Http, Provider
from arena_simulation_setup.adapters.sound import SoundImporter

BASE = "http://festvox.org/cmu_arctic"
VOICE_DIR = "cmu_arctic/cmu_us_{voice}_arctic"
WAV_INDEX = VOICE_DIR + "/wav/"
WAV_FILE = VOICE_DIR + "/wav/arctic_{utterance}.wav"
TRANSCRIPT_FILE = VOICE_DIR + "/etc/txt.done.data"
COPYING_FILE = VOICE_DIR + "/COPYING"
PROMPTS_FILE = "cmuarctic.data"
KIND = "speech"
ORIGIN = "origin::cmu_arctic"
NAME = "cmu_arctic_{item}"
LICENSE = "CMU ARCTIC license: any use, keep the copyright notice and the authors' names, mark modifications"
CITATION = 'John Kominek, Alan W Black. "CMU ARCTIC databases for speech synthesis", Carnegie Mellon University, Language Technologies Institute, Tech Report CMU-LTI-03-177, 2003.'

_ITEM = re.compile(r"^([a-z]{3})_([ab]\d{4})$")
_WAV_LINK = re.compile(r'href="arctic_([ab]\d{4})\.wav"')
_PROMPT = re.compile(r'^\(\s*arctic_([ab]\d{4})\s+"(.*)"\s*\)$')
_COPYRIGHT = re.compile(r"Copyright \(c\) (\d{4})")


@attrs.frozen
class Voice:
    gender: str
    accent: str


VOICES = {
    "awb": Voice("male", "scottish_english"),
    "bdl": Voice("male", "us_english"),
    "clb": Voice("female", "us_english"),
    "jmk": Voice("male", "canadian_english"),
    "ksp": Voice("male", "indian_english"),
    "rms": Voice("male", "us_english"),
    "slt": Voice("female", "us_english"),
}


def split_item(item: str) -> tuple[str, str]:
    """Voice and utterance of an item id such as bdl_a0001. Raises KeyError on another shape or an unknown voice."""
    match = _ITEM.match(item)
    if match is None or match.group(1) not in VOICES:
        raise KeyError(f"cmu_arctic has no item {item!r}, items are <voice>_<utterance> such as bdl_a0001, voices: {', '.join(VOICES)}")
    return match.group(1), match.group(2)


def asset_name(item: str) -> str:
    return NAME.format(item=item)


def asset_tags(voice: str) -> list[str]:
    """The origin facet, then the speaker as bare tags."""
    return [ORIGIN, "read_speech", voice, VOICES[voice].gender, VOICES[voice].accent]


def read_prompts(path: Path) -> dict[str, str]:
    """Utterance to text from a festvox prompt file of ( arctic_a0001 "text" ) lines."""
    prompts: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="latin-1").splitlines(), 1):
        if not line.strip():
            continue
        match = _PROMPT.match(line.strip())
        if match is None:
            raise ValueError(f"{path}:{number}: not a festvox prompt line: {line!r}")
        prompts[match.group(1)] = match.group(2)
    return prompts


def read_copyright_year(path: Path) -> str:
    match = _COPYRIGHT.search(path.read_text(encoding="latin-1"))
    if match is None:
        raise ValueError(f"{path}: no copyright line")
    return match.group(1)


class CMUArctic(SoundImporter):
    meta = IMPORTERS.meta("sound", "cmu_arctic")

    @classmethod
    def default_provider(cls) -> Provider:
        return Http(BASE)

    def _file(self, relpath: str) -> Path:
        return self.provider.get(relpath, self.cache / relpath)

    def utterances(self, voice: str) -> list[str]:
        """Utterances with a wav in the voice's directory listing."""
        body, _ = self.provider.read(self.provider.url(WAV_INDEX.format(voice=voice)))
        return sorted(set(_WAV_LINK.findall(body.decode("latin-1"))))

    def prompts(self, voice: str) -> dict[str, str]:
        """Text of every utterance as the voice read it, the shared prompt list for utterances its transcript lacks."""
        return read_prompts(self._file(PROMPTS_FILE)) | read_prompts(self._file(TRANSCRIPT_FILE.format(voice=voice)))

    def items(self) -> Iterator[str]:
        for voice in VOICES:
            for utterance in self.utterances(voice):
                yield f"{voice}_{utterance}"

    def text(self, item: str) -> str:
        voice, utterance = split_item(item)
        try:
            return self.prompts(voice)[utterance]
        except KeyError:
            raise KeyError(f"cmu_arctic has no text for {item!r} in {TRANSCRIPT_FILE.format(voice=voice)} or {PROMPTS_FILE}") from None

    def fetch(self, item: str) -> Path:
        voice, utterance = split_item(item)
        relpath = WAV_FILE.format(voice=voice, utterance=utterance)
        try:
            return self._file(relpath)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise KeyError(f"cmu_arctic has no utterance {utterance!r} for voice {voice!r}, {self.provider.url(WAV_INDEX.format(voice=voice))} lists them") from None
            raise

    def asset_name(self, item: str) -> str:
        return asset_name(item)

    def convert(self, item: str, src: Path, out: Path) -> str:
        voice, _ = split_item(item)
        reference = self.reference(KIND)
        if reference is None:
            raise ValueError(f"kind {KIND!r} has no default asset in the kinds table to take the emission level from")
        return self.build(item, KIND, reference, self.text(item), asset_tags(voice), src, out)

    def attribution(self, item: str) -> Attribution:
        voice, utterance = split_item(item)
        year = read_copyright_year(self._file(COPYING_FILE.format(voice=voice)))
        return Attribution(
            source=self.meta.name,
            url=self.provider.url(WAV_FILE.format(voice=voice, utterance=utterance)),
            license=LICENSE,
            authors=f"Carnegie Mellon University, copyright {year}, speaker {voice}",
            citation=CITATION,
        )
