from __future__ import annotations

import abc
import asyncio
import enum
import itertools
import logging
import os
import subprocess
import sys
import threading
import time
import typing
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import attrs
from typing_extensions import Self

from arena_simulation_setup import (
    ARENA_ASSETS_DIR,
    DOMAIN_DEFAULT,
)
from arena_simulation_setup.utils.cattrs import Idempotent, Parseable, Serializable


def providers(env: str, default: str) -> Sequence[str]:
    """Bucket list for one asset family, from *env* or *default*. Empty falls back too,
    so the shell can export the override slot without restating the bucket name."""
    return [p for p in (os.environ.get(env) or default).split(',') if p]


NETWORK_PROVIDERS: Sequence[str] = providers('ASSET_BUCKETS', 'default')
WORLD_PROVIDERS: Sequence[str] = providers('WORLD_BUCKETS', 'arena-worlds-prod-public')
BENCHMARK_PROVIDERS: Sequence[str] = providers('BENCHMARK_BUCKETS', 'arena-benchmarks-prod-public')


async def _subprocess_output(args: Sequence[str], **kwargs: object) -> bytes:
    process = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, **kwargs)
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode or -1, list(args), output=stdout, stderr=stderr)
    return stdout


# Utils


class PathContainer:
    def __init__(self, path: Path):
        self._path: Path = path

    @property
    def path(self) -> Path:
        return self._path


class PathView(PathContainer):
    @property
    def name(self) -> str:
        return self.path.name


class DynamicPath(PathContainer):
    @PathContainer.path.setter
    def path(self, value: Path):
        self._path = value

    def __init__(self, path: Path = Path('/dev/null')):
        super().__init__(path)


class DynamicPaths:
    WORLD = DynamicPath()
    ARENA = DynamicPath(Path(os.getenv('ARENA_ASSETS_DIR_LOCAL', ARENA_ASSETS_DIR / 'local')))

    @classmethod
    def as_resolvers(cls, _T: type[IdentifierT], /) -> Iterable[DynamicPathResolver[IdentifierT]]:
        yield DynamicPathResolver(_T, cls.WORLD, fn=lambda p: p / 'assets')
        yield DynamicPathResolver(_T, cls.ARENA)


IdentifierT = typing.TypeVar('IdentifierT', bound='IdentifierProtocol')


@typing.runtime_checkable
class References(typing.Protocol):
    """A loaded asset that names further assets, so closures can walk through it."""

    def identifiers(self) -> Iterable[IdentifierProtocol]: ...


class Verdict(enum.Enum):
    HIT = 'hit'
    MISS = 'miss'
    SHADOWED = 'shadowed'
    STALE = 'stale'


@attrs.define
class ResolverVerdict:
    """One resolver's answer for one identifier, as reported by ``Identifier.probe``."""

    resolver: ResolverBase
    verdict: Verdict
    path: Path | None = None


class ResolverBase(abc.ABC, typing.Generic[IdentifierT]):
    """
    Base class for resolvers.
    """

    @property
    def _asset_type(self) -> str:
        return self._IdentifierT.label()

    _cache: dict[IdentifierT, Path] = attrs.field(factory=dict, init=False)

    def __init__(self, _T: type[IdentifierT], /):
        self._IdentifierT: type[IdentifierT] = _T
        self._cache = {}

    async def resolve(self, identifier: IdentifierT) -> Path | None:
        """Read path for *identifier*, fetching if this resolver can. None if unavailable here."""
        del identifier
        return None

    async def locate(self, identifier: IdentifierT) -> Path | None:
        """Where *identifier* would resolve, without transferring anything."""
        return await self.resolve(identifier)

    def is_stale(self, identifier: IdentifierT) -> bool:
        """Whether the last answer for *identifier* was served from an expired cache."""
        del identifier
        return False

    def destination(self, identifier: IdentifierT) -> Path | None:
        """Write path for *identifier*. None for resolvers that are not write targets."""
        del identifier
        return None

    def invalidate(self):
        self._cache.clear()

    def listall(self, **kwargs: object) -> Iterator[IdentifierT]:
        """
        List all local assets available. Builds the cache in the process.
        """
        return iter(())

    async def listall_async(self, **kwargs: object) -> list[IdentifierT]:
        """Non-blocking variant for callers on an event loop. Default: sync listall."""
        return list(self.listall(**kwargs))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}<{self._asset_type}>'


class PathResolverBase(ResolverBase[IdentifierT], abc.ABC, typing.Generic[IdentifierT]):
    """
    Resolve asset paths from disk.
    """

    _listall_cache: tuple[Path, list[IdentifierT]] | None = None

    # file suffixes probed in order for a relpath that carries none of them
    suffixes: typing.ClassVar[tuple[str, ...]] = ()

    @property
    @abc.abstractmethod
    def path(self) -> Path: ...

    def _candidates(self, relpath: Path) -> list[Path]:
        if not self.suffixes or relpath.name.endswith(self.suffixes):
            return [relpath]
        return [relpath.with_name(relpath.name + suffix) for suffix in self.suffixes]

    async def resolve(self, identifier: IdentifierT) -> Path | None:
        if identifier not in self._cache:
            for candidate in self._candidates(identifier.relpath()):
                if (self.path / candidate).exists():
                    self._cache[identifier] = self.path / candidate
                    break
        return self._cache.get(identifier, None)

    # os.walk's syscalls each drop the GIL, so on a busy node's event loop the walk stretches
    # from milliseconds to seconds. The async variant enumerates in a child process instead.
    _WALK_SCRIPT = """import os, sys
root = sys.argv[1]
for path, _, files in os.walk(root):
    sys.stdout.write('\\0'.join([os.path.relpath(path, root), *files]) + '\\n')
"""

    def _identify(self, relpath: Path, files: Iterable[str]) -> tuple[IdentifierT | None, list[IdentifierT]]:
        """The directory itself as an asset, else the assets among the files it holds."""
        try:
            return self._IdentifierT.from_relpath(relpath), []
        except Exception:
            pass

        found: list[IdentifierT] = []
        for file in files:
            if file.lower() == 'readme.md':
                continue
            try:
                found.append(self._IdentifierT.from_relpath(relpath / file))
            except Exception:
                pass
        return None, found

    def listall(self, **kwargs: object) -> Iterator[IdentifierT]:
        source = self.path
        if not source.is_dir():
            yield from ()
            return
        for root, dirs, files in os.walk(source):
            asset, found = self._identify(Path(root).relative_to(source), files)
            if asset is not None:
                yield asset
                # don't recurse
                dirs.clear()
                continue
            yield from found

    async def listall_async(self, **kwargs: object) -> list[IdentifierT]:
        source = self.path
        if self._listall_cache is not None and self._listall_cache[0] == source:
            return list(self._listall_cache[1])
        if not source.is_dir():
            return []

        try:
            output = await _subprocess_output([sys.executable, '-c', self._WALK_SCRIPT, str(source)])
        except (OSError, subprocess.CalledProcessError):
            import traceback

            logging.warning(traceback.format_exc())
            return list(self.listall(**kwargs))

        result: list[IdentifierT] = []
        pruned: set[str] = set()
        for line in output.decode().splitlines():
            relpath, *files = line.split('\0')
            rel = Path(relpath)
            if any(str(parent) in pruned for parent in rel.parents):
                continue
            asset, found = self._identify(rel, files)
            if asset is not None:
                result.append(asset)
                pruned.add(relpath)
                continue
            result.extend(found)

        self._listall_cache = (source, result)
        return list(result)

    def invalidate(self):
        super().invalidate()
        self._listall_cache = None

    def __repr__(self) -> str:
        return f"{super().__repr__()}(path={self.path})"


class SimplePathResolver(PathResolverBase[IdentifierT], typing.Generic[IdentifierT]):
    """
    Resolve asset paths from a single disk location.
    """

    def __init__(self, _T: type[IdentifierT], /, path: Path, **kwargs: object) -> None:
        super().__init__(_T, **kwargs)
        self._path = path

    @property
    def path(self) -> Path:
        return self._path


class DynamicPathResolver(PathResolverBase[IdentifierT], typing.Generic[IdentifierT]):
    """
    Resolve asset paths from a dynamic disk location.
    """

    def __init__(self, _T: type[IdentifierT], /, path: DynamicPath, *, fn: typing.Callable[[Path], Path] | None = None, **kwargs: object) -> None:
        super().__init__(_T, **kwargs)
        if fn is None:

            def _identity_fn(p: Path) -> Path:
                return p

            fn = _identity_fn
        self._dynamic_path = path
        self._fn = fn

    @property
    def path(self) -> Path:
        return self._fn(self._dynamic_path.path)


class NetResolver(SimplePathResolver[IdentifierT], ResolverBase[IdentifierT], typing.Generic[IdentifierT]):
    """
    Resolve asset paths from network. The TTL governs refresh, not availability: an
    expired cache entry is served when the provider cannot be reached.
    """

    _TTL_MARKER = '.ttl'

    def __init__(
        self,
        _T: type[IdentifierT],
        /,
        provider: str,
        *,
        formats: Iterable[str] | None = None,
        ttl: int | None = None,
        annotated: bool = True,
        list_prefix: str = '',
        **kwargs: object,
    ) -> None:
        path = ARENA_ASSETS_DIR / provider
        super().__init__(_T, path=path, **kwargs)
        self._provider: str = provider
        # payloads without an annotation.yaml sentinel are enumerated by directory instead
        self._annotated = annotated
        self._list_prefix = list_prefix

        # formats=() opts out of the model-format filter, which would otherwise drop
        # every non-model file in a payload (meshes, scenario.json, map images).
        self._formats = list(formats) if formats is not None else os.environ.get('ARENA_MODELS_FORMATS', '').split(',')
        self._ttl = ttl if ttl is not None else int(os.environ.get('ARENA_MODELS_TTL', '86400'))
        self._listing_ttl = int(os.environ.get('ARENA_MODELS_TTL', '86400'))
        self._pending: dict[str, list[asyncio.Future[Path | None]]] = {}
        self._flush_scheduled = False
        self._batch_lock = asyncio.Lock()
        self._stale: set[IdentifierT] = set()

    def invalidate(self):
        super().invalidate()
        self._stale.clear()

    def is_stale(self, identifier: IdentifierT) -> bool:
        return identifier in self._stale

    def _complete(self, candidate: Path) -> bool:
        """Fully fetched once: the marker is written only after a successful fetch."""
        if not candidate.is_dir():
            return candidate.exists()
        try:
            return int((candidate / self._TTL_MARKER).read_text().strip()) > 0
        except (OSError, ValueError):
            return False

    def _serve_stale(self, identifier: IdentifierT, candidate: Path) -> Path | None:
        if not self._complete(candidate):
            return None
        logging.warning('%s: %s unreachable, serving expired cache %s', type(self).__name__, self._provider, candidate)
        self._stale.add(identifier)
        return candidate

    @classmethod
    async def check_output_async(cls, args: Iterable[str], **kwargs: object) -> bytes:
        return await _subprocess_output(list(args), **kwargs)

    async def _batch_request(self, relpath: str) -> Path | None:
        loop = asyncio.get_event_loop()
        future: asyncio.Future[Path | None] = loop.create_future()
        async with self._batch_lock:
            self._pending.setdefault(relpath, []).append(future)
            if not self._flush_scheduled:
                self._flush_scheduled = True
                asyncio.create_task(self._flush())
        return await future

    async def _flush(self):
        # yield once so that requests arriving in the same tick are collected
        await asyncio.sleep(0)

        async with self._batch_lock:
            batch = dict(self._pending)
            self._pending.clear()
            self._flush_scheduled = False

        relpaths = list(batch.keys())
        root_path = ARENA_ASSETS_DIR / self._provider
        logging.info(
            "Batched fetch of %d asset(s) from provider %s: %s",
            len(relpaths),
            self._provider,
            relpaths,
        )

        try:
            await self.check_output_async(
                [
                    'ros2',
                    'run',
                    'arena_models',
                    'arena_models',
                    '-s',
                    'net',
                    self._provider,
                    'fetch',
                    *relpaths,
                    '-o',
                    str(root_path),
                    *itertools.chain.from_iterable(('--format', fmt) for fmt in self._formats if fmt),
                ]
            )
            stamp = str(int(time.time()))
            for relpath, futures in batch.items():
                target = root_path / relpath
                result = target if target.exists() else None
                if result is not None and result.is_dir():
                    try:
                        (result / self._TTL_MARKER).write_text(stamp)
                    except OSError:
                        pass
                for fut in futures:
                    if not fut.done():
                        fut.set_result(result)
        except Exception:
            import traceback

            logging.warning(traceback.format_exc())
            for futures in batch.values():
                for fut in futures:
                    if not fut.done():
                        fut.set_result(None)

    async def resolve(self, identifier: IdentifierT) -> Path | None:
        if identifier in self._cache:
            return self._cache[identifier]

        candidate = self.path / identifier.relpath()
        if candidate.is_dir():
            marker = candidate / self._TTL_MARKER
            try:
                stamp = int(marker.read_text().strip())
            except (OSError, ValueError):
                stamp = 0
            if stamp and time.time() - stamp < self._ttl:
                self._cache[identifier] = candidate
                return candidate

        result = await self._batch_request(str(identifier.relpath()))
        if result is None:
            result = self._serve_stale(identifier, candidate)
        if result is not None:
            self._cache[identifier] = result

        return self._cache.get(identifier, None)

    def _fresh(self, candidate: Path) -> bool:
        """A cache entry counts only while its TTL marker is current, as in resolve()."""
        if not candidate.is_dir():
            return candidate.exists()
        try:
            stamp = int((candidate / self._TTL_MARKER).read_text().strip())
        except (OSError, ValueError):
            return False
        return bool(stamp) and time.time() - stamp < self._ttl

    async def locate(self, identifier: IdentifierT) -> Path | None:
        candidate = self.path / identifier.relpath()
        if self._fresh(candidate):
            return candidate
        try:
            output = await self.check_output_async(
                [
                    'ros2',
                    'run',
                    'arena_models',
                    'arena_models',
                    '-s',
                    'net',
                    self._provider,
                    'exists',
                    '--no-annotation',
                    str(identifier.relpath()),
                ]
            )
        except (subprocess.CalledProcessError, OSError) as e:
            # a failed check is not an absent asset, so say so rather than reporting a silent miss
            logging.warning('%s: could not check %s on %s: %s', type(self).__name__, identifier.relpath(), self._provider, e)
            return self._serve_stale(identifier, candidate)
        return candidate if output.decode().strip().splitlines()[-1:] == ['1'] else None

    _LISTING_FILE = '.listing'
    _listing_inflight: typing.ClassVar[dict[str, asyncio.Future[list[str]]]] = {}

    @property
    def _listing_path(self) -> Path:
        # one listing file per listing command: kinds sharing a bucket list different prefixes
        suffix = '' if self._annotated else f'-{self._list_prefix or "children"}'
        return self.path / f'{self._LISTING_FILE}{suffix}'

    def _listing_args(self) -> list[str]:
        """Remote listing command. Annotated payloads are found by sentinel scan, everything
        else by listing the children of the prefix its identifiers live under."""
        argv = ['ros2', 'run', 'arena_models', 'arena_models', '-s', 'net', self._provider, 'list']
        if self._annotated:
            return argv
        return [*argv, '--children', *((self._list_prefix,) if self._list_prefix else ())]

    def _cached_listing(self) -> list[str] | None:
        """Bucket listing from disk if younger than the listing TTL, else None."""
        try:
            stat = self._listing_path.stat()
        except OSError:
            return None
        if time.time() - stat.st_mtime >= self._listing_ttl:
            return None
        return self._listing_path.read_text().splitlines()

    def _store_listing(self, output: bytes) -> list[str]:
        lines = [line.strip() for line in output.decode().splitlines() if line.strip()]
        if not lines:
            return lines
        try:
            self._listing_path.parent.mkdir(parents=True, exist_ok=True)
            self._listing_path.write_text('\n'.join(lines) + '\n')
        except OSError:
            pass
        return lines

    def _listing(self) -> list[str]:
        cached = self._cached_listing()
        if cached is not None:
            return cached
        try:
            return self._store_listing(subprocess.check_output(self._listing_args(), stderr=subprocess.DEVNULL))
        except (subprocess.CalledProcessError, OSError):
            import traceback

            logging.warning(traceback.format_exc())
            return []

    async def _listing_async(self) -> list[str]:
        cached = self._cached_listing()
        if cached is not None:
            return cached
        key = str(self._listing_path)
        inflight = self._listing_inflight.get(key)
        if inflight is not None:
            return await inflight
        future: asyncio.Future[list[str]] = asyncio.get_running_loop().create_future()
        self._listing_inflight[key] = future
        try:
            lines = self._store_listing(await self.check_output_async(self._listing_args()))
        except (subprocess.CalledProcessError, OSError):
            import traceback

            logging.warning(traceback.format_exc())
            lines = []
        finally:
            del self._listing_inflight[key]
        future.set_result(lines)
        return lines

    def _parse_listing(self, lines: Iterable[str]) -> Iterator[IdentifierT]:
        for line in lines:
            try:
                yield self._IdentifierT.from_relpath(Path(line))
            except Exception:
                pass

    def listall(self, *, network: bool = False, **kwargs: object) -> Iterator[IdentifierT]:
        """
        List all assets available. When *network* is True, also queries the
        remote provider (bucket listing cached on disk for ARENA_MODELS_TTL seconds).
        """
        yield from super(SimplePathResolver, self).listall(**kwargs)
        if network:
            yield from self._parse_listing(self._listing())

    async def listall_async(self, *, network: bool = False, **kwargs: object) -> list[IdentifierT]:
        result = await super(SimplePathResolver, self).listall_async(**kwargs)
        if network:
            result.extend(self._parse_listing(await self._listing_async()))
        return result

    def __repr__(self) -> str:
        return f"{super().__repr__()}(net={self._provider})"

    @classmethod
    def all(cls, _T: type[IdentifierT], /, providers: Iterable[str] = NETWORK_PROVIDERS, **kwargs: object) -> Iterable[Self]:
        for provider in providers:
            yield cls(_T, provider=provider, **kwargs)


class FallbackResolver(ResolverBase[IdentifierT], typing.Generic[IdentifierT]):
    """
    Write target. Never answers reads, so it cannot shadow a real source.
    """

    def __init__(self, _T: type[IdentifierT], /, path: Path, **kwargs: object) -> None:
        super().__init__(_T, **kwargs)
        self._path = path

    def destination(self, identifier: IdentifierT) -> Path | None:
        return self._path / identifier.relpath()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self._path)})"


# IDENTIFIERS


T = typing.TypeVar('T')
T_co = typing.TypeVar('T_co', covariant=True)
ResultT = typing.TypeVar('ResultT')


class IdentifierProtocol(typing.Protocol[T_co]):
    async def resolve(self, **kwargs: object) -> T_co:
        """Resolve and load the asset referenced by this identifier."""
        ...

    @property
    def shortname(self) -> str:
        """Get the short name of the identifier."""
        ...

    def relpath(self) -> Path: ...

    @classmethod
    def from_relpath(cls, relpath: Path) -> Self: ...

    @classmethod
    def label(cls) -> str:
        """Short label for the identifier type."""
        ...

    @classmethod
    def listall(cls, **kwargs: object) -> Iterator[Self]:
        """List all local assets available."""
        ...

    @classmethod
    async def listall_async(cls, **kwargs: object) -> list[Self]:
        """Non-blocking listall for callers on an event loop."""
        ...


@attrs.define(eq=False, hash=False)
class Identifier(IdentifierProtocol[T], Parseable, Serializable, Idempotent, typing.Generic[T]):
    """Represents an identifier referencing an path."""

    name: str

    # True where load() returns a References, so closure walks recurse instead of
    # fetching every leaf just to find out it names nothing
    NESTS: typing.ClassVar[bool] = False

    @property
    def shortname(self) -> str:
        return self.name

    def relpath(self) -> Path:
        return Path(self.name)

    async def resolve_path(self) -> Path:
        for resolver in self._resolvers:
            resolved = await resolver.resolve(self)
            if resolved is not None:
                return resolved
        msg = f'{self} not found among'
        for resolver in self._resolvers:
            msg += f'\n\t{repr(resolver)}'
        raise FileNotFoundError(msg)

    async def resolve_write_path(self) -> Path:
        """Where this identifier is written. Never a cache, never the network."""
        for resolver in self._resolvers:
            destination = resolver.destination(self)
            if destination is not None:
                return destination
        raise FileNotFoundError(f'no write destination registered for {self}')

    def resolve_write_sync(self) -> T:
        """Synchronously load the write target for this identifier, creating nothing."""
        return self._run_sync(self._resolve_write())

    async def _resolve_write(self) -> T:
        path = await self.resolve_write_path()
        return await asyncio.to_thread(self.load, path)

    async def resolve_source(self) -> ResolverVerdict:
        """The resolver that answers for this identifier, stopping at the first hit."""
        for resolver in self._resolvers:
            located = await resolver.locate(self)
            if located is not None:
                return ResolverVerdict(resolver, Verdict.STALE if resolver.is_stale(self) else Verdict.HIT, located)
        msg = f'{self} not found among'
        for resolver in self._resolvers:
            msg += f'\n\t{repr(resolver)}'
        raise FileNotFoundError(msg)

    def resolve_source_sync(self) -> ResolverVerdict:
        return self._run_sync(self.resolve_source())

    async def probe(self) -> list[ResolverVerdict]:
        """Verdict per resolver, in order, without transferring anything."""
        verdicts: list[ResolverVerdict] = []
        found = False
        for resolver in self._resolvers:
            located = await resolver.locate(self)
            if located is None:
                verdicts.append(ResolverVerdict(resolver, Verdict.MISS))
                continue
            verdict = Verdict.SHADOWED if found else Verdict.STALE if resolver.is_stale(self) else Verdict.HIT
            verdicts.append(ResolverVerdict(resolver, verdict, located))
            found = True
        return verdicts

    def probe_sync(self) -> list[ResolverVerdict]:
        return self._run_sync(self.probe())

    async def resolve(self, **kwargs: object) -> T:
        path = await self.resolve_path()
        return await asyncio.to_thread(self.load, path, **kwargs)

    def resolve_sync(self, **kwargs: object) -> T:
        """Synchronously load the asset referenced by this identifier."""
        return self._run_sync(self.resolve(**kwargs))

    @staticmethod
    def _run_sync(coro: typing.Coroutine[typing.Any, typing.Any, ResultT]) -> ResultT:
        """Drive *coro* on a private loop in its own thread, so sync callers can await."""
        result: ResultT = None  # type: ignore
        exc: Exception | None = None

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                nonlocal result
                result = loop.run_until_complete(coro)
            except Exception as e:
                nonlocal exc
                exc = e
            finally:
                loop.close()

        thread = threading.Thread(target=_run)
        thread.start()
        thread.join()

        if exc is not None:
            raise exc
        return result

    @abc.abstractmethod
    def load(self, path: Path, /, **kwargs: object) -> T:
        """Load the asset referenced by this identifier."""

    def serialize(self) -> str:
        return self.shortname

    # Class Methods

    @classmethod
    def label(cls) -> str:
        return cls.__name__

    @classmethod
    def from_relpath(cls, relpath: Path) -> Self:
        assert len(relpath.parts) >= 1, f'Expected at least 1 part in relpath, got {len(relpath.parts)}'
        return cls(name=str(relpath))

    @classmethod
    def parse(cls, value: str | Self) -> Self:
        if isinstance(value, cls):
            return value

        if not isinstance(value, str):
            raise TypeError(f'Expected str, got {repr(value)}')

        return cls(name=value)

    @classmethod
    def converter(cls, *args: object, **kwargs: object) -> typing.Self:
        return super().instance_or(cls.parse)(*args, **kwargs)

    @classmethod
    def listall(cls, **kwargs: object) -> Iterator[Self]:
        seen: set[str] = set()
        for resolver in cls._resolvers:
            for identifier in resolver.listall(**kwargs):
                if identifier.shortname not in seen:
                    seen.add(identifier.shortname)
                    yield identifier

    @classmethod
    async def listall_async(cls, **kwargs: object) -> list[Self]:
        seen: set[str] = set()
        result: list[Self] = []
        for resolver in cls._resolvers:
            for identifier in await resolver.listall_async(**kwargs):
                if identifier.shortname not in seen:
                    seen.add(identifier.shortname)
                    result.append(identifier)
        return result

    # Resolvers
    _resolvers: typing.ClassVar[list[ResolverBase]] = []

    @classmethod
    def use(cls, *resolver: ResolverBase[Self]):
        if '_resolvers' not in cls.__dict__:
            cls._resolvers = []
        cls._resolvers.extend(resolver)

    @classmethod
    def inline(cls: type[Self], data: T, /, name: str = '', modifiers: dict | None = None) -> Self:
        """Create an InlineIdentifier containing the given data.

        Args:
            data(T_co): The asset data.
            name(str, optional): The name of the asset. Defaults to ''.
            modifiers(dict | None, optional): The modifiers dict for the asset. Defaults to None.

        Returns:
            InlineIdentifier[T_co]: The created InlineIdentifier.
        """
        del modifiers

        TypedInline = type(f"TypedInline_{cls.__name__}", (InlineIdentifier, cls), {})
        return typing.cast(Self, TypedInline(data, name=name))


@attrs.define(eq=False, hash=False)
class AssetIdentifier(Identifier[T], typing.Generic[T]):
    """Represents an identifier referencing an asset."""

    name: str

    _asset_type: typing.ClassVar[str]

    def relpath(self) -> Path:
        return Path(self._asset_type) / self.name

    @classmethod
    def label(cls) -> str:
        """Short label for the identifier type."""
        return cls._asset_type

    @classmethod
    def from_relpath(cls, relpath: Path) -> Self:
        assert len(relpath.parts) >= 2, f'Expected at least 2 parts in relpath, got {len(relpath.parts)}'
        assert relpath.parts[0] == cls._asset_type, f'Expected asset type {cls._asset_type}, got {relpath.parts[0]}'
        return cls(name=str(Path(*relpath.parts[1:])))


@attrs.define(eq=False, hash=False)
class DomainAssetIdentifier(AssetIdentifier[T], typing.Generic[T]):
    """Represents an identifier referencing an asset."""

    name: str
    domain: str = attrs.field(default=DOMAIN_DEFAULT)

    def __hash__(self) -> int:
        return hash(
            (
                self.domain,
                self.name,
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DomainAssetIdentifier):
            return False
        return self.domain == other.domain and self.name == other.name

    @classmethod
    def parse(cls, value: str | Self) -> Self:  # type: ignore
        """Parse path of the form[domain:]name into an Identifier.

        Args:
            identifier(str): The identifier string to parse.
            default_target(AssetType): The default target type if not specified.
            default_domain(str): The default domain if not specified.

        Returns:
            Identifier: The parsed Identifier object.
        """
        if isinstance(value, cls):
            return value

        if not isinstance(value, str):
            raise TypeError(f'Expected str, got {repr(value)}')

        parts = list(value.split('/', 3))

        if len(parts) > 1 and parts[1] == cls._asset_type:
            # asset type included, shift
            parts.pop(1)
        if len(parts) > 1:
            # domain included
            domain = parts.pop(0)
        else:
            domain = DOMAIN_DEFAULT
        name = "/".join(parts)

        return cls(
            domain=domain,
            name=name,
        )

    @property
    def shortname(self) -> str:
        return f'{self.domain}/{self._asset_type}/{self.name}'

    def relpath(self) -> Path:
        return Path(self.domain) / self._asset_type / self.name

    @classmethod
    def from_relpath(cls, relpath: Path) -> Self:
        assert len(relpath.parts) >= 3, f'Expected at least 3 parts in relpath, got {len(relpath.parts)}'
        assert relpath.parts[1] == cls._asset_type, f'Expected asset type {cls._asset_type}, got {relpath.parts[0]}'
        return cls(domain=relpath.parts[0], name=str(Path(*relpath.parts[2:])))


@attrs.define(eq=False, hash=False)
class ModifiersDomainAssetIdentifier(DomainAssetIdentifier[T], typing.Generic[T]):
    """DomainIdentifiers that supports modifiers."""

    _modifiers: dict | None = attrs.field(default=None, alias='modifiers', repr=False)

    @property
    def modifiers(self) -> dict:
        """Get the modifiers dictionary."""
        return {} if self._modifiers is None else self._modifiers

    def __hash__(self) -> int:
        return hash((self.domain, self.name, frozenset(self.modifiers.items())))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModifiersDomainAssetIdentifier):
            return False
        return self.domain == other.domain and self.name == other.name and self.modifiers == other.modifiers

    @classmethod
    def parse(cls, value: str | tuple[str, dict] | Self) -> Self:  # type: ignore
        """Parse path of the form[domain:]name into an Identifier.

        Args:
            identifier(str): The identifier string to parse.
            default_target(AssetType): The default target type if not specified.
            default_domain(str): The default domain if not specified.

        Returns:
            Identifier: The parsed Identifier object.
        """
        if isinstance(value, DomainAssetIdentifier):
            if isinstance(value, cls):
                return value
            raise TypeError(f'Received Identifier of type {type(value)}, expected {cls}')

        modifiers = None
        if isinstance(value, Iterable) and not isinstance(value, str):
            if not len(value) == 2:
                raise ValueError(f'Expected Iterable of length 2, got {len(value)}')
            value, modifiers = value[0], value[1]

            if not isinstance(value, str):
                raise TypeError(f'Expected value to be str, got {repr(value)}')
            if not isinstance(modifiers, dict):
                raise TypeError(f'Expected modifiers to be dict, got {repr(modifiers)}')

        base = super().parse(value)

        return cls(
            domain=base.domain,
            name=base.name,
            modifiers=modifiers,
        )

    def serialize(self) -> str | tuple[str, dict]:
        base = super().serialize()
        if self.modifiers is None:
            return base
        return (base, self.modifiers)


class InlineIdentifier(Identifier[T], typing.Generic[T]):
    """An identifier that directly contains the asset data."""

    _data: T

    def __init__(self, data: T, /, name: str = '', **kwargs: object) -> None:
        super().__init__(name=name, **kwargs)
        self._data = data

    def serialize(self) -> str:
        raise TypeError('InlineIdentifier cannot be serialized')

    async def resolve(self, **kwargs: object) -> T:
        del kwargs  # unused
        return self._data

    def load(self, path: Path, /, **kwargs: object) -> T:
        raise NotImplementedError('How did you even get here?')
