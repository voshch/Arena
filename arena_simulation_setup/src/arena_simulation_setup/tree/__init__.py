from __future__ import annotations

import abc
import asyncio
import itertools
import logging
import os
import subprocess
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

NETWORK_PROVIDERS: Sequence[str] = os.environ.get('ASSET_BUCKETS', 'default').split(',')

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

    @abc.abstractmethod
    async def resolve(self, identifier: IdentifierT) -> Path | None: ...

    def invalidate(self):
        self._cache.clear()

    def listall(self, **kwargs: object) -> Iterator[IdentifierT]:
        """
        List all local assets available. Builds the cache in the process.
        """
        return iter(())

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}<{self._asset_type}>'


class PathResolverBase(ResolverBase[IdentifierT], abc.ABC, typing.Generic[IdentifierT]):
    """
    Resolve asset paths from disk.
    """

    @property
    @abc.abstractmethod
    def path(self) -> Path: ...

    async def resolve(self, identifier: IdentifierT) -> Path | None:
        if identifier not in self._cache:
            candidate = self.path / identifier.relpath()
            if candidate.exists():
                self._cache[identifier] = candidate
                return candidate
        return self._cache.get(identifier, None)

    def listall(self, **kwargs: object) -> Iterator[IdentifierT]:
        source = self.path
        if not source.is_dir():
            yield from ()
            return
        for root, dirs, files in os.walk(source):
            relpath = Path(root).relative_to(source)
            try:
                yield self._IdentifierT.from_relpath(relpath)
                # don't recurse
                dirs.clear()
                continue
            except Exception:
                pass

            for file in files:
                if file.lower() == 'readme.md':
                    continue
                file_relpath = relpath / file
                try:
                    yield self._IdentifierT.from_relpath(file_relpath)
                except Exception:
                    pass

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
    Resolve asset paths from network.
    """

    _TTL_MARKER = '.ttl'

    def __init__(self, _T: type[IdentifierT], /, provider: str, **kwargs: object) -> None:
        path = ARENA_ASSETS_DIR / provider
        super().__init__(_T, path=path, **kwargs)
        self._provider: str = provider

        self._formats = os.environ.get('ARENA_MODELS_FORMATS', '').split(',')
        self._ttl = int(os.environ.get('ARENA_MODELS_TTL', '86400'))
        self._pending: dict[str, list[asyncio.Future[Path | None]]] = {}
        self._flush_scheduled = False
        self._batch_lock = asyncio.Lock()

    @classmethod
    async def check_output_async(cls, args: Iterable[str], **kwargs: object) -> bytes:
        process = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, **kwargs)
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode or -1, list(args), output=stdout, stderr=stderr)
        return stdout

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
        if result is not None:
            self._cache[identifier] = result

        return self._cache.get(identifier, None)

    def listall(self, *, network: bool = False, **kwargs: object) -> Iterator[IdentifierT]:
        """
        List all assets available. When *network* is True, also queries the
        remote provider via ``ros2 run arena_models arena_models … list``.
        """
        yield from super(SimplePathResolver, self).listall(**kwargs)
        if network:
            try:
                output = subprocess.check_output(
                    [
                        'ros2',
                        'run',
                        'arena_models',
                        'arena_models',
                        '-s',
                        'net',
                        self._provider,
                        'list',
                    ],
                    stderr=subprocess.DEVNULL,
                )
                for line in output.decode().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield self._IdentifierT.from_relpath(Path(line))
                    except Exception:
                        pass
            except subprocess.CalledProcessError:
                import traceback

                logging.warning(traceback.format_exc())

    def __repr__(self) -> str:
        return f"{super().__repr__()}(net={self._provider})"

    @classmethod
    def all(cls, _T: type[IdentifierT], /) -> Iterable[Self]:
        for provider in NETWORK_PROVIDERS:
            yield cls(_T, provider=provider)


class FallbackResolver(ResolverBase[IdentifierT], typing.Generic[IdentifierT]):
    """
    Resolver that always yields a fallback path.
    """

    def __init__(self, _T: type[IdentifierT], /, path: Path, **kwargs: object) -> None:
        super().__init__(_T, **kwargs)
        self._path = path

    async def resolve(self, identifier: IdentifierT) -> Path | None:
        return self._path / identifier.relpath()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self._path)})"


# IDENTIFIERS


T = typing.TypeVar('T')
T_co = typing.TypeVar('T_co', covariant=True)


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


@attrs.define(eq=False, hash=False)
class Identifier(IdentifierProtocol[T], Parseable, Serializable, Idempotent, typing.Generic[T]):
    """Represents an identifier referencing an path."""

    name: str

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

    async def resolve(self, **kwargs: object) -> T:
        path = await self.resolve_path()
        return await asyncio.to_thread(self.load, path, **kwargs)

    def resolve_sync(self, **kwargs: object) -> T:
        """Synchronously load the asset referenced by this identifier."""
        result: T = None  # type: ignore
        exc: Exception | None = None

        def _run_async_load():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                nonlocal result
                result = loop.run_until_complete(self.resolve(**kwargs))
            except Exception as e:
                nonlocal exc
                exc = e
            finally:
                loop.close()

        thread = threading.Thread(target=_run_async_load)
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
