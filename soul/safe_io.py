"""Open output files without following links or writing through swapped paths."""

import os
import stat
from contextlib import contextmanager
from pathlib import Path


def _is_reparse_point(metadata):
    reparse_attribute = getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, 'st_file_attributes', 0) & reparse_attribute
    )


def _file_identity(metadata):
    identity = (
        metadata.st_dev, metadata.st_ino, metadata.st_mode,
        metadata.st_size, metadata.st_mtime_ns,
    )
    # On Windows st_ctime is creation time; it is not a mutation timestamp.
    if os.name == 'nt':
        return identity
    return (*identity, metadata.st_ctime_ns)


@contextmanager
def open_output(path, *, append=False, binary=False, newline=None):
    """Yield a verified handle to `path`, refusing links and path swaps."""
    path = Path(path)
    try:
        before = path.lstat()
    except FileNotFoundError:
        before = None
    except OSError as error:
        raise ValueError(f'output path cannot be inspected: {path}: {error}') from error

    if before is not None and (
        _is_reparse_point(before) or not stat.S_ISREG(before.st_mode)
    ):
        raise ValueError(f'output must be a regular non-link file: {path}')

    flags = (
        os.O_WRONLY
        | (os.O_APPEND if append else 0)
        | getattr(os, 'O_BINARY', 0)
        | getattr(os, 'O_NONBLOCK', 0)
    )
    if hasattr(os, 'O_NOFOLLOW'):
        flags |= os.O_NOFOLLOW
    # A missing path is created exclusively, so a link inserted after lstat()
    # cannot be followed on platforms without O_NOFOLLOW.
    if before is None:
        flags |= os.O_CREAT | os.O_EXCL

    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise ValueError(f'output could not be opened safely: {path}: {error}') from error

    try:
        opened = os.fstat(descriptor)
        if _is_reparse_point(opened) or not stat.S_ISREG(opened.st_mode):
            raise ValueError(f'output is not a regular file: {path}')
        if before is not None and _file_identity(before) != _file_identity(opened):
            raise ValueError(f'output changed while it was being opened: {path}')
        try:
            after = path.lstat()
        except OSError as error:
            raise ValueError(f'output changed while it was being opened: {path}') from error
        if (
            _is_reparse_point(after)
            or not stat.S_ISREG(after.st_mode)
            or _file_identity(after) != _file_identity(opened)
        ):
            raise ValueError(f'output changed while it was being opened: {path}')

        mode = ('ab' if append else 'wb') if binary else ('a' if append else 'w')
        options = {} if binary else {'encoding': 'utf-8'}
        if not binary and not append:
            options['newline'] = '\n' if newline is None else newline
        with os.fdopen(descriptor, mode, **options) as stream:
            descriptor = None
            if not append:
                stream.seek(0)
                stream.truncate()
            yield stream
            stream.flush()
    finally:
        if descriptor is not None:
            os.close(descriptor)
def read_regular_bytes(path, *, max_bytes, label):
    """Read a bounded regular file without following links or blocking on FIFOs."""
    path = Path(path)
    flags = (os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
             | getattr(os, 'O_BINARY', 0))
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f'{label} cannot be opened safely: {path}: {error}') from error
    try:
        metadata = os.fstat(descriptor)
        if _is_reparse_point(metadata) or not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f'{label} must be a regular non-link file: {path}')
        if metadata.st_size > max_bytes:
            raise ValueError(
                f'{label} exceeds the {max_bytes}-byte limit: {path}')
        chunks = []
        total = 0
        while total <= max_bytes:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(
                    f'{label} exceeds the {max_bytes}-byte limit: {path}')
            chunks.append(chunk)
        return b''.join(chunks)
    finally:
        os.close(descriptor)

def write_text_safe(path, text, *, encoding='utf-8'):
    """Write text through a verified handle instead of a path."""
    with open_output(path, append=False) as stream:
        stream.write(text)