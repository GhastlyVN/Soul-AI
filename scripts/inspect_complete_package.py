#!/usr/bin/env python3
"""Inspect a SOUL AI artifact ZIP without extracting or deserializing model files.

The output is an inventory and integrity report, not proof that a model is safe to execute
or that the package's claimed training metrics are reproducible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
import unicodedata
import zipfile

MAX_ARCHIVE_ENTRIES = 100_000
# The uncompressed total is capped at 8 GiB; allow reasonable ZIP metadata overhead,
# but do not hash or parse an arbitrarily large input archive.
MAX_ARCHIVE_BYTES = 9 * 1024**3
MAX_MEMBER_UNCOMPRESSED_BYTES = 2 * 1024**3
MAX_TOTAL_UNCOMPRESSED_BYTES = 8 * 1024**3
MAX_JSON_BYTES = 32 * 1024**2
MAX_PATH_UTF8_BYTES = 4096
MAX_PATH_COMPONENTS = 64
MAX_COMPONENT_UTF8_BYTES = 255
MAX_COMPONENT_UTF16_UNITS = 255
CHUNK_SIZE = 1024 * 1024

WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL", "CLOCK$", "CONIN$", "CONOUT$",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
    "COM¹", "COM²", "COM³", "LPT¹", "LPT²", "LPT³",
}


def _portable_windows_component(part: str) -> bool:
    """Reject names that alias devices or alternate streams on Windows."""
    stem = part.split(".", 1)[0].rstrip(" .").upper()
    return (
        bool(part)
        and len(part.encode("utf-8")) <= MAX_COMPONENT_UTF8_BYTES
        and len(part.encode("utf-16-le")) // 2 <= MAX_COMPONENT_UTF16_UNITS
        and not part.endswith((" ", "."))
        and not any(ord(char) < 32 or char in '<>:"|?*' for char in part)
        and stem not in WINDOWS_RESERVED_NAMES
    )




def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _collision_key(value: str) -> str:
    """Normalize canonical Unicode equivalents before case-insensitive comparison."""
    normalized = unicodedata.normalize("NFC", value)
    return unicodedata.normalize("NFC", normalized.casefold())


def _validate_name(info: zipfile.ZipInfo) -> tuple[str, bool]:
    raw = info.filename.replace(chr(92), "/")
    path = PurePosixPath(raw)
    is_directory = info.is_dir() or raw.endswith("/")
    try:
        path_bytes = len(raw.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise ValueError(f"ZIP path is not valid Unicode: {info.filename}") from error
    if path_bytes > MAX_PATH_UTF8_BYTES or len(path.parts) > MAX_PATH_COMPONENTS:
        raise ValueError(f"ZIP path exceeds portable path limits: {info.filename}")
    if (
        not path.parts
        or raw.startswith("/")
        or chr(0) in raw
        or ".." in path.parts
        or any(not _portable_windows_component(part) for part in path.parts)
    ):
        raise ValueError(f"unsafe ZIP path: {info.filename}")
    mode = (info.external_attr >> 16) & 0o170000
    if mode == stat.S_IFLNK:
        raise ValueError(f"symlink entry refused: {info.filename}")
    if mode not in (0, stat.S_IFREG, stat.S_IFDIR):
        raise ValueError(f"special ZIP entry refused: {info.filename}")
    if mode == stat.S_IFDIR and not is_directory:
        raise ValueError(f"directory mode/name mismatch: {info.filename}")
    if mode == stat.S_IFREG and is_directory:
        raise ValueError(f"file mode/name mismatch: {info.filename}")
    return "/".join(path.parts), is_directory


def _hash_archive_bounded(stream, snapshot=None):
    """Hash one bounded input stream and optionally copy the exact bytes to a snapshot."""
    archive_hash = hashlib.sha256()
    total_bytes = 0
    while True:
        # Read at most the remaining allowance plus one byte. The extra byte is
        # enough to prove the stream crossed the limit without consuming a full
        # chunk beyond the configured resource ceiling.
        remaining = MAX_ARCHIVE_BYTES - total_bytes
        chunk = stream.read(min(CHUNK_SIZE, remaining + 1))
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > MAX_ARCHIVE_BYTES:
            raise ValueError(
                f"ZIP archive exceeds inspection size limit while hashing: "
                f"{total_bytes} (limit {MAX_ARCHIVE_BYTES})"
            )
        archive_hash.update(chunk)
        if snapshot is not None:
            snapshot.write(chunk)
    return archive_hash


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    """Use portable file identity fields; Windows ctime is creation time, not change time."""
    identity = (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )
    if os.name != "nt":
        return (*identity, metadata.st_ctime_ns)
    return identity


def inspect_package(archive_path: Path) -> dict[str, object]:
    archive_path = Path(archive_path)
    try:
        path_before = archive_path.lstat()
    except OSError as error:
        raise ValueError(f"not a readable ZIP archive: {archive_path}") from error
    if not stat.S_ISREG(path_before.st_mode):
        raise ValueError(f"not a regular non-symlink ZIP archive: {archive_path}")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(archive_path, flags)
    except OSError as error:
        raise ValueError(f"could not open ZIP archive safely: {archive_path}: {error}") from error

    # Hash and inspect one private disk-backed snapshot. This prevents a path swap
    # or concurrent source mutation from making the report hash describe different
    # bytes than the ZIP metadata/member inventory. The 9 GiB input cap bounds disk use.
    with os.fdopen(descriptor, "rb") as raw_archive, tempfile.TemporaryFile(mode="w+b") as snapshot:
        before = os.fstat(raw_archive.fileno())
        if (
            not stat.S_ISREG(before.st_mode)
            or _file_identity(before) != _file_identity(path_before)
        ):
            raise ValueError(f"ZIP archive changed while it was being opened: {archive_path}")
        if before.st_size > MAX_ARCHIVE_BYTES:
            raise ValueError(
                f"ZIP archive exceeds inspection size limit: {before.st_size} "
                f"(limit {MAX_ARCHIVE_BYTES})"
            )

        archive_hash = _hash_archive_bounded(raw_archive, snapshot)
        if snapshot.tell() != before.st_size:
            raise ValueError(f"ZIP archive size changed while creating inspection snapshot: {archive_path}")
        snapshot.flush()
        snapshot.seek(0)
        if not zipfile.is_zipfile(snapshot):
            raise ValueError(f"not a readable ZIP archive: {archive_path}")
        snapshot.seek(0)

        inventory: list[dict[str, object]] = []
        seen: set[str] = set()
        seen_folded: set[str] = set()
        files_folded: set[str] = set()
        prefix_spellings: dict[str, str] = {}
        directories = 0
        total_uncompressed = 0
        actual_total_uncompressed = 0

        with zipfile.ZipFile(snapshot) as zf:
            entries = zf.infolist()
            if not entries:
                raise ValueError("empty ZIP archive")
            if len(entries) > MAX_ARCHIVE_ENTRIES:
                raise ValueError(
                    f"too many ZIP entries: {len(entries)} (limit {MAX_ARCHIVE_ENTRIES})"
                )

            validated: list[tuple[zipfile.ZipInfo, str, bool]] = []
            for info in entries:
                name, is_directory = _validate_name(info)
                folded = _collision_key(name)
                if name in seen or folded in seen_folded:
                    raise ValueError(
                        f"duplicate or case-insensitive ZIP path collision: {info.filename}"
                    )
                parts = PurePosixPath(name).parts
                parents = [
                    "/".join(parts[:i])
                    for i in range(1, len(parts))
                ]
                conflict = next(
                    (
                        _collision_key(parent)
                        for parent in parents
                        if _collision_key(parent) in files_folded
                    ),
                    None,
                )
                if conflict is not None:
                    raise ValueError(
                        f"file/directory path collision: {info.filename} is below {conflict}"
                    )
                prefix_count = len(parts) if is_directory else len(parts) - 1
                for index in range(1, prefix_count + 1):
                    prefix = "/".join(parts[:index])
                    folded_prefix = _collision_key(prefix)
                    previous_prefix = prefix_spellings.get(folded_prefix)
                    if previous_prefix is not None and previous_prefix != prefix:
                        raise ValueError(
                            f"case-insensitive ZIP directory prefix collision: {info.filename}"
                        )
                    prefix_spellings[folded_prefix] = prefix
                if not is_directory and folded in prefix_spellings:
                    raise ValueError(
                        f"file/directory path collision: {info.filename} replaces a directory"
                    )
                if info.flag_bits & 0x1:
                    raise ValueError(f"encrypted ZIP entry refused: {info.filename}")
                if not is_directory:
                    if info.file_size < 0 or info.file_size > MAX_MEMBER_UNCOMPRESSED_BYTES:
                        raise ValueError(f"ZIP member exceeds size limit: {info.filename}")
                    total_uncompressed += info.file_size
                    if total_uncompressed > MAX_TOTAL_UNCOMPRESSED_BYTES:
                        raise ValueError("ZIP archive exceeds total uncompressed size limit")
                    files_folded.add(folded)
                else:
                    directories += 1
                seen.add(name)
                seen_folded.add(folded)
                validated.append((info, name, is_directory))

            for info, name, is_directory in validated:
                if is_directory:
                    inventory.append({"path": name, "kind": "directory"})
                    continue
                digest = hashlib.sha256()
                json_bytes = bytearray() if name.lower().endswith(".json") else None
                actual_size = 0
                with zf.open(info) as source:
                    while chunk := source.read(CHUNK_SIZE):
                        actual_size += len(chunk)
                        actual_total_uncompressed += len(chunk)
                        if actual_size > MAX_MEMBER_UNCOMPRESSED_BYTES:
                            raise ValueError(
                                f"actual ZIP member size exceeds limit: {name}"
                            )
                        if actual_total_uncompressed > MAX_TOTAL_UNCOMPRESSED_BYTES:
                            raise ValueError(
                                "actual ZIP archive exceeds total uncompressed size limit"
                            )
                        digest.update(chunk)
                        if json_bytes is not None:
                            if len(json_bytes) + len(chunk) > MAX_JSON_BYTES:
                                raise ValueError(f"JSON manifest exceeds inspection limit: {name}")
                            json_bytes.extend(chunk)
                if actual_size != info.file_size:
                    raise ValueError(f"ZIP member size mismatch: {name}")
                if json_bytes is not None:
                    try:
                        json.loads(bytes(json_bytes), object_pairs_hook=_reject_duplicate_keys)
                    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as error:
                        raise ValueError(f"invalid JSON file {name}: {error}") from error
                inventory.append({
                    "path": name,
                    "kind": "file",
                    "bytes": actual_size,
                    "sha256": digest.hexdigest(),
                    "json_validated": json_bytes is not None,
                })

        after = os.fstat(raw_archive.fileno())
        try:
            path_after = archive_path.lstat()
        except OSError as error:
            raise ValueError("ZIP archive path changed during inspection") from error
        if (
            not stat.S_ISREG(path_after.st_mode)
            or _file_identity(before) != _file_identity(after)
            or _file_identity(after) != _file_identity(path_after)
        ):
            raise ValueError("ZIP archive changed during inspection")

    files = [item for item in inventory if item["kind"] == "file"]
    return {
        "schema_version": 1,
        "archive": archive_path.name,
        "archive_sha256": archive_hash.hexdigest(),
        "summary": {
            "entries": len(inventory),
            "files": len(files),
            "directories": directories,
            "uncompressed_bytes": total_uncompressed,
        },
        "inventory": inventory,
        "limitations": [
            "Model weights and NumPy archives are hashed but never loaded or deserialized.",
            "JSON syntax and duplicate keys are checked; package-specific manifest hash references are not interpreted.",
            "This inventory does not reproduce training metrics or establish model safety.",
        ],
    }


def _atomic_write_text(destination: Path, content: str) -> None:
    """Write a complete report beside its destination, then atomically replace it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="SOUL AI artifact ZIP")
    parser.add_argument("--output", type=Path, help="Optional path for the JSON inventory report")
    args = parser.parse_args()
    if args.output is not None:
        same_path = args.output.resolve() == args.archive.resolve()
        same_file = (
            args.output.exists()
            and args.archive.exists()
            and args.output.samefile(args.archive)
        )
        if same_path or same_file:
            parser.error("--output must not overwrite the input archive")
    try:
        report = inspect_package(args.archive)
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as error:
        parser.exit(2, f"Package inspection failed: {error}\n")
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        _atomic_write_text(args.output, rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())