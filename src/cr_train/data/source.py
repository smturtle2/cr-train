from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .constants import CACHE_LAYOUT_VERSION, CATALOG_METADATA_WORKERS, DATASETS_SERVER_BASE, StartupCallback
from .store import read_json, write_json_atomic


def emit_startup_event(startup_callback: StartupCallback | None, **event: Any) -> None:
    if startup_callback is not None:
        startup_callback(event)


def run_startup_stage(
    startup_callback: StartupCallback | None,
    *,
    stage: str,
    split: str,
    operation,
    **fields: Any,
):
    import time

    emit_startup_event(startup_callback, stage=stage, split=split, status="start", **fields)
    started_at = time.perf_counter()
    try:
        result = operation()
    except Exception as exc:
        emit_startup_event(
            startup_callback,
            stage=stage,
            split=split,
            status="error",
            elapsed_sec=time.perf_counter() - started_at,
            error=str(exc),
            **fields,
        )
        raise

    emit_startup_event(
        startup_callback,
        stage=stage,
        split=split,
        status="done",
        elapsed_sec=time.perf_counter() - started_at,
        **fields,
    )
    return result


def resolve_layout_root(cache_root: Path) -> Path:
    path = cache_root / f"layout-v{CACHE_LAYOUT_VERSION}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_source_root(cache_root: Path, source_signature: str) -> Path:
    path = resolve_layout_root(cache_root) / source_signature
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_catalog_root(source_root: Path) -> Path:
    path = source_root / "catalogs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_catalog_path(source_root: Path, split: str) -> Path:
    return resolve_catalog_root(source_root) / f"{split}.json"


def resolve_source_metadata_path(source_root: Path) -> Path:
    return source_root / "source.json"


def request_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError(f"unexpected JSON payload from {url}")
    return payload


def datasets_server_url(endpoint: str, *, dataset_name: str, revision: str | None) -> str:
    query = {"dataset": dataset_name}
    if revision is not None:
        query["revision"] = revision
    return f"{DATASETS_SERVER_BASE}/{endpoint}?{urllib.parse.urlencode(query)}"


def normalize_parquet_uri(url: str) -> str:
    if url.startswith("hf://"):
        return url

    parsed = urllib.parse.urlparse(url)
    path_parts = [part for part in parsed.path.split("/") if part]
    try:
        datasets_index = path_parts.index("datasets")
        resolve_index = path_parts.index("resolve")
    except ValueError:
        return url

    if resolve_index - datasets_index != 3:
        return url

    repo_id = "/".join(path_parts[datasets_index + 1:resolve_index])
    revision = urllib.parse.unquote(path_parts[resolve_index + 1])
    file_path = "/".join(path_parts[resolve_index + 2:])
    if not repo_id or not revision or not file_path:
        return url
    return f"hf://datasets/{repo_id}@{revision}/{file_path}"


def normalize_split_sizes(info_payload: dict[str, Any]) -> dict[str, int]:
    dataset_info = info_payload.get("dataset_info")
    if not isinstance(dataset_info, dict):
        raise ValueError("dataset_info payload is missing")
    default_config = dataset_info.get("default")
    if not isinstance(default_config, dict):
        raise ValueError("default config payload is missing")
    splits = default_config.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("split payload is missing")
    return {
        str(split): int(split_info["num_examples"])
        for split, split_info in splits.items()
        if isinstance(split_info, dict)
    }


def normalize_parquet_files(parquet_payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    parquet_files = parquet_payload.get("parquet_files")
    if not isinstance(parquet_files, list):
        raise ValueError("parquet_files payload is missing")

    normalized: dict[str, list[dict[str, Any]]] = {}
    for entry in parquet_files:
        if not isinstance(entry, dict):
            continue
        split = str(entry["split"])
        normalized.setdefault(split, []).append(
            {
                "url": normalize_parquet_uri(str(entry["url"])),
                "filename": str(entry["filename"]),
                "config": str(entry.get("config", "default")),
            }
        )

    for files in normalized.values():
        files.sort(key=lambda item: item["filename"])
    return normalized


def load_source_descriptor(dataset_name: str, revision: str | None) -> dict[str, Any]:
    info_payload = request_json(datasets_server_url("info", dataset_name=dataset_name, revision=revision))
    parquet_payload = request_json(datasets_server_url("parquet", dataset_name=dataset_name, revision=revision))
    split_sizes = normalize_split_sizes(info_payload)
    parquet_files_by_split = normalize_parquet_files(parquet_payload)
    signature_payload = {
        "cache_layout_version": CACHE_LAYOUT_VERSION,
        "dataset_name": dataset_name,
        "revision": revision,
        "split_sizes": split_sizes,
        "split_urls": {
            split: [entry["url"] for entry in parquet_files_by_split.get(split, [])]
            for split in sorted(split_sizes)
        },
    }
    source_signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    return {
        "dataset_name": dataset_name,
        "revision": revision,
        "source_signature": source_signature,
        "split_sizes": split_sizes,
        "parquet_files_by_split": parquet_files_by_split,
    }


def ensure_source_root(
    *,
    dataset_name: str,
    revision: str | None,
    cache_root: Path,
) -> tuple[Path, dict[str, Any]]:
    descriptor = load_source_descriptor(dataset_name, revision)
    source_root = resolve_source_root(cache_root, str(descriptor["source_signature"]))
    metadata_path = resolve_source_metadata_path(source_root)
    if not metadata_path.exists():
        write_json_atomic(
            metadata_path,
            {
                "cache_layout_version": CACHE_LAYOUT_VERSION,
                "dataset_name": descriptor["dataset_name"],
                "revision": descriptor["revision"],
                "source_signature": descriptor["source_signature"],
                "split_sizes": descriptor["split_sizes"],
            },
        )
    return source_root, descriptor


def read_parquet_shard_metadata(entry: dict[str, Any]) -> dict[str, Any]:
    parquet_file = pq.ParquetFile(str(entry["url"]))
    metadata = parquet_file.metadata
    return {
        "url": str(entry["url"]),
        "filename": str(entry["filename"]),
        "row_count": int(metadata.num_rows),
        "row_group_rows": [
            int(metadata.row_group(index).num_rows)
            for index in range(metadata.num_row_groups)
        ],
    }


def build_catalog(
    *,
    split: str,
    total_rows: int,
    parquet_files: list[dict[str, Any]],
) -> dict[str, Any]:
    if total_rows > 0 and not parquet_files:
        raise ValueError(f"missing parquet files for split {split}")

    max_workers = min(CATALOG_METADATA_WORKERS, max(1, len(parquet_files)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        shard_metadata = list(executor.map(read_parquet_shard_metadata, parquet_files))

    shards: list[dict[str, Any]] = []
    current_start = 0
    total_row_groups = 0
    for shard_index, shard in enumerate(shard_metadata):
        row_count = int(shard["row_count"])
        row_group_rows = [int(value) for value in shard["row_group_rows"]]
        current_stop = current_start + row_count
        shards.append(
            {
                "index": shard_index,
                "url": str(shard["url"]),
                "filename": str(shard["filename"]),
                "row_count": row_count,
                "global_start": current_start,
                "global_stop": current_stop,
                "row_group_rows": row_group_rows,
            }
        )
        current_start = current_stop
        total_row_groups += len(row_group_rows)

    if current_start != total_rows:
        raise ValueError(
            f"catalog row count mismatch for split {split}: expected {total_rows}, got {current_start}"
        )

    return {
        "cache_layout_version": CACHE_LAYOUT_VERSION,
        "split": split,
        "total_rows": total_rows,
        "total_row_groups": total_row_groups,
        "shards": shards,
    }


def _write_and_reload_catalog(*, catalog_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    write_json_atomic(catalog_path, payload)
    return read_json(catalog_path)


def ensure_split_catalog(
    *,
    source_root: Path,
    descriptor: dict[str, Any],
    split: str,
    startup_callback: StartupCallback | None,
) -> dict[str, Any]:
    catalog_path = resolve_catalog_path(source_root, split)
    if catalog_path.exists():
        return read_json(catalog_path)

    split_sizes = descriptor["split_sizes"]
    if split not in split_sizes:
        raise KeyError(f"split {split!r} does not exist in source descriptor")

    return run_startup_stage(
        startup_callback,
        stage="ensure catalog",
        split=split,
        operation=lambda: _write_and_reload_catalog(
            catalog_path=catalog_path,
            payload=build_catalog(
                split=split,
                total_rows=int(split_sizes[split]),
                parquet_files=list(descriptor["parquet_files_by_split"].get(split, [])),
            ),
        ),
        total_rows=int(split_sizes[split]),
    )


__all__ = [
    "datasets_server_url",
    "emit_startup_event",
    "ensure_source_root",
    "ensure_split_catalog",
    "normalize_parquet_uri",
    "pq",
    "request_json",
    "resolve_catalog_path",
    "run_startup_stage",
]
