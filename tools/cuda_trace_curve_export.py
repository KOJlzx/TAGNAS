#!/usr/bin/env python3
"""Export GPU kernel-active time fractions from an Nsight Systems SQLite trace.

These are interval-union time fractions, not SM Activity or hardware occupancy.
CPU CUDA API/NVTX durations never contribute to device-active time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import quote


NANOSECONDS = 1_000_000_000
CATEGORIES = ("non_nccl_kernel", "nccl_kernel", "memcpy", "memset")
COLORS = {"non_nccl_kernel": "#2369a5", "nccl_kernel": "#d17720", "memcpy": "#399352", "memset": "#9467bd"}
LABELS = {"non_nccl_kernel": "Non-NCCL kernels", "nccl_kernel": "NCCL kernels", "memcpy": "Memory copies",
          "memset": "Memory sets", "any_kernel": "Any kernel"}


@dataclass(frozen=True)
class Activity:
    start: int
    end: int
    device_id: int
    category: str
    name: str
    global_pid: int | None = None
    context_id: int | None = None
    stream_id: int | None = None
    correlation_id: int | None = None


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return disjoint intervals, counting concurrent streams only once."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            raise ValueError(f"Nonpositive activity interval: {start}, {end}")
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def interval_fractions(intervals: Iterable[tuple[int, int]], begin: int, end: int,
                       window: int) -> list[float]:
    """Clip and union GPU intervals, then divide by each actual bin width."""
    if end <= begin or window <= 0:
        raise ValueError("A positive time range and window are required")
    bins = (end - begin + window - 1) // window
    active = [0] * bins
    clipped = ((max(start, begin), min(stop, end)) for start, stop in intervals if stop > begin and start < end)
    for start, stop in merge_intervals(clipped):
        first = (start - begin) // window
        last = (stop - 1 - begin) // window
        for index in range(first, last + 1):
            lower = begin + index * window
            upper = min(lower + window, end)
            active[index] += min(stop, upper) - max(start, lower)
    return [value / min(window, end - begin - index * window) for index, value in enumerate(active)]


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}


def capture_bounds(sqlite_path: Path, expected_ranks: Iterable[int] | None = None,
                   requested_duration_seconds: float = 40) -> dict:
    """Find the real common capture interval from explicit per-rank NVTX marks.

    A valid interval may be shorter than the requested duration. It is never
    extended using the request or the first/last observed GPU kernel.
    """
    if not math.isfinite(requested_duration_seconds) or requested_duration_seconds <= 0:
        raise ValueError("Requested duration must be positive and finite")
    expected = None if expected_ranks is None else sorted(set(int(rank) for rank in expected_ranks))
    if expected is not None and (not expected or expected[0] < 0):
        raise ValueError("Expected ranks must be a nonempty set of nonnegative ranks")
    result = dict(status="missing", expected_ranks=expected, expected_ranks_inferred=expected is None,
                  requested_duration_seconds=requested_duration_seconds, requested_duration_met=False,
                  common_start_ns=None, common_end_ns=None, common_duration_seconds=None, markers={},
                  rule="Intersection of actual per-rank NVTX START/STOP mark timestamps; no duration padding.")
    connection = sqlite3.connect(f"file:{quote(str(sqlite_path.resolve()))}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        tables = _tables(connection)
        if "NVTX_EVENTS" not in tables:
            return {**result, "reason": "NVTX_EVENTS table absent"}
        columns = _columns(connection, "NVTX_EVENTS")
        if "start" not in columns or not {"text", "textId"} & columns:
            return {**result, "reason": "NVTX_EVENTS lacks timestamp or label columns"}
        strings = ({row[0]: row[1] for row in connection.execute("SELECT id,value FROM StringIds")}
                   if "StringIds" in tables else {})
        selected = [key for key in ("start", "end", "text", "textId", "globalTid", "eventType") if key in columns]
        query = "SELECT " + ",".join(f'"{key}"' for key in selected) + " FROM NVTX_EVENTS ORDER BY start"
        markers = defaultdict(lambda: {"start": [], "stop": []})
        for row in connection.execute(query):
            value = dict(row)
            label = strings.get(value.get("textId")) or value.get("text")
            match = re.fullmatch(r"CUDA_TRACE_CAPTURE_(START|STOP)_RANK_(\d+)", str(label))
            if not match:
                continue
            marker = {"timestamp_ns": int(value["start"]), "label": str(label),
                      "global_tid": value.get("globalTid"), "event_type": value.get("eventType")}
            markers[int(match[2])][match[1].lower()].append(marker)
        result["markers"] = dict(markers)
        observed = sorted(markers)
        ranks = expected if expected is not None else observed
        result["observed_ranks"] = observed
        if not ranks:
            return {**result, "reason": "No explicit capture boundary marks found"}
        if expected is not None and set(observed) - set(expected):
            return {**result, "status": "invalid", "reason": "Unexpected ranks in capture markers"}
        missing = [rank for rank in ranks if not markers[rank]["start"] or not markers[rank]["stop"]]
        if missing:
            return {**result, "missing_ranks": missing, "reason": "Missing start or stop capture mark"}
        duplicate = [rank for rank in ranks if len(markers[rank]["start"]) != 1 or len(markers[rank]["stop"]) != 1]
        if duplicate:
            return {**result, "status": "ambiguous", "ambiguous_ranks": duplicate,
                    "reason": "Each rank must have exactly one start and one stop mark"}
        starts, stops, per_rank = [], [], []
        for rank in ranks:
            start, stop = markers[rank]["start"][0], markers[rank]["stop"][0]
            if stop["timestamp_ns"] <= start["timestamp_ns"]:
                return {**result, "status": "invalid", "reason": f"Nonpositive capture duration for rank {rank}"}
            if start["global_tid"] is not None and stop["global_tid"] is not None:
                if (int(start["global_tid"]) >> 24) != (int(stop["global_tid"]) >> 24):
                    return {**result, "status": "invalid", "reason": f"Rank {rank} marks belong to different processes"}
            starts.append(start["timestamp_ns"])
            stops.append(stop["timestamp_ns"])
            per_rank.append({"rank": rank, "start_ns": start["timestamp_ns"], "end_ns": stop["timestamp_ns"],
                             "duration_seconds": (stop["timestamp_ns"] - start["timestamp_ns"]) / NANOSECONDS})
        begin, end = max(starts), min(stops)
        if end <= begin:
            return {**result, "status": "invalid", "per_rank": per_rank,
                    "reason": "No common capture interval across expected ranks"}
        return {**result, "status": "valid", "common_start_ns": begin, "common_end_ns": end,
                "common_duration_seconds": (end - begin) / NANOSECONDS, "per_rank": per_rank,
                "requested_duration_met": end - begin >= round(requested_duration_seconds * NANOSECONDS)}
    finally:
        connection.close()


def _normal_uuid(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.hex() if len(value) == 16 else None
    text = str(value).strip().lower()
    if text.startswith("gpu-"):
        text = text[4:]
    text = text.strip("{}").replace("-", "")
    return text if re.fullmatch(r"[0-9a-f]{32}", text) else None


def _normal_pci(value) -> tuple[int | None, int, int, int] | None:
    if value is None:
        return None
    match = re.fullmatch(r"(?:(?P<domain>[0-9a-f]{1,8}):)?(?P<bus>[0-9a-f]{1,2}):"
                         r"(?P<device>[0-9a-f]{1,2})\.(?P<function>[0-7])", str(value).strip().lower())
    if not match:
        return None
    domain = int(match["domain"], 16) if match["domain"] is not None else None
    return domain, int(match["bus"], 16), int(match["device"], 16), int(match["function"], 16)


def resolve_device_map(target_info_gpu: dict[int, dict], inventory: list[dict]) -> dict:
    """Resolve *metadata* physical indices from unique UUID/PCI evidence.

    Names and CUDA/Nsight ordinals are deliberately not matching evidence.
    The inventory must be the driver's pre-capture nvidia-smi snapshot.
    This does not associate process-local CUPTI device IDs with these records;
    use ``resolve_activity_devices`` for actual activity-to-physical mapping.
    """
    prepared = []
    for row in inventory:
        value = dict(row)
        try:
            index = int(value["index"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("GPU inventory rows require an integer physical index") from error
        if index < 0:
            raise ValueError("GPU inventory indices must be nonnegative")
        prepared.append({"index": index, "uuid": _normal_uuid(value.get("uuid")),
                         "pci": _normal_pci(value.get("pci.bus_id", value.get("pci_bus_id", value.get("busLocation")))),
                         "raw": value})
    mapping, devices = {}, []
    for device, metadata in sorted(target_info_gpu.items(), key=lambda pair: int(pair[0])):
        device = int(device)
        uuid = _normal_uuid(metadata.get("uuid"))
        pci = _normal_pci(metadata.get("busLocation", metadata.get("pci.bus_id", metadata.get("pci_bus_id"))))
        by_uuid = [row for row in prepared if uuid is not None and row["uuid"] == uuid]
        by_pci = [row for row in prepared if pci is not None and row["pci"] is not None and
                  pci[1:] == row["pci"][1:] and
                  (pci[0] is None or row["pci"][0] is None or pci[0] == row["pci"][0])]
        record = {"nsys_device_id": device, "status": "unmapped", "nsys_metadata": metadata,
                  "uuid_candidate_indices": [row["index"] for row in by_uuid],
                  "pci_candidate_indices": [row["index"] for row in by_pci], "evidence": []}
        candidate = None
        if len(by_uuid) > 1 or len(by_pci) > 1:
            record.update(status="ambiguous", reason="UUID or PCI matches multiple inventory rows")
        elif by_uuid and by_pci and by_uuid[0]["index"] != by_pci[0]["index"]:
            record.update(status="conflict", reason="UUID and PCI identify different physical GPUs")
        else:
            candidate = (by_uuid or by_pci or [None])[0]
            if candidate is not None:
                if uuid is not None and candidate["uuid"] is not None and uuid != candidate["uuid"]:
                    record.update(status="conflict", reason="UUID disagrees with PCI-matched inventory GPU")
                    candidate = None
                elif pci is not None and candidate["pci"] is not None and (
                        pci[1:] != candidate["pci"][1:] or
                        (pci[0] is not None and candidate["pci"][0] is not None and pci[0] != candidate["pci"][0])):
                    record.update(status="conflict", reason="PCI disagrees with UUID-matched inventory GPU")
                    candidate = None
                elif sum(row["index"] == candidate["index"] for row in prepared) != 1:
                    record.update(status="ambiguous", reason="Physical index appears repeatedly in inventory")
                    candidate = None
        if candidate is not None and record["status"] == "unmapped":
            record.update(status="verified", physical_index=candidate["index"], inventory_gpu=candidate["raw"],
                          evidence=(["unique_uuid"] if by_uuid else []) + (["unique_pci"] if by_pci else []))
            mapping[device] = f"Physical GPU {candidate['index']}"
        devices.append(record)
    return {"mapping": mapping, "devices": devices,
            "rule": "Unique consistent UUID/PCI match only; names and device ordinals are never inferred."}


def activity_scope(event: Activity) -> str:
    """CUDA local device/context IDs are scoped by the traced process."""
    return f"{event.global_pid}:{event.device_id}:{event.context_id}"


def resolve_activity_devices(sqlite_path: Path, inventory: list[dict], snapshots: list[dict] | dict,
                             *, expected_cards: Iterable[int | str] | None = None,
                             events: list[Activity] | None = None) -> dict:
    """Map activity scopes through PROCESSES.pid and owned NVML PID/UUID proof.

    NVML snapshots must come from this owned attempt. Their ``owned_pids``
    have already been identity checked by the collector. Alternatively, an
    explicit ``process_identities`` record requires a marker and birth ticks.
    A PID observed on multiple GPUs is rejected rather than guessing which
    local device ordinal belongs to which physical device.
    """
    if events is None:
        events, _, _ = read_sqlite(sqlite_path)
    snapshots = [snapshots] if isinstance(snapshots, dict) else list(snapshots)
    connection = sqlite3.connect(f"file:{quote(str(sqlite_path.resolve()))}?mode=ro", uri=True)
    try:
        if "PROCESSES" not in _tables(connection) or not {"globalPid", "pid"} <= _columns(connection, "PROCESSES"):
            return {"status": "unresolved", "activity_device_map": {}, "mapping": {}, "devices": [],
                    "errors": ["PROCESSES globalPid-to-PID mapping is absent"]}
        process_ids = defaultdict(set)
        for global_pid, pid in connection.execute("SELECT globalPid,pid FROM PROCESSES"):
            if global_pid is not None and pid is not None:
                process_ids[int(global_pid)].add(int(pid))
    finally:
        connection.close()
    observed = defaultdict(list)
    for snapshot in snapshots:
        timestamp = snapshot.get("time_unix")
        if not isinstance(timestamp, (float, int)) or not math.isfinite(timestamp):
            continue
        owned = {int(pid) for pid in snapshot.get("owned_pids", [])}
        identities = snapshot.get("process_identities", {})
        for pid, identity in identities.items():
            marker, birth = identity.get("marker"), identity.get("start_ticks")
            requested = snapshot.get("requested_marker")
            matching_marker = marker and (not requested or marker in (requested, f"PAPER_V3_ATTEMPT={requested}"))
            if matching_marker and isinstance(birth, int) and birth >= 0:
                owned.add(int(pid))
        for process in snapshot.get("processes", []):
            pid, uuid = int(process["pid"]), _normal_uuid(process.get("gpu_uuid"))
            if pid in owned and uuid is not None:
                observed[pid].append({"uuid": uuid, "time_unix": timestamp,
                                      "identity": identities.get(str(pid), identities.get(pid)),
                                      "ownership_source": "process_identity" if str(pid) in identities or pid in identities
                                      else "collector_verified_owned_pids"})
    inventory_by_uuid = defaultdict(list)
    for row in inventory:
        uuid = _normal_uuid(row.get("uuid"))
        if uuid is not None:
            inventory_by_uuid[uuid].append(row)
    scopes = {activity_scope(event): event for event in events}
    mapping, physical_labels, records, errors = {}, {}, [], []
    for scope, event in sorted(scopes.items()):
        record = {"scope": scope, "global_pid": event.global_pid, "local_device_id": event.device_id,
                  "context_id": event.context_id, "status": "unresolved"}
        candidates = process_ids.get(event.global_pid, set())
        if len(candidates) != 1:
            reason = "Activity globalPid has no unique PROCESSES.pid"
        else:
            pid = next(iter(candidates))
            record["pid"] = pid
            witnesses = observed.get(pid, [])
            uuids = {witness["uuid"] for witness in witnesses}
            if len(uuids) != 1:
                reason = "No unique owned NVML GPU UUID for traced PID"
            else:
                uuid = next(iter(uuids))
                inventory_candidates = inventory_by_uuid.get(uuid, [])
                if len(inventory_candidates) != 1:
                    reason = "NVML UUID has no unique physical inventory GPU"
                else:
                    gpu = inventory_candidates[0]
                    physical = int(gpu["index"])
                    mapping[scope] = physical
                    physical_labels[physical] = f"Physical GPU {physical}"
                    record.update(status="verified", physical_index=physical, gpu_uuid=gpu["uuid"],
                                  inventory_gpu=gpu, evidence="PROCESSES.pid + owned NVML PID/UUID + unique inventory UUID",
                                  nvml_witness_count=len(witnesses), nvml_first_observed_unix=min(w["time_unix"] for w in witnesses),
                                  nvml_last_observed_unix=max(w["time_unix"] for w in witnesses),
                                  nvml_witnesses=witnesses)
                    reason = None
        if reason is not None:
            record["reason"] = reason
            errors.append(f"{scope}: {reason}")
        records.append(record)
    if expected_cards is not None:
        expected = {int(card) for card in expected_cards}
        if set(physical_labels) != expected:
            errors.append(f"Observed physical GPUs {sorted(physical_labels)} != requested GPUs {sorted(expected)}")
    return {"status": "valid" if not errors else "unresolved", "activity_device_map": mapping,
            "mapping": physical_labels, "devices": records, "errors": errors,
            "rule": "Activity (globalPid,localDeviceId,contextId) -> PROCESSES.pid -> owned NVML PID/UUID -> unique physical inventory UUID.",
            "snapshot_count": len(snapshots)}


def read_sqlite(path: Path) -> tuple[list[Activity], dict[int, dict], dict]:
    """Read device timestamps and preserve Nsight device IDs without guessing ordinals."""
    connection = sqlite3.connect(f"file:{quote(str(path.resolve()))}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        tables = _tables(connection)
        strings = ({int(row[0]): row[1] for row in connection.execute('SELECT id, value FROM StringIds')}
                   if "StringIds" in tables else {})
        events: list[Activity] = []
        schemas = {}
        for table, default_category in (("CUPTI_ACTIVITY_KIND_KERNEL", "non_nccl_kernel"),
                                        ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"),
                                        ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
            if table not in tables:
                continue
            columns = _columns(connection, table)
            schemas[table] = sorted(columns)
            required = {"start", "end", "deviceId"}
            if not required <= columns:
                raise ValueError(f"{table} is missing columns: {sorted(required - columns)}")
            selected = [key for key in ("start", "end", "deviceId", "globalPid", "contextId", "streamId",
                                        "correlationId", "demangledName", "shortName", "mangledName") if key in columns]
            query = 'SELECT ' + ','.join(f'"{key}"' for key in selected) + f' FROM "{table}" ORDER BY start'
            for row in connection.execute(query):
                value = dict(row)
                start, end = int(value["start"]), int(value["end"])
                if end <= start:
                    raise ValueError(f"{table} contains nonpositive GPU interval: {start}, {end}")
                name = next((str(strings.get(value[key], value[key])) for key in
                             ("demangledName", "shortName", "mangledName") if value.get(key) is not None),
                            "unresolved_kernel_name" if default_category == "non_nccl_kernel" else default_category)
                category = ("nccl_kernel" if default_category == "non_nccl_kernel" and
                            re.search(r"nccl", name, flags=re.IGNORECASE) else default_category)
                events.append(Activity(start, end, int(value["deviceId"]), category, name,
                                       value.get("globalPid"), value.get("contextId"), value.get("streamId"),
                                       value.get("correlationId")))
        if not any(event.category.endswith("kernel") for event in events):
            raise ValueError("No GPU kernel execution intervals found; CUDA API traces alone are insufficient")
        metadata = {}
        if "TARGET_INFO_GPU" in tables:
            for row in connection.execute('SELECT * FROM TARGET_INFO_GPU'):
                value = dict(row)
                if "id" in value:
                    device = int(value["id"])
                    decoded = {key: (item.hex() if isinstance(item, bytes) else item) for key, item in value.items()}
                    if device in metadata and metadata[device] != decoded:
                        raise ValueError(f"Ambiguous TARGET_INFO_GPU device id {device}")
                    metadata[device] = decoded
        return sorted(events, key=lambda event: event.start), metadata, {"tables": sorted(tables), "schemas": schemas}
    finally:
        connection.close()


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(output: Path, rows: list[dict], grouped: dict, labels: dict[int, str], begin: int, end: int,
          origin: int, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    devices = sorted(labels)
    figure, axes = plt.subplots(len(devices), 1, figsize=(12, 2.8 * len(devices)), squeeze=False, sharex=True)
    for device, axis in zip(devices, axes[:, 0]):
        selected = [row for row in rows if row["device_group_id"] == device]
        times = [row["window_start_seconds"] for row in selected] + [selected[-1]["window_end_seconds"]]
        for category in ("any_kernel", "non_nccl_kernel", "nccl_kernel", "memcpy"):
            values = [100 * row[f"{category}_active_fraction"] for row in selected]
            color = "#555555" if category == "any_kernel" else COLORS[category]
            axis.step(times, values + [values[-1]], where="post", color=color,
                      label=LABELS[category], linewidth=1.15, alpha=.85,
                      linestyle="--" if category == "any_kernel" else "-")
        axis.set_title(labels[device], loc="left", fontsize=10)
        axis.set_ylabel("Active time (%)")
        axis.set_ylim(-2, 102)
        axis.grid(alpha=.2)
    axes[0, 0].legend(ncol=4, fontsize=8, loc="upper right")
    axes[-1, 0].set_xlabel("Time from trace origin (seconds)")
    axes[-1, 0].set_xlim((begin - origin) / NANOSECONDS, (end - origin) / NANOSECONDS)
    figure.suptitle(title + " — GPU execution time fractions (not SM Activity)", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, .96))
    for extension in ("png", "pdf"):
        figure.savefig(output / f"kernel_activity_curve.{extension}", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(len(devices), 1, figsize=(12, 2.6 * len(devices)), squeeze=False, sharex=True)
    for device, axis in zip(devices, axes[:, 0]):
        for index, category in enumerate(CATEGORIES):
            clipped = ((max(start, begin), min(stop, end)) for start, stop in grouped[device][category]
                       if stop > begin and start < end)
            spans = [((start - origin) / NANOSECONDS, (stop - start) / NANOSECONDS)
                     for start, stop in merge_intervals(clipped)]
            axis.broken_barh(spans, (index - .3, .6), facecolors=COLORS[category])
        axis.set_yticks(range(len(CATEGORIES)), [LABELS[category] for category in CATEGORIES], fontsize=8)
        axis.set_title(labels[device], loc="left", fontsize=10)
        axis.set_ylim(-.6, 3.6)
        axis.grid(axis="x", alpha=.2)
    axes[-1, 0].set_xlabel("Time from trace origin (seconds)")
    axes[-1, 0].set_xlim((begin - origin) / NANOSECONDS, (end - origin) / NANOSECONDS)
    figure.suptitle(title + " — GPU activity intervals, union across streams", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, .96))
    for extension in ("png", "pdf"):
        figure.savefig(output / f"kernel_timeline.{extension}", dpi=180)
    plt.close(figure)


def export(sqlite_path: Path, output: Path, *, window_seconds: float = .05,
           start_seconds: float = 0, duration_seconds: float | None = None,
           origin_ns: int | None = None, device_map: dict[int, str] | None = None,
           activity_device_map: dict[str, int] | None = None, activity_mapping_evidence: dict | None = None,
           title: str = "CUDA trace", make_plots: bool = True) -> dict:
    """Export raw intervals, binned union fractions, summary and two figures."""
    if not math.isfinite(window_seconds) or window_seconds <= 0 or not math.isfinite(start_seconds):
        raise ValueError("Window must be positive and time offsets finite")
    if duration_seconds is not None and (not math.isfinite(duration_seconds) or duration_seconds <= 0):
        raise ValueError("Duration must be positive and finite")
    events, metadata, schema = read_sqlite(sqlite_path)
    if activity_mapping_evidence is not None and activity_mapping_evidence.get("status") != "valid":
        raise ValueError("Activity physical-device mapping evidence is unresolved")
    if activity_device_map is None:
        processes_by_device = defaultdict(set)
        for event in events:
            processes_by_device[event.device_id].add(event.global_pid)
        if any(len(processes) > 1 for processes in processes_by_device.values()):
            raise ValueError("Multiple processes share local device IDs; scoped activity_device_map is required")
    else:
        missing = {activity_scope(event) for event in events} - set(activity_device_map)
        if missing:
            raise ValueError(f"Missing scoped activity device mappings: {sorted(missing)}")
        if activity_mapping_evidence is not None and activity_mapping_evidence.get("activity_device_map") != activity_device_map:
            raise ValueError("Activity mapping differs from its evidence")
    origin = min(event.start for event in events) if origin_ns is None else int(origin_ns)
    begin = origin + round(start_seconds * NANOSECONDS)
    end = max(event.end for event in events) if duration_seconds is None else begin + round(duration_seconds * NANOSECONDS)
    window = round(window_seconds * NANOSECONDS)
    if end <= begin or window <= 0:
        raise ValueError("Requested time range/window has no positive nanosecond extent")
    output.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(lambda: defaultdict(list))
    original_devices = defaultdict(set)
    event_counts = defaultdict(int)
    raw_rows = []
    for event in events:
        group = event.device_id if activity_device_map is None else int(activity_device_map[activity_scope(event)])
        grouped[group][event.category].append((event.start, event.end))
        original_devices[group].add(event.device_id)
        event_counts[group] += 1
        value = asdict(event)
        value["device_group_id"] = group
        value["physical_gpu"] = group if activity_device_map is not None else None
        value["start_seconds"] = (event.start - origin) / NANOSECONDS
        value["end_seconds"] = (event.end - origin) / NANOSECONDS
        raw_rows.append(value)
    labels = {device: (device_map[device] if device_map and device in device_map else
                       f"Physical GPU {device}" if activity_device_map is not None else f"Nsight device {device}")
              for device in sorted(grouped)}
    rows, summaries = [], []
    for device, categories in sorted(grouped.items()):
        categories["any_kernel"] = categories["non_nccl_kernel"] + categories["nccl_kernel"]
        fractions = {category: interval_fractions(categories[category], begin, end, window)
                     for category in (*CATEGORIES, "any_kernel")}
        for index in range(len(fractions["any_kernel"])):
            lower, upper = begin + index * window, min(begin + (index + 1) * window, end)
            rows.append({"nsys_device_id": next(iter(original_devices[device])) if len(original_devices[device]) == 1 else None,
                         "device_group_id": device, "physical_gpu": device if activity_device_map is not None else None,
                         "device_label": labels[device],
                         "window_start_seconds": (lower - origin) / NANOSECONDS,
                         "window_end_seconds": (upper - origin) / NANOSECONDS,
                         "window_duration_seconds": (upper - lower) / NANOSECONDS,
                         **{f"{category}_active_fraction": values[index] for category, values in fractions.items()}})
        summaries.append({"nsys_device_id": next(iter(original_devices[device])) if len(original_devices[device]) == 1 else None,
                          "device_group_id": device, "physical_gpu": device if activity_device_map is not None else None,
                          "original_nsys_device_ids": sorted(original_devices[device]), "device_label": labels[device],
                          "event_count": event_counts[device],
                          **{f"{category}_active_fraction": interval_fractions(categories[category], begin, end, end - begin)[0]
                             for category in (*CATEGORIES, "any_kernel")}})
    raw_fields = [*asdict(events[0]), "device_group_id", "physical_gpu", "start_seconds", "end_seconds"]
    _write_csv(output / "gpu_activity_intervals_seconds.csv", raw_rows, raw_fields)
    _write_csv(output / "kernel_activity_windows_seconds.csv", rows, list(rows[0]))
    with sqlite_path.open("rb") as source_stream:
        source_sha256 = hashlib.file_digest(source_stream, "sha256").hexdigest()
    observed_markers = capture_bounds(sqlite_path)
    summary = {
        "schema": "nsys-gpu-interval-union-v1", "source_sqlite": str(sqlite_path.resolve()),
        "source_sha256": source_sha256, "raw_start_end_unit": "nanoseconds in original Nsight clock",
        "origin_ns": origin, "origin_rule": "first_observed_gpu_activity" if origin_ns is None else "caller_supplied",
        "begin_ns": begin, "end_ns": end, "duration_seconds": (end - begin) / NANOSECONDS,
        "window_seconds": window / NANOSECONDS,
        "end_rule": "last_observed_gpu_activity" if duration_seconds is None else "fixed_caller_duration",
        "definitions": {
            "active_fraction": "Duration of union of actual GPU execution intervals clipped to window, divided by window width.",
            "non_nccl_kernel": "Kernel name does not contain 'nccl' (case-insensitive). This is a compute proxy, not proof that all kernels are arithmetic.",
            "nccl_kernel": "Kernel name contains 'nccl' (case-insensitive). Other communication implementations are not identified.",
            "any_kernel": "Union of non-NCCL and NCCL GPU kernel execution intervals; excludes memcpy/memset.",
            "memcpy": "Union of CUPTI GPU memory-copy intervals, reported independently from kernels.",
            "overlap": "Each category is unioned independently; categories may overlap, so their fractions must not be summed.",
            "idle": "No captured process activity in this category; other users/processes outside the capture are not measured.",
            "not_sm_activity": "No SM hardware counters or warp occupancy are measured. CPU launch/NVTX duration is never used as GPU execution time.",
            "extent": "Default extent is observed GPU activity only. Supply origin/start/duration for an externally verified fixed capture interval, including idle tails.",
        },
        "devices": summaries, "target_info_gpu": metadata, "sqlite_schema": schema,
        "activity_device_map": activity_device_map, "activity_mapping_evidence": activity_mapping_evidence,
        "capture_marker_metadata": observed_markers,
        "export_extent_within_observed_markers": (
            begin >= observed_markers["common_start_ns"] and end <= observed_markers["common_end_ns"]
            if observed_markers["status"] == "valid" else None),
        "raw_event_count": len(events), "window_row_count": len(rows),
        "unresolved_kernel_names": sum(event.name == "unresolved_kernel_name" or event.name.isdecimal()
                                       for event in events if event.category.endswith("kernel")),
    }
    (output / "definitions_and_results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    if make_plots:
        _plot(output, rows, grouped, labels, begin, end, origin, title)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=.05)
    parser.add_argument("--start-seconds", type=float, default=0)
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--origin-ns", type=int)
    parser.add_argument("--device-map", type=Path, help="JSON mapping Nsight device IDs to verified physical GPU labels")
    parser.add_argument("--title", default="CUDA trace")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    mapping = {int(key): str(value) for key, value in json.loads(args.device_map.read_text()).items()} if args.device_map else None
    summary = export(args.sqlite, args.output, window_seconds=args.window_seconds, start_seconds=args.start_seconds,
                     duration_seconds=args.duration_seconds, origin_ns=args.origin_ns, device_map=mapping,
                     title=args.title, make_plots=not args.no_plots)
    print(json.dumps({"output": str(args.output.resolve()), "raw_events": summary["raw_event_count"],
                      "duration_seconds": summary["duration_seconds"], "devices": summary["devices"]}, indent=2))


if __name__ == "__main__":
    main()
