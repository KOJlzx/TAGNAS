"""CPU-only contracts shared by the independent CPU sensitivity experiment."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

THREADS = (2, 4, 8, 16, 32, 64)
MARKER = "TAGNAS_CPU_SENSITIVITY_ATTEMPT"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(pid):
    try:
        directory = Path('/proc') / str(pid)
        fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'pid': int(pid), 'uid': directory.stat().st_uid, 'start_ticks': int(fields[19])}
    except (OSError, IndexError, ValueError):
        return None


def cpus_for_threads(threads):
    if type(threads) is not int or threads not in THREADS:
        raise ValueError('Unregistered logical CPU budget')
    # This machine numbers 32 distinct physical cores first, then SMT siblings.
    return list(range(threads))


def live_family(marker):
    expected = f'{MARKER}={marker}'.encode()
    found = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        ident = identity(int(path.name))
        if not ident or ident['uid'] != os.getuid():
            continue
        try:
            if expected in (path / 'environ').read_bytes().split(b'\0'):
                found.append(ident)
        except OSError:
            pass
    return found


def job_grid():
    # Fixed nonmonotone order, before any numerical result is observed.
    order = {'trex': (16, 4, 64, 8, 32), 'citation': (32, 8, 64, 4, 16)}
    jobs = []
    for index in range(5):
        for dataset in ('trex', 'citation'):
            threads = order[dataset][index]
            jobs.append(dict(key=f'search_{dataset}_t{threads}', kind='search', dataset=dataset,
                             threads=threads, gpu_count=3, cap_seconds=1800 if dataset == 'trex' else 10800))
    # Same per-GPU text batch as the search; do not reuse old global-batch-48 references.
    for count in (1, 2, 3, 4):
        jobs.append(dict(key=f'lm_calibration_citation_g{count}', kind='lm_calibration', dataset='citation',
                         threads=64, gpu_count=count, cap_seconds=10800))
    return jobs


def exact_counts(telemetry, modality, record_count):
    if modality not in ('graph', 'text'):
        raise ValueError('Unknown modality')
    rows = int(telemetry['seed_nodes']) if modality == 'graph' else int(record_count)
    updated = telemetry.get('update_applied')
    if rows < 0 or updated not in (0, 0.0, 1, 1.0):
        raise ValueError('Malformed batch accounting')
    return dict(forward_rows=rows, success_rows=rows * int(updated), pushed_rows=int(record_count),
                updates=int(updated), batches=1)


def calibration_ready(events, now, text_roles, train_nodes, window=300.0):
    """Earliest observed eligible boundary; no retrospective favorable window."""
    if not events or now - events[0]['monotonic'] < window:
        return False
    recent = [row for row in events if now - window <= row['monotonic'] <= now]
    for role in text_roles:
        history = [row for row in events if row['role'] == role]
        if not history or now - history[0]['monotonic'] < window:
            return False
        rows = [row for row in recent if row['role'] == role]
        if len(rows) < 5 or now - rows[-1]['monotonic'] > 30.0:
            return False
        if sum(row['updates'] for row in rows) / len(rows) < 0.95:
            return False
        if rows[-1]['consumer_cache_size'] / max(train_nodes, 1) < 0.9995:
            return False
    return True


def window_rates(events, start, end):
    if not end > start:
        raise ValueError('Window must have positive duration')
    rows = [row for row in events if start <= row['monotonic'] < end]
    roles = sorted({row['role'] for row in rows})
    by_role = {}
    for role in roles:
        selected = [row for row in rows if row['role'] == role]
        by_role[role] = {key: sum(row[key] for row in selected)
                         for key in ('forward_rows', 'success_rows', 'pushed_rows', 'updates', 'batches')}
        by_role[role]['success_rows_per_second'] = by_role[role]['success_rows'] / (end - start)
    return dict(start_monotonic=start, end_monotonic=end, duration_seconds=end-start, by_role=by_role,
                text_success_rows_per_second=sum(v['success_rows'] for k, v in by_role.items()
                                                 if k.startswith('text_worker_')) / (end-start))


def verify_sources(root, hashes):
    for relative, expected in hashes.items():
        if sha256(Path(root) / relative) != expected:
            raise ValueError('Frozen source mismatch: ' + relative)


def now_record():
    return dict(time_unix=time.time(), monotonic=time.monotonic())


import subprocess

def spawn_cpu(command: list[str], *, cpus: list[int], **kwargs):
    """The child inherits its budget from fork, including taskset's bootstrap."""
    inherited = os.sched_getaffinity(0)
    if not set(cpus) <= inherited:
        raise ValueError("Requested child budget is outside controller affinity")
    try:
        os.sched_setaffinity(0, cpus)
        return subprocess.Popen(command, **kwargs)
    finally:
        os.sched_setaffinity(0, inherited)


def thread_snapshot(pid: int, cpus: list[int]) -> dict:
    """Observe every existing TID, without changing another process's affinity."""
    threads = []
    for task in sorted((Path("/proc") / str(pid) / "task").glob("[0-9]*")):
        try:
            fields = (task / "stat").read_text().rsplit(")", 1)[1].split()
            mask = sorted(os.sched_getaffinity(int(task.name)))
            threads.append(dict(tid=int(task.name), start_ticks=int(fields[19]),
                                name=(task / "comm").read_text().strip(), affinity=mask,
                                user_ticks=int(fields[11]), system_ticks=int(fields[12]),
                                within_budget=set(mask) <= set(cpus)))
        except (FileNotFoundError, ProcessLookupError):
            continue  # A naturally exited TID is not an observer failure.
    return dict(identity=identity(pid), threads=threads,
                outside_budget=[item for item in threads if not item["within_budget"]])

