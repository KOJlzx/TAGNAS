#!/usr/bin/env python3
"""Resource configuration and buffered observations around unchanged native training."""
from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import MARKER, atomic_json, calibration_ready, exact_counts, identity, window_rates, thread_snapshot

SOURCE = Path(os.environ['CPU_SENSITIVITY_SOURCE_ROOT'])
sys.path[:0] = [str(SOURCE / 'src'), str(SOURCE / 'bin'), str(SOURCE)]
# Assert the inherited cpuset before importing Torch/CUDA/jemalloc.
_expected_cpus = [int(item) for item in os.environ['CPU_SENSITIVITY_CPUS'].split(',')]
_pre_torch = thread_snapshot(os.getpid(), _expected_cpus)
if not _pre_torch['threads'] or _pre_torch['outside_budget']:
    raise RuntimeError('Native interpreter was not restricted before imports')
import torch
import async_coexec_train as native
from clip_graph.async_coexec import train_runtime as runtime

_native_worker = native.worker_main
_native_build = native.build_training_service_for_role
_native_push = runtime.BaseTrainingService.push_embeddings
_native_record = runtime.CoordinatorTrainingService._record_telemetry
_native_snapshot = runtime.CoordinatorTrainingService.metrics_snapshot
_native_limit = runtime.CoordinatorTrainingService._limit_stop_reason
_native_batch = runtime.PubmedRuntimeStore.build_graph_batch
_records = []
_records_lock = threading.Lock()
_batch_context = threading.local()


def observed_batch(self, node_ids, *args, **kwargs):
    batch = _native_batch(self, node_ids, *args, **kwargs)
    edges = batch.graph_edge_index
    _batch_context.shape = dict(graph_node_rows=int(batch.graph_x.shape[0]),
                               graph_edge_entries=int(edges.shape[1]),
                               graph_endpoint_entries=int(edges.numel()))
    return batch


def observed_push(self, modality, records, *, telemetry):
    counts = exact_counts(telemetry, modality, len(records))
    cache = self.text_cache if modality == 'graph' else self.graph_cache
    observed = dict(counts, role=self.role_name, modality=modality, monotonic=time.monotonic(),
                    time_unix=time.time(), consumer_cache_size=len(cache),
                    phase=telemetry.get('phase'), loss=telemetry.get('loss'),
                    local_hit_nodes=telemetry.get('local_hit_nodes', 0),
                    miss_nodes=telemetry.get('miss_nodes', 0))
    if modality == 'graph':
        observed.update(getattr(_batch_context, 'shape', {}))
    payload = dict(telemetry, cpu_sensitivity=observed)
    before = time.monotonic()
    accepted = _native_push(self, modality, records, telemetry=payload)
    observed.update(push_seconds=time.monotonic()-before, accepted_by_coordinator=bool(accepted),
                    completed_monotonic=time.monotonic())
    with _records_lock:
        _records.append(observed)
    return accepted


def observed_record(self, modality, telemetry):
    _native_record(self, modality, telemetry)
    item = telemetry.get('cpu_sensitivity')
    if not item:
        raise ValueError('Native telemetry lacks exact CPU experiment row counts')
    event = dict(item, monotonic=time.monotonic(), time_unix=time.time())
    with self._stats_lock:
        if not hasattr(self, '_cpu_events'):
            self._cpu_events = []
        self._cpu_events.append(event)


def observed_snapshot(self, now, *, final=False):
    value = _native_snapshot(self, now, final=final)
    with self._stats_lock:
        events = list(getattr(self, '_cpu_events', []))
    value['cpu_sensitivity'] = dict(by_role={})
    for role in sorted({item['role'] for item in events}):
        rows = [item for item in events if item['role'] == role]
        value['cpu_sensitivity']['by_role'][role] = {
            key: sum(item[key] for item in rows)
            for key in ('forward_rows', 'success_rows', 'pushed_rows', 'updates', 'batches')}
        value['cpu_sensitivity']['by_role'][role]['consumer_cache_size'] = rows[-1]['consumer_cache_size']
    if final:
        output = Path(os.environ['CPU_SENSITIVITY_ATTEMPT_DIR'])
        with (output / 'coordinator_events.jsonl').open('x') as handle:
            for event in events:
                handle.write(json.dumps(event, sort_keys=True)+'\n')
        anchor = getattr(self, '_cpu_calibration_anchor', None)
        if anchor is not None:
            value['cpu_sensitivity']['steady_window'] = window_rates(events, anchor, anchor+120.0)
        atomic_json(output / 'coordinator_observation.json', dict(
            final_flush=True, role=self.role_name, identity=identity(os.getpid()),
            marker=os.environ.get(MARKER), event_count=len(events),
            accounting=value['cpu_sensitivity'], stop_reason=self._stop_reason))
    return value


def observed_limit(self, now):
    if (Path(os.environ['CPU_SENSITIVITY_ATTEMPT_DIR']) / 'stop.request').exists():
        return 'user_requested_stop'
    reason = _native_limit(self, now)
    if reason is not None or os.environ.get('CPU_SENSITIVITY_KIND') != 'lm_calibration':
        return reason
    anchor = getattr(self, '_cpu_calibration_anchor', None)
    if anchor is None:
        with self._stats_lock:
            events = list(getattr(self, '_cpu_events', []))
        roles = [f'text_worker_{i}' for i in range(self.config.num_text_workers)]
        if calibration_ready(events, now, roles, self.training_node_count):
            self._cpu_calibration_anchor = now
            atomic_json(Path(os.environ['CPU_SENSITIVITY_ATTEMPT_DIR']) / 'calibration_anchor.json', dict(
                monotonic=now, time_unix=time.time(), coverage_minimum=0.9995,
                previous_seconds=300, minimum_success_ratio=0.95, measurement_seconds=120,
                scope='First prospectively eligible boundary; no later-window replacement'))
    elif now >= anchor + 122.0:
        # Capture at least 122 seconds, analyze its common first 120 seconds.
        return 'calibration_window_completed'
    return None


def observed_build(*args, **kwargs):
    service = _native_build(*args, **kwargs)
    tower = getattr(service, 'graph_tower', getattr(service, 'text_tower', None))
    if tower is not None:
        dtypes = sorted({str(parameter.dtype) for parameter in tower.parameters()})
        if dtypes != ['torch.float32']:
            raise ValueError('Native model parameters are not uniformly FP32: '+str(dtypes))
    else:
        dtypes = []
    atomic_json(Path(os.environ['CPU_SENSITIVITY_ATTEMPT_DIR']) / f'role_ready_{service.role_name}.json', dict(
        identity=identity(os.getpid()), role=service.role_name, marker=os.environ.get(MARKER),
        affinity=sorted(os.sched_getaffinity(0)), intra_op_threads=torch.get_num_threads(),
        inter_op_threads=torch.get_num_interop_threads(), parameter_dtypes=dtypes,
        tensorpipe_worker_threads=32, time_unix=time.time(),
        thread_affinity_pre_torch=_pre_torch, thread_affinity_ready=thread_snapshot(os.getpid(), _expected_cpus)))
    return service


def worker_main(rank, config):
    cpus = [int(item) for item in os.environ['CPU_SENSITIVITY_CPUS'].split(',')]
    os.sched_setaffinity(0, cpus)
    graph_threads = int(os.environ['CPU_SENSITIVITY_THREADS'])
    torch.set_num_threads(graph_threads if rank == 1 else 1)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    completed = False
    try:
        _native_worker(rank, config)
        completed = True
    finally:
        output = Path(os.environ['CPU_SENSITIVITY_ATTEMPT_DIR'])
        role = config.role_for_rank(rank)
        with _records_lock:
            records = list(_records)
        with (output / f'work_{role}.jsonl').open('x') as handle:
            for row in records:
                handle.write(json.dumps(row, sort_keys=True)+'\n')
        usage = resource.getrusage(resource.RUSAGE_SELF)
        atomic_json(output / f'role_final_{role}.json', dict(
            identity=identity(os.getpid()), marker=os.environ.get(MARKER), role=role,
            completed_normally=completed, buffered_flush_complete=True, records=len(records),
            duration_seconds=time.monotonic()-started, user_cpu_seconds=usage.ru_utime,
            system_cpu_seconds=usage.ru_stime, max_rss_kib=usage.ru_maxrss,
            actual_affinity=sorted(os.sched_getaffinity(0)), intra_op_threads=torch.get_num_threads(),
            thread_affinity_final=thread_snapshot(os.getpid(), _expected_cpus),
            totals={key: sum(row[key] for row in records)
                    for key in ('forward_rows','success_rows','pushed_rows','updates','batches')}))


runtime.PubmedRuntimeStore.build_graph_batch = observed_batch
runtime.BaseTrainingService.push_embeddings = observed_push
runtime.CoordinatorTrainingService._record_telemetry = observed_record
runtime.CoordinatorTrainingService.metrics_snapshot = observed_snapshot
runtime.CoordinatorTrainingService._limit_stop_reason = observed_limit
native.build_training_service_for_role = observed_build
native.worker_main = worker_main

if __name__ == '__main__':
    native.main()
