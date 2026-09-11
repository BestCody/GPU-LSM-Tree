"""Reports for the common suite; legacy reports remain unchanged."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

NAMES = {"gpulsmopt": "GPULSMOpt", "lsmu": "LSMu", "gpu_btree": "GPU B-tree",
         "flix": "FliX", "slabhash": "SlabHash", "warpcore": "WarpCore",
         "sorted_array": "Sorted array"}
COLORS = dict(zip(NAMES.values(), ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
                                  "#E69F00", "#56B4E9", "#666666"]))


def flix_rows(folder, case):
    raw = []
    for path in sorted(folder.glob("*.csv")):
        with path.open() as stream:
            raw.extend(csv.DictReader(stream, skipinitialspace=True))
    steps = sorted({int(r['step']) for r in raw if r.get('step', '').strip()})
    result = []
    for step in steps:
        group = [r for r in raw if r.get('step', '').strip() and int(r['step']) == step]
        row = group[0]
        metrics = {r['DESCRIPTION']: float(r['VALUE']) for r in group}
        base = dict(system=NAMES[case['backend']], protocol='flix_xy_complete_unsorted_v1',
                    state=step, resident_elements=int(row['after_update_size']),
                    index_bytes=metrics['after_update_bytes'], input_checksum=row['request_checksum'])
        for count_key, prefix, operation, scenario in [
            ('hit_query_count', 'probe', 'lookup', 'all_existing'),
            ('miss_query_count', 'probe_miss', 'lookup', 'none_existing'),
            ('deleted_query_count', 'deleted_keys_probe', 'lookup_deleted', 'all_deleted')]:
            count = int(row[count_key])
            if count:
                times = {'time_ms': metrics[prefix + '_time_ms'],
                         'wall_ms': metrics[prefix + '_wall_time_ms']}
                times.update({part+'_ms': metrics[prefix+'_'+part+'_time_ms']
                              for part in ('prepare', 'search', 'restore')})
                result.append(dict(base, operation=operation, scenario=scenario, items=count, **times))
        count = int(row['current_batch_size'])
        if count and step > 0:
            operation = 'insert' if row['do_insert'] in ('1', 'true') else 'delete'
            result.append(dict(base, operation=operation, scenario='xy_updates', items=count,
                               time_ms=metrics['insert_or_delete_time_ms'], wall_ms=np.nan,
                               prepare_ms=np.nan, search_ms=np.nan, restore_ms=np.nan))
        if metrics['rebuild_time_ms'] > 0:
            result.append(dict(base, operation='rebuild', scenario='required_maintenance', items=0,
                               time_ms=metrics['rebuild_time_ms'], wall_ms=np.nan,
                               prepare_ms=np.nan, search_ms=np.nan, restore_ms=np.nan))
    return result


def load_runs(root, manifest):
    frames = []
    for case in manifest['cases']:
        for rep in range(manifest['settings']['repetitions']):
            folder = root / case['id'] / f'rep_{rep:02}'
            completion = folder / 'completion.json'
            if not completion.exists():
                raise RuntimeError(f'Missing completed repetition: {folder}')
            record = json.loads(completion.read_text())
            for name, digest in record['artifacts'].items():
                if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
                    raise RuntimeError(f'Changed result artifact: {folder / name}')
            if case['family'] == 'paper':
                with (folder / 'measurements.csv').open() as stream:
                    rows = list(csv.DictReader(stream))
            else:
                rows = flix_rows(folder, case)
            frame = pd.DataFrame(rows)
            for key in ('family', 'backend', 'kind', 'batch_log'):
                frame[key] = case[key]
            frame['repetition'] = rep
            frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    for name in ('state', 'resident_elements', 'items', 'time_ms', 'wall_ms', 'prepare_ms',
                 'search_ms', 'restore_ms', 'index_bytes'):
        frame[name] = pd.to_numeric(frame[name], errors='raise')
    frame['rate_mops'] = frame['items'] / frame['time_ms'] / 1000
    return frame


def summarize(root, no_plots=False):
    root = Path(root)
    manifest = json.loads((root / 'run_manifest.json').read_text())
    if manifest['protocol'] != 'flix_paper_suite_v1':
        raise RuntimeError('Unknown suite protocol')
    validation = json.loads((root / 'validation.json').read_text())
    if validation['state'] != 'passed':
        raise RuntimeError('Suite validation has not passed')
    raw = load_runs(root, manifest)
    out = root / 'summary'; out.mkdir(exist_ok=True)
    raw.to_csv(out / 'measurements.csv', index=False)
    keys = ['family', 'kind', 'batch_log', 'system', 'operation', 'scenario',
            'state', 'resident_elements', 'items']
    states = raw.groupby(keys, dropna=False, sort=True).agg(
        repetitions=('time_ms', 'count'), mean_ms=('time_ms', 'mean'),
        median_ms=('time_ms', 'median'), stddev_ms=('time_ms', 'std'),
        minimum_ms=('time_ms', 'min'), maximum_ms=('time_ms', 'max'),
        mean_wall_ms=('wall_ms', 'mean'), median_rate_mops=('rate_mops', 'median'),
        maximum_index_bytes=('index_bytes', 'max')).reset_index()
    if not states['repetitions'].eq(manifest['settings']['repetitions']).all():
        raise RuntimeError('Unequal or duplicated repetition coverage')
    states.to_csv(out / 'states.csv', index=False)
    trace_keys = ['family', 'kind', 'batch_log', 'system', 'operation', 'scenario']
    totals = raw.groupby(trace_keys + ['repetition'], sort=True).agg(
        total_ms=('time_ms', 'sum'), items=('items', 'sum'), states=('state', 'count'),
        first_resident=('resident_elements', 'min'), last_resident=('resident_elements', 'max')).reset_index()
    totals['rate_mops'] = totals['items'] / totals['total_ms'] / 1000
    totals.to_csv(out / 'trace_totals.csv', index=False)
    totals.groupby(trace_keys, sort=True).agg(
        repetitions=('total_ms', 'count'), mean_ms=('total_ms', 'mean'),
        median_ms=('total_ms', 'median'), stddev_ms=('total_ms', 'std'),
        median_rate_mops=('rate_mops', 'median'), states=('states', 'first'),
        items=('items', 'first'), first_resident=('first_resident', 'first'),
        last_resident=('last_resident', 'first')).to_csv(out / 'traces.csv')
    if not no_plots:
        plot_states(states, out)
    (out / 'README.md').write_text(
        '# Common FliX paper suite\n\n'
        'Only completed timed repetitions appear here. Warmups and sanitizer runs are excluded. '
        'states.csv reports variability across repetitions of the same state; traces.csv sums '
        'time and work within each repetition before aggregation. Compare trace throughput only '
        'when the state coverage and item counts match.\n\n'
        'Paper-family inputs use the historical bijective key permutation shifted by two to '
        'exclude reserved keys. All backends build the first batch; subsequent insertions grow '
        'the same prefixes. Deletes remove those batches in reverse order. Values retain their '
        'original insertion ordinals. Sorted array sorts and merges insertion batches, '
        'and filters records against sorted deletion keys. Common ranges are sums, not '
        'materialized range enumeration; GPULSMOpt, LSMu, GPU B-tree, FliX, and the '
        'sorted array participate. Preparation and lazy range workspace allocation '
        'are inside the timed calls. Dynamic ranges also run after deletions. '
        'Live-key overwrites and explicit LSM cleanup are outside this common matrix.\n\n'
        'Lookup starts with unsorted device inputs and ends with device answers in original '
        'order, including required preparation and restoration. Update timing includes required '
        'input sorting and the update API. CUDA and synchronized wall times are recorded by the '
        'paper driver; the original FliX family has wall times for lookup only. FliX rebuild '
        'maintenance is reported separately. All samples use fresh processes and indexes; '
        'workspace reuse occurs only within one trace.\n\n'
        'Paper construction and index destruction are timed separately. The original FliX X/Y '
        'family does not time initial construction. These are resident-operation results, not '
        'transfer-inclusive lifecycle or peak-memory measurements. index_bytes is an adapter '
        'snapshot, excludes harness buffers, and is not a measured allocation peak. Input '
        'generation, checksum collection, and correctness checking are outside operation timers. '
        'Validation between operations may affect cache state. No forced final publication is '
        'required for visible results; explicit cleanup is available in the legacy family.\n\n'
        'The common_initialized_v1 protocol changes initialization, values, and lookup timing '
        'relative to historical results. Do not merge the two datasets. The manifest records '
        'source, GPU, build, workload, and repetition settings.\n')
    print('Wrote suite reports: ' + str(out))


def plot_states(frame, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    graphs = out / 'graphs'; graphs.mkdir(exist_ok=True)
    keys = ['family', 'kind', 'batch_log', 'operation', 'scenario']
    for identity, group in frame.groupby(keys, sort=True):
        family, kind, batch, operation, scenario = identity
        if operation == 'destroy': continue
        fig, ax = plt.subplots(figsize=(9, 5))
        for system, points in group.groupby('system', sort=False):
            points = points.sort_values('state')
            ax.errorbar(points['state'], points['median_ms'],
                        yerr=[points['median_ms']-points['minimum_ms'],
                              points['maximum_ms']-points['median_ms']],
                        marker='o', markersize=3, capsize=2, color=COLORS[system], label=system)
        layout = f', public batch log {batch}' if family == 'paper' else ''
        ax.set(xlabel='Workload state', ylabel='Complete operation time (ms)',
               title=f'{family}: {kind}{layout}, {operation}, {scenario}')
        ax.set_yscale('log'); ax.grid(alpha=.2); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(graphs / ('_'.join(map(str, identity)) + '.png'), dpi=160)
        plt.close(fig)
