#!/usr/bin/env python3
"""Reproduce the ten-object image -> matching -> NeRF -> rendering experiment."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stages', nargs='+', default=['render', 'match', 'train', 'geometry', 'gallery', 'report', 'validate'],
                   choices=['render', 'match', 'train', 'geometry', 'sfm', 'sfm-train', 'ablation', 'gallery', 'report', 'validate'])
    p.add_argument('--uid', action='append', help='Repeat to select a subset; default all ten')
    p.add_argument('--source-root', type=Path, default=ROOT.parent, help='Assignment 1 directory')
    p.add_argument('--device', default='cuda')
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--resolution', type=int, default=96)
    p.add_argument('--rays', type=int, default=4096)
    p.add_argument('--samples', type=int, default=128)
    p.add_argument('--image-size', type=int, default=256)
    p.add_argument('--render-size', type=int, default=256)
    p.add_argument('--force', action='store_true', help='Re-run existing successful stages')
    args = p.parse_args()
    selection = json.loads((ROOT/'manifests/selection.json').read_text())
    objects = selection['objects']
    if args.uid:
        unknown = set(args.uid)-{o['uid'] for o in objects}
        if unknown:
            p.error(f'Unknown UIDs: {sorted(unknown)}')
        objects = [o for o in objects if o['uid'] in args.uid]
    for o in selection['objects']:
        o['source_path'] = str(args.source_root.resolve()/o['source_relative_path'])
    logs = ROOT/'logs'
    logs.mkdir(exist_ok=True)
    runtime_manifest = logs/'selection_runtime.json'
    runtime_manifest.write_text(json.dumps(selection, ensure_ascii=False, indent=2))

    def run(script, params, label):
        command = [sys.executable, str(ROOT/'src'/script), *map(str, params)]
        print(f'[{label}] {" ".join(command)}', flush=True)
        started = time.time()
        with (logs/(label+'.log')).open('w') as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        with (logs/'commands.jsonl').open('a') as stream:
            stream.write(json.dumps({'command': command, 'stage': label, 'returncode': result.returncode,
                                     'wall_seconds': time.time()-started, 'unix_started': started})+'\n')
        if result.returncode:
            print((logs/(label+'.log')).read_text()[-6000:], file=sys.stderr)
            raise SystemExit(result.returncode)
        print(f'[{label}] complete in {time.time()-started:.1f}s', flush=True)

    for stage in args.stages:
        if stage in ('report', 'validate'):
            run('build_report.py' if stage=='report' else 'validate_submission.py',
                [] if stage=='report' else ['--source-root', args.source_root], stage)
            continue
        if stage=='gallery':
            run('build_gallery.py', [], stage)
            continue
        if stage=='ablation':
            run('run_ablations.py', ['--device', args.device], stage)
            continue
        if stage=='sfm-train':
            run('train_sfm_branch.py', ['--device', args.device], stage)
            continue
        if stage=='match' and not args.force and (ROOT/'manifests/repository_inventory.json').exists():
            run('restore_matching.py', [], 'restore_matching')
        for obj in objects:
            uid = obj['uid']
            data, output = ROOT/'data'/uid, ROOT/'results'/uid
            if stage=='render':
                done = data/'transforms_test.json'
                params = ['--manifest', runtime_manifest, '--output', ROOT/'data', '--uid', uid]
                if args.force:
                    params.append('--overwrite')
                script='render_dataset.py'
            elif stage=='match':
                done=output/'matching/summary.json'
                params=['--data', data, '--output', output/'matching']
                script='match_features.py'
            elif stage=='train':
                done=output/'orbit.gif'
                params=['--data', data, '--output', output, '--device', args.device, '--steps', args.steps,
                        '--resolution', args.resolution, '--rays', args.rays, '--samples', args.samples,
                        '--image-size', args.image_size, '--render-size', args.render_size]
                script='nerf.py'
            elif stage=='geometry':
                done=output/'reconstruction.ply'
                params=['--checkpoint', output/'checkpoint.pt', '--output', output/'reconstruction.ply']
                script='export_geometry.py'
            elif stage=='sfm':
                done=output/'sfm/metrics.json'
                params=['--data', data, '--output', output/'sfm', '--threads', 4]
                script='run_sfm.py'
            complete=done.exists()
            if stage=='train':
                complete=complete and all((output/name).exists() for name in ['checkpoint.pt','config.json','metrics.json'])
            elif stage=='match':
                complete=complete and len(list((output/'matching/features').glob('*.npz')))==72 and len(list((output/'matching/pairs').glob('*.npz')))==288
            elif stage=='geometry' and (output/'checkpoint.pt').exists() and complete:
                complete=done.stat().st_mtime >= (output/'checkpoint.pt').stat().st_mtime
            if complete and not args.force:
                print(f'[{stage}:{obj["object_id"]}] existing artifact, skip', flush=True)
                continue
            run(script, params, stage+'_'+uid)


if __name__=='__main__':
    main()
