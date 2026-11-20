#!/usr/bin/env python3
"""Fetch only the ten selected, pinned Objaverse GLBs and verify SHA-256."""
import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path,default=ROOT.parent,
                   help='Assignment 1 root, or a new directory with the same relative layout')
    args=p.parse_args()
    selection=json.loads((ROOT/'manifests/selection.json').read_text())
    revision=selection['source_revision']
    for obj in selection['objects']:
        target=args.output_root/obj['source_relative_path']
        if target.exists():
            if digest(target)!=obj['sha256']:
                raise RuntimeError(f'Existing source checksum mismatch: {target}')
            print(f'{obj["object_id"]}: verified existing file',flush=True)
            continue
        url=f'https://huggingface.co/datasets/allenai/objaverse/resolve/{revision}/glbs/000-000/{obj["uid"]}.glb'
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix('.glb.part')
        with urllib.request.urlopen(url,timeout=180) as response, temporary.open('wb') as stream:
            shutil.copyfileobj(response,stream)
        if digest(temporary)!=obj['sha256']:
            raise RuntimeError(f'Download checksum mismatch: {temporary}')
        temporary.replace(target)
        print(f'{obj["object_id"]}: downloaded and verified',flush=True)

if __name__=='__main__':
    main()
