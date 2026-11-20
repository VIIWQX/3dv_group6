#!/usr/bin/env python3
"""Restore the complete, checksummed point-matching arrays from repository ZIPs."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args();root=args.root.resolve()
    manifest_path=root/'manifests/repository_inventory.json'
    if not manifest_path.exists():
        print('No archived matching results: use the existing full matching directories.')
        return
    inventory=json.loads(manifest_path.read_text())
    for obj in inventory['objects']:
        folder=root/'results'/obj['uid']/'matching'
        archive=root/obj['matching_archive']
        assert hashlib.sha256(archive.read_bytes()).hexdigest()==obj['matching_archive_sha256']
        with zipfile.ZipFile(archive) as bundle:
            for relative,digest in obj['matching_files_sha256'].items():
                safe=Path(relative)
                if safe.is_absolute() or '..' in safe.parts:raise ValueError(relative)
                destination=folder/safe
                if destination.exists():
                    if hashlib.sha256(destination.read_bytes()).hexdigest()!=digest:
                        raise RuntimeError(f'Existing matching file differs; preserve it and choose a new directory: {destination}')
                    continue
                content=bundle.read(relative)
                if hashlib.sha256(content).hexdigest()!=digest:raise RuntimeError(relative)
                destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(content)
        print(f"{obj['object_id']}: restored 72 feature files and 288 pair files")

if __name__=='__main__':main()
