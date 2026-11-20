#!/usr/bin/env python3
"""Create a checked delivery ZIP, excluding environments and redundant models."""
import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def selected():
    for name in ['README.md','.gitignore','requirements.txt','run_pipeline.py','index.html']:
        yield ROOT/name
    for name in ['src','configs','docs','manifests','results','experiments']:
        for folder, dirs, files in os.walk(ROOT/name,followlinks=False):
            dirs[:]=sorted(d for d in dirs if d not in {'__pycache__','rejected','sfm_inputs','camera_smoke'})
            for filename in sorted(files):
                p=Path(folder)/filename
                if p.is_symlink():continue
                if name=='experiments' and filename=='checkpoint.pt':continue
                if p.suffix in {'.pyc','.part'}:continue
                yield p

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT.parent/'assignment2_submission.zip')
    args=p.parse_args()
    verification=json.loads((ROOT/'manifests/validation.json').read_text())
    if not verification.get('complete'):raise SystemExit('Required validation has not passed; refusing to package')
    paths=list(selected())
    lines=[]
    for file in paths:
        digest=hashlib.sha256(file.read_bytes()).hexdigest();lines.append(f'{digest}  {file.relative_to(ROOT).as_posix()}')
    checksum=ROOT/'ARTIFACTS.sha256';checksum.write_text('\n'.join(lines)+'\n')
    paths.append(checksum)
    with zipfile.ZipFile(args.output,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as bundle:
        for file in paths:bundle.write(file,Path('assignment2')/file.relative_to(ROOT))
    with zipfile.ZipFile(args.output) as bundle:
        bad=bundle.testzip()
        if bad:raise RuntimeError(f'ZIP CRC check failed: {bad}')
        names=bundle.namelist()
        for obj in json.loads((ROOT/'manifests/selection.json').read_text())['objects']:
            base=f'assignment2/results/{obj["uid"]}/'
            for suffix in ['orbit.gif','checkpoint.pt','matching/summary.json','reconstruction.ply']:
                if base+suffix not in names:raise RuntimeError(f'Missing deliverable: {base+suffix}')
    manifest={'zip':str(args.output),'bytes':args.output.stat().st_size,'sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),'files':len(paths),'crc_test':'passed','required_artifacts':'all ten GIFs, checkpoints, matching results, learned-density meshes','omissions':['Python environment and package cache','reproducible input images and source GLBs','rejected pilot output','redundant ablation checkpoints; code, configs, metrics and comparisons retained']}
    (args.output.with_suffix('.manifest.json')).write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest))

if __name__=='__main__':main()
