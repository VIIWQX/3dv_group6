#!/usr/bin/env python3
"""Check the course-required repository contents without CUDA or source GLBs."""
import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path
from PIL import Image,ImageSequence

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args();root=args.root.resolve()
    team=json.loads((root/'configs/team.json').read_text())
    assert str(team['group_number'])=='6'
    expected={'260010184':'李昱晔','260010242':'李宇怀','260010041':'仵庆熹','260010171':'曾子荣','260010193':'刘景寒'}
    assert {m['id']:m['name'] for m in team['members']}==expected
    assert sum(m['share_percent'] for m in team['members'])==100
    submission=json.loads((root/'configs/submission.json').read_text())
    assert submission['repository_url']=='https://github.com/VIIWQX/3dv_group6'
    report=(root/'docs/pipeline_report.md').read_text()
    assert submission['repository_url'] in report
    assert all(name in report and sid in report for sid,name in expected.items())
    for relative in ['README.md','requirements.txt','run_pipeline.py','docs/PROMPTS.md','docs/pipeline_report.docx','docs/pipeline_report.html','src/nerf.py','src/render_dataset.py','src/match_features.py','src/run_sfm.py']:
        assert (root/relative).is_file(),relative
    prompt_folder=root/'docs/prompt_records/presentation'
    assert len(list(prompt_folder.glob('slide_*.json')))==14
    assert (prompt_folder/'sample_prompt.txt').is_file() and (prompt_folder/'style.md').is_file()
    inventory=json.loads((root/'manifests/repository_inventory.json').read_text())
    assert len(inventory['objects'])==len({o['uid'] for o in inventory['objects']})==10
    details=[]
    for obj in inventory['objects']:
        folder=root/'results'/obj['uid']
        archive=root/obj['matching_archive']
        assert hashlib.sha256(archive.read_bytes()).hexdigest()==obj['matching_archive_sha256']
        with zipfile.ZipFile(archive) as bundle:
            assert bundle.testzip() is None
            names=bundle.namelist()
            features=[n for n in names if n.startswith('features/') and n.endswith('.npz')]
            pairs=[n for n in names if n.startswith('pairs/') and n.endswith('.npz')]
            assert len(features)==72 and len(pairs)==288
            for relative,digest in obj['matching_files_sha256'].items():
                assert hashlib.sha256(bundle.read(relative)).hexdigest()==digest
            assert len(list(csv.DictReader(io.StringIO(bundle.read('pair_stats.csv').decode()))))==288
        digests=set()
        with Image.open(folder/'orbit.gif') as gif:
            assert gif.format=='GIF' and gif.n_frames==60
            for frame in ImageSequence.Iterator(gif):
                assert frame.size==(256,256)
                digests.add(hashlib.sha256(frame.convert('RGB').tobytes()).hexdigest())
            assert len(digests)>1
        metrics=json.loads((folder/'metrics.json').read_text());assert metrics['test_views']==12
        details.append({'object':obj['object_id'],'uid':obj['uid'],'feature_files':72,'matching_pairs':288,'gif_frames':60,'matching_hashes':'passed'})
    result={'complete':True,'group_number':6,'members':5,'repository':submission['repository_url'],'objects':10,'feature_files':720,'matching_pairs':2880,'gif_files':10,'all_gifs_60_frames':True,'objects_detail':details}
    (root/'manifests/repository_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='objects_detail'},ensure_ascii=False))

if __name__=='__main__':main()
