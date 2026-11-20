#!/usr/bin/env python3
"""Prepare a compact GitHub tree with every original point-matching result."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT.parent/'submission/group6_github')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    def copy(source,relative=None):
        target=out/(relative or source.relative_to(ROOT));target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    for name in ['README.md','.gitignore','requirements.txt','run_pipeline.py','index.html']:copy(ROOT/name)
    for name in ['src','configs','docs','manifests']:
        for source in sorted((ROOT/name).rglob('*')):
            if source.is_file() and '__pycache__' not in source.parts and source.suffix!='.pyc' and source.name not in ['repository_inventory.json','repository_validation.json']:
                copy(source)
    submission_path=out/'configs/submission.json'
    submission=json.loads(submission_path.read_text());submission.pop('publication_status',None)
    submission_path.write_text(json.dumps(submission,ensure_ascii=False,indent=2)+'\n')
    for source in (ROOT/'results').glob('*'):
        if source.is_file():copy(source)
    for name in ['ablation_summary.json','ablation_summary.csv','sfm_pose_summary.json','sfm_pose_summary.csv']:
        source=ROOT/'experiments'/name
        if source.exists():copy(source)
    for name in ['camera','cart','telephone','sfm_pose']:
        source_root=ROOT/'experiments'/name
        if source_root.exists():
            for source in source_root.rglob('*'):
                if source.is_file() and source.name in {'config.json','metrics.json','training.csv','comparison.png','run.log'}:copy(source)
    objects=json.loads((ROOT/'manifests/selection.json').read_text())['objects'];items=[]
    for obj in objects:
        uid=obj['uid'];base=ROOT/'results'/uid;target=out/'results'/uid
        for filename in ['orbit.gif','metrics.json','config.json','training.csv','comparison.png','reconstruction.ply','reconstruction.json']:
            copy(base/filename)
        for source in (base/'cameras').glob('*.json'):copy(source)
        sfm=base/'sfm'
        if sfm.exists():
            for filename in ['metrics.json','colmap_options.json','camera_poses.json','transforms_train_sfm.json','sparse_points.ply']:
                if (sfm/filename).exists():copy(sfm/filename)
        matching=base/'matching';folder=target/'matching';folder.mkdir(exist_ok=True)
        archive=folder/'complete_matching.zip';fingerprints={}
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as bundle:
            for source in sorted(matching.rglob('*')):
                if source.is_file() and '__pycache__' not in source.parts:
                    relative=source.relative_to(matching).as_posix();fingerprints[relative]=hashlib.sha256(source.read_bytes()).hexdigest();bundle.write(source,relative)
        for filename in ['summary.json','pair_stats.csv']:copy(matching/filename)
        for source in (matching/'visualizations').glob('*.jpg'):copy(source)
        items.append({'object_id':obj['object_id'],'uid':uid,'matching_archive':archive.relative_to(out).as_posix(),'matching_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'matching_files_sha256':fingerprints,'feature_files':72,'pair_files':288,'gif':f'results/{uid}/orbit.gif'})
    inventory={'schema_version':'group6-repository-inventory-v1','group_number':6,'repository':'https://github.com/VIIWQX/3dv_group6','point_matching_storage':'Losslessly compressed per-object ZIP; no matching arrays omitted. Use src/restore_matching.py to restore original paths.','omitted_optional_artifacts':['reproducible input images and GLBs','main training checkpoints retained in full local submission package','per-frame orbit PNGs represented by full GIFs','redundant COLMAP databases and raw sparse binary models retained locally','presentation assets not required by the two submission items'],'objects':items}
    (out/'manifests/repository_inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2))
    readme=out/'README.md';text=readme.read_text();text=text.replace('新视角渲染只需要检查点与相机 JSON，不需要原 GLB。评价还需要对应测试 PNG。完整运行和校验命令见报告。','新视角渲染只需要检查点与相机 JSON，不需要原 GLB。仓库默认不保存训练检查点；执行训练后会生成。评价还需要对应测试 PNG。完整运行和校验命令见报告。')
    text+='\n## GitHub结果存储与验收\n\n每个对象的 `matching/complete_matching.zip` 无损包含全部72份特征、288份图像对匹配、逐对CSV、稀疏点与可视化；`manifests/repository_inventory.json` 保存每个原始文件的SHA256。浏览器可直接查看同目录的匹配图和CSV；下载ZIP即可查看全部数组。训练检查点、逐帧PNG和COLMAP数据库可由完整代码重新生成，完整本地提交包另保留这些可选产物。\n\n```bash\npython -m pip install Pillow\npython src/verify_repository.py\npython src/restore_matching.py\n# 从头生成输入、匹配、检查点和GIF（需先准备源GLB与依赖）\npython run_pipeline.py --force\n```\n'
    readme.write_text(text)
    from build_gallery import collect,build_html
    build_html(out,collect(out,objects))
    print(json.dumps({'directory':str(out),'objects':len(items),'files':sum(p.is_file() for p in out.rglob('*')),'bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file())},ensure_ascii=False))

if __name__=='__main__':main()
