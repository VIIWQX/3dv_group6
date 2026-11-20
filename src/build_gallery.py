#!/usr/bin/env python3
"""Build an offline bilingual result gallery and a true-render-only montage.

This script reads existing NeRF outputs, metrics and matching artifacts. It
never reads GLBs and never creates replacement object images. Missing runs
remain visibly pending, so the gallery can also monitor an incomplete batch.
"""
from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


LABELS={"camera":"胶片相机", "barrel":"木桶", "rock":"岩石", "telephone":"电话机",
        "boots":"长靴", "radio":"收音机", "lion":"石狮", "cart":"木推车",
        "chest":"宝箱", "truck":"皮卡"}
STILL_FRAMES={"telephone":15,"boots":15}


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def font(size):
    candidates=[Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
    spec=importlib.util.find_spec("matplotlib")
    if spec is not None and spec.origin:
        candidates.append(Path(spec.origin).parent/"mpl-data/fonts/ttf/DejaVuSans.ttf")
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate),size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def collect(root,objects):
    records=[]
    for item in objects:
        uid=item["uid"]
        folder=root/"results"/uid
        metrics=read_json(folder/"metrics.json")
        matching=read_json(folder/"matching/summary.json")
        visualizations=sorted((folder/"matching/visualizations").glob("*.jpg"))
        name=item.get("object_id",item.get("name",uid[:8]))
        gif=folder/"orbit.gif"
        frames=sorted((folder/"orbit").glob("*.png"))
        record={"uid":uid,"name":name,"chinese":LABELS.get(name,""),
                "long_name":item.get("name",name),"folder":folder,"gif":gif,
                "ready":gif.exists() and bool(metrics),"metrics":metrics,"matching":matching,
                "visualizations":visualizations,"frames":frames}
        records.append(record)
    return records


def write_matching_summary(root,records):
    """Refresh the summary when an object was rerendered and rematched."""
    fields=["keypoints_total","ratio_matches_total","ransac_inliers_total","pairs_with_8_inliers",
            "triangulated_observations","sparse_points","median_estimated_rotation_error_degrees","seconds"]
    rows=[{"object_id":record["name"],"uid":record["uid"],
           **{key:record["matching"].get(key) for key in fields}} for record in records]
    results=root/"results"
    results.mkdir(exist_ok=True)
    (results/"matching_summary.json").write_text(json.dumps(rows,indent=2)+"\n")
    with (results/"matching_summary.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["object_id","uid"]+fields)
        writer.writeheader()
        writer.writerows(rows)


def montage(root,records,frame_count=60,cell=192):
    cols=5
    rows=math.ceil(len(records)/cols)
    header,label,footer=32,20,24
    width,height=cols*cell,header+rows*(cell+label)+footer
    label_font=font(13)
    title_font=font(16)
    small_font=font(12)
    sources={}
    durations=[]
    for record in records:
        if record["gif"].exists():
            im=Image.open(record["gif"])
            sources[record["uid"]]=im
            durations.append(im.info.get("duration",80))
    if not sources:
        return None
    animation=[]
    for fi in range(frame_count):
        canvas=Image.new("RGB",(width,height),"#fafaf7")
        draw=ImageDraw.Draw(canvas)
        draw.text((12,8),"Novel view reconstruction / 10 objects",font=title_font,fill="#13242c")
        counter=f"FRAME {fi+1:02d} / {frame_count:02d}"
        box=draw.textbbox((0,0),counter,font=small_font)
        draw.text((width-(box[2]-box[0])-12,11),counter,font=small_font,fill="#52636a")
        for i,record in enumerate(records):
            x=(i%cols)*cell
            y=header+(i//cols)*(cell+label)
            source=sources.get(record["uid"])
            if source is not None:
                source.seek(fi%getattr(source,"n_frames",1))
                picture=source.convert("RGB").resize((cell,cell),Image.Resampling.LANCZOS)
                canvas.paste(picture,(x,y))
            else:
                draw.rectangle((x,y,x+cell,y+cell),fill="#eef0ec")
                draw.text((x+55,y+85),"PENDING",font=label_font,fill="#7b8581")
            text=record["name"].replace("_"," ").title()
            bbox=draw.textbbox((0,0),text,font=label_font)
            draw.text((x+(cell-(bbox[2]-bbox[0]))/2,y+cell+2),text,font=label_font,fill="#263b44")
        draw.text((12,height-footer+6),"Trained voxel NeRF  |  Synthetic images + calibrated cameras  |  360-degree orbit",font=small_font,fill="#52636a")
        animation.append(canvas)
    results=root/"results"
    docs=root/"docs"
    results.mkdir(exist_ok=True)
    docs.mkdir(exist_ok=True)
    animation[0].save(results/"overview.gif",save_all=True,append_images=animation[1:],duration=int(sum(durations)/len(durations)),loop=0,optimize=False,disposal=2)
    # Choose only real frames 5 or 15; a three-quarter view reveals most shapes
    # better, while telephone/boots are clearer at the alternate side view.
    static=animation[min(5,len(animation)-1)].copy()
    draw=ImageDraw.Draw(static)
    draw.rectangle((0,0,width,header-1),fill="#fafaf7")
    draw.text((12,8),"Selected novel views / 10 objects",font=title_font,fill="#13242c")
    for i,record in enumerate(records):
        source=sources.get(record["uid"])
        x=(i%cols)*cell
        y=header+(i//cols)*(cell+label)
        index=STILL_FRAMES.get(record["name"],5)
        if source is not None:
            index=index%getattr(source,"n_frames",1)
            source.seek(index)
            static.paste(source.convert("RGB").resize((cell,cell),Image.Resampling.LANCZOS),(x,y))
        draw.rectangle((x,y+cell,x+cell,y+cell+label-1),fill="#fafaf7")
        text=f"{record['name'].title()}  /  {index+1:02d}"
        bbox=draw.textbbox((0,0),text,font=label_font)
        draw.text((x+(cell-(bbox[2]-bbox[0]))/2,y+cell+2),text,font=label_font,fill="#263b44")
    static.save(docs/"overview.jpg",quality=95)
    for source in sources.values():
        source.close()
    return {"width":width,"height":height,"frames":frame_count,"completed_objects":len(sources),"objects":len(records)}


def build_html(root,records):
    def relative(path):
        return path.relative_to(root).as_posix()
    def esc(value):
        return html.escape(str(value),quote=True)
    def metric(value,digits=2,suffix=""):
        return f"{value:.{digits}f}{suffix}" if isinstance(value,(int,float)) else "—"
    def resource(path,label):
        if path.exists():
            return f'<a href="{esc(relative(path))}">{esc(label)} <span aria-hidden="true">↗</span></a>'
        return ''
    completed=sum(r["ready"] for r in records)
    psnrs=[r["metrics"]["psnr"] for r in records if "psnr" in r["metrics"]]
    avg_psnr=sum(psnrs)/len(psnrs) if psnrs else None
    sparse=sum(r["matching"].get("sparse_points",0) for r in records)
    cards=[]
    for index,record in enumerate(records):
        uid=record["uid"]
        folder=record["folder"]
        m,s=record["metrics"],record["matching"]
        state='<span class="status"><i></i> Rendered</span>' if record["ready"] else '<span class="status pending"><i></i> Pending</span>'
        if record["gif"].exists():
            poster=record["frames"][min(STILL_FRAMES.get(record["name"],5),len(record["frames"])-1)] if record["frames"] else record["gif"]
            media=f'<img class="orbit" src="{esc(relative(record["gif"]))}" data-animation="{esc(relative(record["gif"]))}" data-poster="{esc(relative(poster))}" alt="{esc(record["long_name"])} — trained NeRF orbit" loading="lazy" width="256" height="256">'
        else:
            media='<div class="placeholder">等待重建结果<br><small>Reconstruction pending</small></div>'
        image_links="".join(resource(p,f"匹配图 {n+1}") for n,p in enumerate(record["visualizations"]))
        if not image_links:
            image_links='<span class="missing">匹配图 · pending</span>'
        metrics_html="".join(f'<div><dt>{label}</dt><dd>{value}</dd></div>' for label,value in [
            ("PSNR ↑",metric(m.get("psnr"),2," dB")),("FG PSNR ↑",metric(m.get("foreground_psnr"),2," dB")),
            ("SSIM ↑",metric(m.get("ssim"),3)),("Mask IoU ↑",metric(m.get("mask_iou"),3))])
        links="".join([resource(record["gif"],"GIF"),resource(folder/"matching/sparse.ply","Sparse PLY"),
                       resource(folder/"reconstruction.ply","NeRF PLY"),resource(folder/"checkpoint.pt","Checkpoint"),
                       resource(folder/"comparison.png","测试对照"),resource(folder/"metrics.json","Metrics JSON"),
                       resource(folder/"matching/complete_matching.zip","完整点匹配 ZIP")])
        cards.append(f'''<article class="object-card" id="{esc(record['name'])}">
          <div class="card-top"><span class="number">{index+1:02d}</span>{state}</div>
          <a class="render" href="{esc(relative(record['gif'])) if record['gif'].exists() else '#'+esc(record['name'])}">{media}</a>
          <div class="card-content"><div class="card-heading"><h3>{esc(record['chinese'] or record['name'].title())}</h3><span>{esc(record['name'].title())}</span></div>
          <p class="original-name">{esc(record['long_name'])}</p><code title="Object UID">{esc(uid)}</code>
          <dl class="metrics">{metrics_html}</dl>
          <p class="matching-stat">SIFT <strong>{s.get('keypoints_total',0):,}</strong> · RANSAC <strong>{s.get('ransac_inliers_total',0):,}</strong> · Sparse <strong>{s.get('sparse_points',0):,}</strong></p>
          <details><summary>查看匹配与数据 · Inspect artifacts</summary><nav class="artifact-links">{links}{image_links}{resource(folder/'matching/pair_stats.csv','Pair CSV')}{resource(root/'data'/uid/'contact_sheet.jpg','输入多视图')}</nav></details>
          </div></article>''')
    overview='<img src="results/overview.gif" alt="All trained NeRF reconstructions in a 60-frame orbit" width="960" height="480">' if (root/"results/overview.gif").exists() else '<div class="overview-pending">Overview pending · 等待首个训练结果</div>'
    docs_links="".join(resource(root/name,label) for name,label in [
        ("README.md","项目说明 README"),("docs/pipeline_report.html","实验报告 Report HTML"),("docs/pipeline_report.md","报告 Markdown"),("docs/pipeline_report.docx","报告 Word"),("docs/PROMPTS.md","全部提示词 Prompts"),
        ("docs/DATA_LICENSES.md","数据授权 Licenses"),("results/matching_summary.csv","匹配汇总 Matching CSV")])
    document='''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><title>10 Objects · NeRF Reconstruction Lab</title>
<style>
:root{--ink:#18313d;--muted:#66797e;--paper:#f5f6f0;--card:#fff;--line:#dce4df;--accent:#267457;--soft:#e7f0e8}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,"Segoe UI","PingFang SC","Microsoft YaHei",system-ui,sans-serif;line-height:1.6}a{color:inherit}button{font:inherit}.wrap{width:min(1240px,calc(100% - 64px));margin:auto}.topbar{display:flex;align-items:center;justify-content:space-between;padding:26px 0;border-bottom:1px solid var(--line)}.brand{font-weight:750;font-size:14px;letter-spacing:.1em}.brand:before{content:"";display:inline-block;width:12px;height:12px;background:var(--accent);border-radius:3px;margin-right:10px}.topbar>a{font-size:13px;text-decoration:none;color:var(--muted)}header{display:grid;grid-template-columns:1.5fr 1fr;gap:60px;padding:64px 0 40px}.eyebrow{text-transform:uppercase;font-size:11px;letter-spacing:.18em;color:var(--accent);font-weight:700}h1{font-size:clamp(35px,4.6vw,61px);font-weight:650;letter-spacing:-.045em;line-height:1.12;margin:18px 0 22px}h1 span{font-size:.62em;letter-spacing:-.025em}header p{max-width:610px;color:var(--muted);font-size:15px}.protocol{border-left:2px solid #b6d0be;padding:0 0 0 25px;align-self:end;margin-bottom:8px}.protocol strong{font-size:14px}.protocol p{font-size:13px;margin:10px 0}.protocol .en{font-size:12px}.stats{display:grid;grid-template-columns:repeat(3,1fr);border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:22px 0;margin:0 0 36px}.stats>div{padding:0 28px;border-right:1px solid var(--line)}.stats>div:first-child{padding-left:0}.stats>div:last-child{border:none}.stats strong{display:block;font-size:31px;font-weight:600;letter-spacing:-.04em}.stats span{font-size:12px;color:var(--muted)}.overview{border:1px solid var(--line);border-radius:18px;overflow:hidden;background:#fafaf7}.overview img{display:block;width:100%;height:auto}.caption{display:flex;justify-content:space-between;padding:14px 20px;font-size:12px;color:var(--muted);border-top:1px solid var(--line)}.caption a{font-weight:600;text-decoration:none}.pipeline{display:grid;grid-template-columns:repeat(5,1fr);gap:16px;margin:32px 0 70px}.pipeline div{font-size:12px;line-height:1.7}.pipeline b{display:block;font-size:14px;font-weight:600;margin-bottom:4px}.pipeline em{display:block;font-style:normal;color:var(--accent);font-size:11px;margin-bottom:8px;letter-spacing:.08em}.pipeline span{color:var(--muted)}.section-heading{display:flex;align-items:end;justify-content:space-between;margin-bottom:23px}.section-heading h2{margin:0;font-size:29px;letter-spacing:-.03em;font-weight:600}.section-heading p{margin:4px 0 0;color:var(--muted);font-size:13px}.motion{border:1px solid var(--line);border-radius:24px;background:white;padding:9px 16px;font-size:12px;cursor:pointer;color:var(--ink)}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:22px}.object-card{background:var(--card);border:1px solid var(--line);border-radius:17px;overflow:hidden}.card-top{padding:16px 20px 0;display:flex;justify-content:space-between;align-items:center}.number{font-size:11px;color:#8a999b;letter-spacing:.13em}.status{font-size:10px;font-weight:600;color:var(--accent);background:var(--soft);padding:3px 8px;border-radius:10px}.status i{display:inline-block;width:5px;height:5px;border-radius:50%;background:var(--accent);margin-right:5px}.status.pending{background:#f8f0db;color:#937126}.status.pending i{background:#937126}.render{display:block;text-decoration:none;aspect-ratio:1.28;background:white}.render img{width:100%;height:100%;object-fit:contain;display:block}.card-content{padding:0 20px 19px}.card-heading{display:flex;justify-content:space-between;align-items:baseline;gap:8px}.card-heading h3{font-size:20px;font-weight:600;margin:0}.card-heading>span{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}.original-name{font-size:11px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin:3px 0 7px}code{display:block;font-size:9px;letter-spacing:.025em;color:#849190;overflow-wrap:anywhere}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;border-top:1px solid var(--line);padding-top:13px;margin:15px 0 12px}.metrics dt{font-size:9px;color:var(--muted)}.metrics dd{margin:2px 0 0;font-size:14px;font-weight:600;white-space:nowrap}.matching-stat{font-size:9px;color:var(--muted);margin:0 0 13px}.matching-stat strong{font-weight:500;color:#354c54}details{border-top:1px solid var(--line);padding-top:11px}summary{cursor:pointer;font-size:10px;color:var(--accent)}.artifact-links{display:flex;gap:7px;flex-wrap:wrap;padding-top:12px}.artifact-links a,.artifact-links .missing{font-size:10px;padding:4px 8px;border:1px solid var(--line);border-radius:5px;text-decoration:none}.artifact-links a:hover{background:var(--soft)}.missing{color:#9a9a8c}.placeholder{height:100%;display:flex;align-items:center;justify-content:center;flex-direction:column;color:#85918e;font-size:15px;background:#f7f9f6}.placeholder small{font-size:11px}.method-note{margin:40px 0 0;background:#eaf0e7;border-radius:13px;padding:24px 28px;font-size:13px;color:#40584b}.method-note strong{font-weight:650}.method-note p{margin:8px 0 0;font-size:12px}.resources{display:flex;gap:19px;flex-wrap:wrap;margin:26px 0 45px}.resources a{font-size:12px;text-decoration:none;color:var(--accent)}footer{border-top:1px solid var(--line);padding:21px 0 32px;color:var(--muted);font-size:11px;display:flex;justify-content:space-between;gap:20px}.overview-pending{padding:100px;text-align:center;color:var(--muted)}@media(min-width:1150px){.object-card:last-child{grid-column:2}}@media(max-width:1000px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}header{gap:30px}.pipeline{gap:12px}.metrics dd{font-size:13px}}@media(max-width:640px){.wrap{width:calc(100% - 32px)}.topbar{padding:19px 0}header{grid-template-columns:1fr;padding-top:35px;gap:18px}h1{font-size:40px}.protocol{padding-left:17px}.stats strong{font-size:25px}.stats>div{padding:0 12px}.stats span{font-size:10px}.pipeline{grid-template-columns:1fr 1fr;margin-bottom:40px;gap:20px}.grid{grid-template-columns:1fr}.section-heading h2{font-size:25px}.section-heading p{font-size:11px}.caption{font-size:9px;padding:12px}.metrics dt{font-size:10px}.metrics dd{font-size:16px}.matching-stat{font-size:10px}.resources{gap:12px}footer{flex-direction:column;gap:5px}.method-note{padding:20px}}
</style></head><body><main class="wrap">
<div class="topbar"><div class="brand">3DV / RECONSTRUCTION LAB</div><a href="#objects">Assignment 02 · 2026</a></div>
<header><div><div class="eyebrow">From cleaned assets to novel views</div><h1>十个物体，重建三维。<br><span>10 objects. Learned in 3D.</span></h1><p>从第一次作业清洗后的 <strong>000-000</strong> 分片选取十个物体，完成多视图生成、特征匹配、稀疏三角化与 NeRF 训练，用环绕动画呈现新视角结果。</p></div><aside class="protocol"><strong>实验设置 / Experimental protocol</strong><p>合成多视图图像 + 已知渲染相机。<br>72 个训练视角 · 12 个独立测试视角 · 60 帧新视角环绕。</p><p class="en">Synthetic images with calibrated cameras. The GIFs show a trained voxel NeRF, rendered from held-out orbit poses.</p></aside></header>
<div class="stats"><div><strong>__COMPLETE__ / __COUNT__</strong><span>已完成重建 · completed objects</span></div><div><strong>__PSNR__</strong><span>测试集平均 PSNR · mean dB</span></div><div><strong>__SPARSE__</strong><span>SIFT 三角化稀疏点 · sparse points</span></div></div>
<div class="overview">__OVERVIEW__<div class="caption"><span>学习到的辐射场渲染 · Trained radiance fields</span><a href="docs/overview.jpg">静态总览 / Still image ↗</a></div></div>
<div class="pipeline"><div><em>01 / SOURCE</em><b>清洗数据 · Cleaned assets</b><span>第一次作业 000-000<br>可追溯 UID 与授权信息</span></div><div><em>02 / IMAGES</em><b>多视图 · Multi-view</b><span>固定光照与相机标定<br>训练 / 测试 / 环绕分离</span></div><div><em>03 / MATCHING</em><b>匹配 · SIFT + RANSAC</b><span>保存全部特征、匹配与掩码<br>由标定相机三角化稀疏点</span></div><div><em>04 / LEARNING</em><b>重建 · Voxel NeRF</b><span>RGB + alpha + 相机作为输入<br>可微体渲染优化辐射场</span></div><div><em>05 / EVALUATION</em><b>评估 · Novel views</b><span>PSNR / SSIM / Mask IoU<br>真实训练结果 360° GIF</span></div></div>
<div class="section-heading" id="objects"><div><div class="eyebrow">Result collection</div><h2>重建结果 / The objects</h2><p>打开卡片下方的数据链接，查看匹配图、三维点云和模型权重。</p></div><button class="motion" type="button" id="motion" aria-pressed="false">暂停卡片动画 · Pause</button></div>
<div class="grid">__CARDS__</div>
<div class="method-note"><strong>结果边界 / How to read these results</strong><p>这些图像由清洗后的 GLB 生成。重建阶段只读取 RGB、alpha 和已知相机，不读取源网格。独立估计的 Essential Matrix 姿态用于分析匹配质量；主要 NeRF 实验使用标定相机。Sparse PLY 是匹配三角化点云，NeRF PLY 是从学习密度提取的等值面，权重保存在 checkpoint 中。</p><p>The input is synthetic and camera calibration is known. These results evaluate a controlled image-to-radiance-field baseline; they do not claim successful pose recovery from uncalibrated real photographs. Test views are excluded from optimization.</p></div>
<nav class="resources">__DOCS__</nav><footer><span>Assignment 02 · Reproducible 3D reconstruction</span><span>本页可直接离线打开 · Open this file locally; no server or network required.</span></footer>
</main><script>const button=document.getElementById('motion');button.addEventListener('click',()=>{const paused=button.getAttribute('aria-pressed')!=='true';document.querySelectorAll('img.orbit').forEach(img=>{img.src=paused?img.dataset.poster:img.dataset.animation;});button.setAttribute('aria-pressed',String(paused));button.textContent=paused?'播放卡片动画 · Play':'暂停卡片动画 · Pause';});</script></body></html>
'''
    replacements={"__COMPLETE__":str(completed),"__COUNT__":str(len(records)),"__PSNR__":metric(avg_psnr),"__SPARSE__":f"{sparse:,}","__OVERVIEW__":overview,"__CARDS__":"\n".join(cards),"__DOCS__":docs_links}
    for key,value in replacements.items():
        document=document.replace(key,value)
    (root/"index.html").write_text(document)
    return completed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--frames",type=int,default=60)
    parser.add_argument("--cell-size",type=int,default=192)
    args=parser.parse_args()
    root=args.root.resolve()
    manifest=args.manifest or root/"manifests/selection.json"
    records=collect(root,json.loads(manifest.read_text())["objects"])
    write_matching_summary(root,records)
    info=montage(root,records,args.frames,args.cell_size)
    completed=build_html(root,records)
    print(json.dumps({"gallery":str(root/"index.html"),"completed_objects":completed,"total_objects":len(records),"montage":info},indent=2))


if __name__=="__main__":
    main()
