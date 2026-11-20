#!/usr/bin/env python3
"""SIFT matching, essential-matrix RANSAC and sparse triangulation for all views.

All keypoints/descriptors, ratio-test matches, RANSAC masks, triangulation masks,
pair statistics and sparse colored points are saved. Rendered camera calibration
is explicitly used for triangulation; pose errors report the independent
essential-matrix estimate against that calibration.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def cv_camera(c2w):
    conversion=np.diag([1.,-1.,-1.])
    rotation=conversion@c2w[:3,:3].T
    translation=-rotation@c2w[:3,3]
    return np.concatenate([rotation,translation[:,None]],axis=1)


def skew(vector):
    x,y,z=vector
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def write_ply(path,points,colors):
    with path.open("w") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point,color in zip(points,colors):
            handle.write("%.6f %.6f %.6f %d %d %d\n"%(*point,*color))


def frame_pairs(frames):
    pairs=[]
    for i,a in enumerate(frames):
        for j in range(i+1,len(frames)):
            b=frames[j]
            az=abs(float(a["azimuth"])-float(b["azimuth"]))%360
            az=min(az,360-az)
            el=abs(float(a["elevation"])-float(b["elevation"]))
            if (el<1 and az<=30.01) or (el<=25.01 and az<=15.01):
                pairs.append((i,j))
    # Generic fallback for datasets without orbit metadata.
    return pairs


def keypoint_array(keypoints):
    return np.array([[k.pt[0],k.pt[1],k.size,k.angle,k.response,k.octave,k.class_id]
                     for k in keypoints],dtype=np.float32).reshape(-1,7)


def feature_image(path,size):
    rgba=np.asarray(Image.open(path).convert("RGBA"))
    if rgba.shape[0]!=size or rgba.shape[1]!=size:
        rgba=cv2.resize(rgba,(size,size),interpolation=cv2.INTER_AREA)
    rgb=rgba[:,:,:3]
    mask=(rgba[:,:,3]>127).astype(np.uint8)*255
    # Erode the silhouette so artificial background edges cannot become matches.
    mask=cv2.erode(mask,np.ones((3,3),np.uint8),iterations=1)
    return rgb,mask


def geometric_filter(points0,points1,proj0,proj1,K):
    if len(points0)==0:
        return np.empty((0,3)),np.zeros(0,dtype=bool),np.empty(0),np.empty(0)
    hom=cv2.triangulatePoints(K@proj0,K@proj1,points0.T,points1.T).T
    points=hom[:,:3]/np.where(abs(hom[:,3:4])>1e-12,hom[:,3:4],1e-12)
    cam0=points@proj0[:,:3].T+proj0[:,3]
    cam1=points@proj1[:,:3].T+proj1[:,3]
    screen0=cam0@K.T
    screen1=cam1@K.T
    screen0=screen0[:,:2]/np.maximum(screen0[:,2:],1e-12)
    screen1=screen1[:,:2]/np.maximum(screen1[:,2:],1e-12)
    reproj=np.maximum(np.linalg.norm(screen0-points0,axis=1),np.linalg.norm(screen1-points1,axis=1))
    center0=-proj0[:,:3].T@proj0[:,3]
    center1=-proj1[:,:3].T@proj1[:,3]
    ray0=points-center0
    ray1=points-center1
    cos=np.sum(ray0*ray1,axis=1)/(np.linalg.norm(ray0,axis=1)*np.linalg.norm(ray1,axis=1)+1e-12)
    angle=np.rad2deg(np.arccos(np.clip(cos,-1,1)))
    valid=(cam0[:,2]>0)&(cam1[:,2]>0)&(reproj<3.0)&(angle>1.0)&np.all(np.isfinite(points),axis=1)&(np.max(abs(points),axis=1)<2.0)
    return points,valid,reproj,angle


def visualize_pair(image0,image1,kp0,kp1,matches,inliers,path,label):
    h,w=image0.shape[:2]
    canvas=np.concatenate([image0,image1],axis=1).copy()
    canvas[np.all(canvas==0,axis=2)]=245
    ids=np.flatnonzero(inliers)
    if len(ids)>80:
        ids=ids[np.linspace(0,len(ids)-1,80).astype(int)]
    rng=np.random.default_rng(7)
    for idx in ids:
        a,b=matches[idx]
        p=tuple(np.round(kp0[a,:2]).astype(int))
        q=tuple(np.round(kp1[b,:2]+[w,0]).astype(int))
        color=tuple(int(c) for c in rng.integers(40,230,3))
        cv2.line(canvas,p,q,color,1,cv2.LINE_AA)
        cv2.circle(canvas,p,2,color,-1,cv2.LINE_AA)
        cv2.circle(canvas,q,2,color,-1,cv2.LINE_AA)
    result=Image.new("RGB",(2*w,h+40),(245,245,245))
    result.paste(Image.fromarray(canvas),(0,40))
    ImageDraw.Draw(result).text((8,12),label,fill=(20,20,20))
    result.save(path)


def run(data,output,args):
    started=time.time()
    output.mkdir(parents=True,exist_ok=True)
    (output/"features").mkdir(exist_ok=True)
    (output/"pairs").mkdir(exist_ok=True)
    (output/"visualizations").mkdir(exist_ok=True)
    metadata=json.loads((data/"transforms_train.json").read_text())
    original_frames=metadata["frames"]
    frames=original_frames[::args.view_stride]
    source_indices=list(range(0,len(original_frames),args.view_stride))
    scale=args.size/metadata["w"]
    K=np.array([[metadata["fl_x"]*scale,0,metadata["cx"]*scale],
                [0,metadata["fl_y"]*scale,metadata["cy"]*scale],[0,0,1.]])
    cv2.setRNGSeed(20261105)
    cv2.setNumThreads(args.threads)
    sift=cv2.SIFT_create(nfeatures=args.max_features,contrastThreshold=0.025,edgeThreshold=10)
    images,keypoints,descriptors,cameras=[],[],[],[]
    feature_stats=[]
    for i,frame in enumerate(frames):
        rgb,mask=feature_image(data/frame["file_path"],args.size)
        kp,desc=sift.detectAndCompute(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),mask)
        ka=keypoint_array(kp)
        if desc is None:
            desc=np.empty((0,128),dtype=np.float32)
        images.append(rgb)
        keypoints.append(ka)
        descriptors.append(desc)
        cameras.append(cv_camera(np.array(frame["transform_matrix"])))
        np.savez_compressed(output/"features"/f"{source_indices[i]:03d}.npz",keypoints=ka,descriptors=desc,
                            source_image=frame["file_path"],K=K,camera_to_world=np.array(frame["transform_matrix"]))
        feature_stats.append({"frame":source_indices[i],"file_path":frame["file_path"],"keypoints":len(kp),"foreground_pixels":int((mask>0).sum())})
    pairs=frame_pairs(frames) if "azimuth" in frames[0] else [(i,j) for i in range(len(frames)) for j in range(i+1,min(i+4,len(frames)))]
    if args.max_pairs:
        pairs=pairs[:args.max_pairs]
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    pair_stats=[]
    sparse_points,sparse_colors,sparse_pairs=[],[],[]
    for number,(i,j) in enumerate(pairs):
        matches=np.empty((0,2),dtype=np.int32)
        distances=np.empty((0,2),dtype=np.float32)
        if len(descriptors[i])>=2 and len(descriptors[j])>=2:
            candidates=matcher.knnMatch(descriptors[i],descriptors[j],k=2)
            accepted=[(a,b) for pair in candidates if len(pair)==2 for a,b in [pair] if a.distance<args.ratio*b.distance]
            matches=np.array([[a.queryIdx,a.trainIdx] for a,b in accepted],dtype=np.int32).reshape(-1,2)
            distances=np.array([[a.distance,b.distance] for a,b in accepted],dtype=np.float32).reshape(-1,2)
        pts0=keypoints[i][matches[:,0],:2]
        pts1=keypoints[j][matches[:,1],:2]
        inliers=np.zeros(len(matches),dtype=bool)
        E=np.zeros((3,3))
        relative_rotation=cameras[j][:,:3]@cameras[i][:,:3].T
        relative_translation=cameras[j][:,3]-relative_rotation@cameras[i][:,3]
        E_known=skew(relative_translation)@relative_rotation
        rotation_error=translation_error=None
        if len(matches)>=8:
            estimated,mask=cv2.findEssentialMat(pts0,pts1,K,method=cv2.RANSAC,prob=0.999,threshold=args.ransac_threshold)
            if estimated is not None and mask is not None:
                E=estimated[:3,:3]
                inliers=mask.ravel().astype(bool)
                if inliers.sum()>=8:
                    try:
                        _,R,t,posemask=cv2.recoverPose(E,pts0,pts1,K,mask=mask.copy())
                        err=np.clip((np.trace(R@relative_rotation.T)-1)/2,-1,1)
                        rotation_error=float(np.rad2deg(np.arccos(err)))
                        td=np.dot(t.ravel(),relative_translation)/(np.linalg.norm(relative_translation)+1e-12)
                        translation_error=float(np.rad2deg(np.arccos(np.clip(td,-1,1))))
                    except cv2.error:
                        pass
        xyz,valid,reproj,parallax=geometric_filter(pts0,pts1,cameras[i],cameras[j],K)
        triangulated=inliers&valid
        if triangulated.any():
            xy=np.clip(np.round(pts0[triangulated]).astype(int),0,args.size-1)
            rgb=images[i][xy[:,1],xy[:,0]]
            sparse_points.append(xyz[triangulated])
            sparse_colors.append(rgb)
            sparse_pairs.append(np.tile([source_indices[i],source_indices[j]],(int(triangulated.sum()),1)))
        name=f"{source_indices[i]:03d}_{source_indices[j]:03d}"
        np.savez_compressed(output/"pairs"/f"{name}.npz",matches=matches,distances=distances,
                            ransac_inliers=inliers,triangulated=triangulated,points3d=xyz.astype(np.float32),
                            reprojection_error=reproj.astype(np.float32),parallax_degrees=parallax.astype(np.float32),
                            E_estimated=E,E_known=E_known,K=K,frame_indices=np.array([source_indices[i],source_indices[j]]))
        row={"frame_i":source_indices[i],"frame_j":source_indices[j],"features_i":len(keypoints[i]),"features_j":len(keypoints[j]),
             "ratio_matches":len(matches),"ransac_inliers":int(inliers.sum()),"inlier_ratio":float(inliers.mean()) if len(inliers) else 0.,
             "triangulated_points":int(triangulated.sum()),"mean_reprojection_error":float(reproj[triangulated].mean()) if triangulated.any() else None,
             "median_parallax_degrees":float(np.median(parallax[triangulated])) if triangulated.any() else None,
             "rotation_error_degrees":rotation_error,"translation_error_degrees":translation_error,"pair_file":f"pairs/{name}.npz"}
        pair_stats.append(row)
        if (number+1)%24==0 or number==len(pairs)-1:
            print(f"{data.name}: matched {number+1}/{len(pairs)} pairs",flush=True)
    with (output/"pair_stats.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(pair_stats[0]) if pair_stats else ["frame_i","frame_j"])
        writer.writeheader()
        writer.writerows(pair_stats)
    (output/"feature_stats.json").write_text(json.dumps(feature_stats,indent=2)+"\n")
    points=np.concatenate(sparse_points) if sparse_points else np.empty((0,3))
    colors=np.concatenate(sparse_colors) if sparse_colors else np.empty((0,3),dtype=np.uint8)
    provenance=np.concatenate(sparse_pairs) if sparse_pairs else np.empty((0,2),dtype=np.int32)
    np.savez_compressed(output/"sparse_raw.npz",points=points.astype(np.float32),colors=colors,frame_pairs=provenance)
    # Retain one observation per 0.005-unit voxel; raw observations remain available.
    if len(points):
        _,first=np.unique(np.round(points/args.voxel_size).astype(np.int32),axis=0,return_index=True)
        unique_points,unique_colors=points[first],colors[first]
    else:
        unique_points,unique_colors=points,colors
    np.savez_compressed(output/"sparse.npz",points=unique_points.astype(np.float32),colors=unique_colors)
    write_ply(output/"sparse.ply",unique_points,unique_colors)
    if pair_stats:
        ordered=sorted(range(len(pair_stats)),key=lambda n:pair_stats[n]["ransac_inliers"])
        chosen=list(dict.fromkeys([ordered[-1],ordered[-2] if len(ordered)>1 else ordered[-1],ordered[len(ordered)//2],ordered[max(0,len(ordered)//2-1)]]))
        for rank,index in enumerate(chosen):
            i,j=pairs[index]
            row=pair_stats[index]
            values=np.load(output/row["pair_file"])
            label=f"{row['frame_i']:03d} -> {row['frame_j']:03d} | ratio matches {row['ratio_matches']} | RANSAC inliers {row['ransac_inliers']} | 3D {row['triangulated_points']}"
            visualize_pair(images[i],images[j],keypoints[i],keypoints[j],values["matches"],values["ransac_inliers"],output/"visualizations"/f"{rank+1:02d}_{row['frame_i']:03d}_{row['frame_j']:03d}.jpg",label)
    rotations=[r["rotation_error_degrees"] for r in pair_stats if r["rotation_error_degrees"] is not None]
    summary={"object_uid":data.name,"method":"OpenCV SIFT + Lowe ratio + essential-matrix RANSAC + calibrated-camera triangulation",
             "pose_source":"Known rendering camera calibration; independent essential-matrix pose estimates are evaluated, not substituted",
             "coordinate_convention":"Normalized +Z-up world coordinates, same as dataset transforms",
             "image_size":args.size,"views":len(frames),"view_stride":args.view_stride,"pairs":len(pairs),
             "ratio_threshold":args.ratio,"ransac_threshold_px":args.ransac_threshold,"max_features":args.max_features,
             "keypoints_total":sum(r["keypoints"] for r in feature_stats),
             "keypoints_mean":float(np.mean([r["keypoints"] for r in feature_stats])),
             "ratio_matches_total":sum(r["ratio_matches"] for r in pair_stats),
             "ransac_inliers_total":sum(r["ransac_inliers"] for r in pair_stats),
             "pairs_with_8_inliers":sum(r["ransac_inliers"]>=8 for r in pair_stats),
             "triangulated_observations":len(points),"sparse_points":len(unique_points),
             "median_estimated_rotation_error_degrees":float(np.median(rotations)) if rotations else None,
             "triangulation_filters":{"max_reprojection_error_px":3.,"minimum_parallax_degrees":1.,"positive_depth_both_views":True,"max_absolute_world_coordinate":2.,"deduplication_voxel_size":args.voxel_size},
             "seconds":time.time()-started,"seed":20261105}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True,help="Directory with transforms_train.json and PNG views")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--size",type=int,default=384)
    parser.add_argument("--ratio",type=float,default=0.75)
    parser.add_argument("--ransac-threshold",type=float,default=1.5)
    parser.add_argument("--max-features",type=int,default=4096)
    parser.add_argument("--view-stride",type=int,default=1)
    parser.add_argument("--max-pairs",type=int,default=0)
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--voxel-size",type=float,default=0.005)
    args=parser.parse_args()
    run(args.data,args.output,args)


if __name__=="__main__":
    main()
