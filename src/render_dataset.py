#!/usr/bin/env python3
"""Render cleaned GLB assets to calibrated, textured multi-view RGBA images.

The renderer is a deterministic CPU z-buffer rasterizer; it needs neither a
display server nor OpenGL. GLB geometry is used only to make the image dataset.
Reconstruction programs consume the PNG files and camera calibration, never
the mesh. Run ``--help`` for the complete rendering protocol.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import trimesh
from numba import njit


@njit(cache=True)
def rasterize(xy, depth, faces, uvs, colors, normals, material_ids,
              texture_pixels, texture_offsets, texture_widths, texture_heights,
              factors, width, height):
    out = np.zeros((height, width, 4), dtype=np.uint8)
    zbuf = np.full((height, width), 1e10, dtype=np.float32)
    for fi in range(len(faces)):
        i0, i1, i2 = faces[fi]
        z0, z1, z2 = depth[i0], depth[i1], depth[i2]
        if min(z0, z1, z2) <= 0.05:
            continue
        x0, y0 = xy[i0]
        x1, y1 = xy[i1]
        x2, y2 = xy[i2]
        den = (y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
        if abs(den) < 1e-8:
            continue
        xmin = max(0, int(math.floor(min(x0, x1, x2))))
        xmax = min(width-1, int(math.ceil(max(x0, x1, x2))))
        ymin = max(0, int(math.floor(min(y0, y1, y2))))
        ymax = min(height-1, int(math.ceil(max(y0, y1, y2))))
        mid = material_ids[fi]
        tw, th = texture_widths[mid], texture_heights[mid]
        for y in range(ymin, ymax+1):
            py = y+0.5
            for x in range(xmin, xmax+1):
                px = x+0.5
                b0 = ((y1-y2)*(px-x2)+(x2-x1)*(py-y2))/den
                b1 = ((y2-y0)*(px-x2)+(x0-x2)*(py-y2))/den
                b2 = 1.0-b0-b1
                if b0 < -1e-5 or b1 < -1e-5 or b2 < -1e-5:
                    continue
                iz = b0/z0+b1/z1+b2/z2
                z = 1.0/iz
                if z >= zbuf[y, x]:
                    continue
                b0, b1, b2 = b0/z0/iz, b1/z1/iz, b2/z2/iz
                tx = ty = 0
                if tw > 0:
                    u = b0*uvs[i0, 0]+b1*uvs[i1, 0]+b2*uvs[i2, 0]
                    v = b0*uvs[i0, 1]+b1*uvs[i1, 1]+b2*uvs[i2, 1]
                    # Bilinear interpolation with repeat wrapping, GLTF UV origin.
                    uf = (u-math.floor(u))*tw-0.5
                    vf = (1.0-(v-math.floor(v)))*th-0.5
                    tx, ty = int(math.floor(uf)), int(math.floor(vf))
                    fx, fy = uf-tx, vf-ty
                nx = b0*normals[i0, 0]+b1*normals[i1, 0]+b2*normals[i2, 0]
                ny = b0*normals[i0, 1]+b1*normals[i1, 1]+b2*normals[i2, 1]
                nz = b0*normals[i0, 2]+b1*normals[i1, 2]+b2*normals[i2, 2]
                nn = max(1e-8, math.sqrt(nx*nx+ny*ny+nz*nz))
                light = 0.62 + 0.25*abs((nx*0.42-ny*0.34+nz*0.84)/nn) + 0.13*abs((-nx*0.69+ny*0.57+nz*0.45)/nn)
                rgba = np.empty(4, dtype=np.float64)
                for c in range(4):
                    vc = b0*colors[i0,c]+b1*colors[i1,c]+b2*colors[i2,c]
                    tex = 1.0
                    if tw > 0:
                        off = texture_offsets[mid]
                        p00 = texture_pixels[off+(ty%th)*tw+(tx%tw), c]
                        p10 = texture_pixels[off+(ty%th)*tw+((tx+1)%tw), c]
                        p01 = texture_pixels[off+((ty+1)%th)*tw+(tx%tw), c]
                        p11 = texture_pixels[off+((ty+1)%th)*tw+((tx+1)%tw), c]
                        tex = ((1-fy)*((1-fx)*p00+fx*p10)+fy*((1-fx)*p01+fx*p11))/255.0
                    rgba[c] = vc*tex*factors[mid,c]
                if rgba[3] < 0.35:
                    continue
                zbuf[y,x] = z
                for c in range(3):
                    out[y,x,c] = int(min(255.0, max(0.0, rgba[c]*light*255.0)))
                out[y,x,3] = 255
    return out


def camera_pose(azimuth, elevation, radius=3.0):
    az, el = np.deg2rad([azimuth, elevation])
    pos = radius*np.array([math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), math.sin(el)])
    backward = pos/np.linalg.norm(pos)
    right = np.cross([0., 0., 1.], backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    c2w = np.eye(4)
    c2w[:3,:3] = np.stack([right, up, backward], axis=1)
    c2w[:3,3] = pos
    return c2w


def read_mesh(path, texture_limit=2048):
    scene = trimesh.load(path, force="scene", process=False)
    vertices, faces, normals, uvs, colors, mids = [], [], [], [], [], []
    textures, factors = [], []
    material_count = 0
    source_vertices = 0
    source_faces = 0
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        mesh = scene.geometry[name]
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            continue
        verts = trimesh.transform_points(mesh.vertices, transform)
        normal_matrix = np.linalg.inv(transform[:3,:3]).T
        ns = np.asarray(mesh.vertex_normals) @ normal_matrix.T
        ns /= np.maximum(np.linalg.norm(ns, axis=1, keepdims=True), 1e-9)
        vc = np.ones((len(verts), 4), dtype=np.float32)
        uv = np.zeros((len(verts), 2), dtype=np.float32)
        visual = mesh.visual
        mats = []
        local_mids = np.zeros(len(mesh.faces), dtype=np.int32)
        if visual.kind == "texture":
            if visual.uv is not None and len(visual.uv) == len(verts):
                uv = np.asarray(visual.uv, dtype=np.float32)
            material = visual.material
            if hasattr(material, "materials"):
                mats = list(material.materials)
                if visual.face_materials is not None:
                    local_mids = np.asarray(visual.face_materials, dtype=np.int32)
            else:
                mats = [material]
            # COLOR_0 is stored in TextureVisuals.vertex_attributes by GLTF loader.
            attrs = getattr(visual, "vertex_attributes", {})
            if "color" in attrs:
                raw = np.asarray(attrs["color"], dtype=np.float32)
                if raw.max() > 1.0:
                    raw /= 255.0
                vc[:,:raw.shape[1]] = raw
        elif visual.kind == "vertex":
            vc = np.asarray(visual.vertex_colors, dtype=np.float32)/255.0
        elif visual.kind == "face":
            # Expand vertices so face colors remain sharp across mesh boundaries.
            idx = np.asarray(mesh.faces).reshape(-1)
            verts, ns, uv = verts[idx], ns[idx], uv[idx]
            vc = np.repeat(np.asarray(visual.face_colors, dtype=np.float32)/255., 3, axis=0)
        if not mats:
            mats = [None]
        for mat in mats:
            texture = None
            factor = np.ones(4, dtype=np.float32)
            if mat is not None:
                texture = getattr(mat, "baseColorTexture", None)
                if texture is None:
                    texture = getattr(mat, "image", None)
                raw_factor = getattr(mat, "baseColorFactor", None)
                if raw_factor is None and texture is None:
                    raw_factor = getattr(mat, "diffuse", None)
                if raw_factor is not None:
                    raw_factor = np.asarray(raw_factor, dtype=np.float32)
                    if raw_factor.max() > 1.0:
                        raw_factor /= 255.0
                    factor[:len(raw_factor)] = raw_factor
            if texture is not None:
                texture = texture.convert("RGBA")
                texture.thumbnail((texture_limit, texture_limit), Image.Resampling.LANCZOS)
                textures.append(np.asarray(texture, dtype=np.uint8))
            else:
                textures.append(None)
            factors.append(factor)
        offset = sum(len(v) for v in vertices)
        fs = np.arange(len(verts), dtype=np.int32).reshape(-1,3) if visual.kind == "face" else np.asarray(mesh.faces, dtype=np.int32)
        vertices.append(verts)
        normals.append(ns)
        uvs.append(uv)
        colors.append(vc)
        faces.append(fs+offset)
        mids.append(local_mids+material_count)
        material_count += len(mats)
        source_vertices += len(mesh.vertices)
        source_faces += len(mesh.faces)
    if not vertices:
        raise ValueError(f"No triangle geometry: {path}")
    verts = np.concatenate(vertices)
    ns = np.concatenate(normals)
    # GLTF is right-handed +Y-up. Convert to right-handed +Z-up.
    rotation = np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
    verts = verts @ rotation.T
    ns = ns @ rotation.T
    low, high = verts.min(axis=0), verts.max(axis=0)
    center = (low+high)/2
    scale = 1.6/np.max(high-low)
    verts = (verts-center)*scale
    norm_transform = np.eye(4)
    norm_transform[:3,:3] = rotation*scale
    norm_transform[:3,3] = -center*scale
    offsets, widths, heights, pixels = [], [], [], []
    total = 0
    for texture in textures:
        offsets.append(total)
        if texture is None:
            widths.append(0)
            heights.append(0)
        else:
            h,w = texture.shape[:2]
            widths.append(w)
            heights.append(h)
            pixels.append(texture.reshape(-1,4))
            total += w*h
    geometry = {
        "vertices": verts.astype(np.float32), "faces": np.concatenate(faces).astype(np.int32),
        "uvs": np.concatenate(uvs).astype(np.float32), "colors": np.concatenate(colors).astype(np.float32),
        "normals": ns.astype(np.float32), "material_ids": np.concatenate(mids).astype(np.int32),
        "texture_pixels": np.concatenate(pixels) if pixels else np.zeros((1,4),dtype=np.uint8),
        "texture_offsets": np.array(offsets,dtype=np.int64),
        "texture_widths": np.array(widths,dtype=np.int32), "texture_heights": np.array(heights,dtype=np.int32),
        "factors": np.array(factors,dtype=np.float32),
    }
    stats = {"vertices":source_vertices,"triangles":source_faces,"instances":len(vertices),
             "materials":len(textures),"textured_materials":sum(t is not None for t in textures),
             "normalization_matrix":norm_transform.tolist(), "original_rotated_bounds":[low.tolist(),high.tolist()],
             "normalization_max_extent":1.6,"alpha_cutoff":0.35,
             "renderer":"CPU perspective-correct textured triangle z-buffer; fixed two-sided world diffuse lighting"}
    return geometry, stats


def render(geometry, c2w, size=384, fov=40.0):
    focal = 0.5*size/math.tan(math.radians(fov)/2)
    p = (geometry["vertices"]-c2w[:3,3]) @ c2w[:3,:3]
    depth = -p[:,2]
    xy = np.stack([focal*p[:,0]/depth+size/2., -focal*p[:,1]/depth+size/2.],axis=1).astype(np.float32)
    keys = ("faces","uvs","colors","normals","material_ids","texture_pixels",
            "texture_offsets","texture_widths","texture_heights","factors")
    return rasterize(xy,depth.astype(np.float32),*(geometry[k] for k in keys),size,size)


def contact_sheet(images, labels, path, cols=6, tile=192):
    rows = math.ceil(len(images)/cols)
    sheet = Image.new("RGB",(cols*tile,rows*(tile+25)),(242,242,242))
    draw = ImageDraw.Draw(sheet)
    for i,(image,label) in enumerate(zip(images,labels)):
        if not isinstance(image,Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGBA")
        bg = Image.new("RGBA",image.size,(242,242,242,255))
        bg.alpha_composite(image)
        bg = bg.convert("RGB").resize((tile,tile),Image.Resampling.LANCZOS)
        x,y=(i%cols)*tile,(i//cols)*(tile+25)
        sheet.paste(bg,(x,y))
        draw.text((x+5,y+tile+4),str(label)[:29],fill=(25,25,25))
    path.parent.mkdir(parents=True,exist_ok=True)
    sheet.save(path)


def build_dataset(item, output, args):
    started = time.time()
    uid = item["uid"]
    radius = float(item.get("render_overrides", {}).get("radius", args.radius))
    folder = output/uid
    folder.mkdir(parents=True,exist_ok=True)
    geometry, stats = read_mesh(item["source_path"],args.texture_limit)
    if args.preview:
        views = [(a,e) for e in (15,40) for a in (0,90,180,270)]
        images = [render(geometry,camera_pose(a,e,radius),args.size,args.fov) for a,e in views]
        contact_sheet(images,[f"az={a} el={e}" for a,e in views],folder/"preview.jpg",cols=4)
        stats["seconds"] = time.time()-started
        (folder/"render_stats.json").write_text(json.dumps(stats,indent=2)+"\n")
        print(json.dumps({"uid":uid,"preview":str(folder/"preview.jpg"),"seconds":round(stats["seconds"],2)}),flush=True)
        return
    (folder/"source.json").write_text(json.dumps(item,indent=2)+"\n")
    (folder/"normalized_transform.json").write_text(json.dumps({"matrix":stats["normalization_matrix"],"coordinate_convention":"GLTF +Y-up to normalized +Z-up; c2w OpenGL -Z-forward +Y-up"},indent=2)+"\n")
    plans = {
        "train": [(i*360/args.views_per_ring,e) for e in args.elevations for i in range(args.views_per_ring)],
        "test": [(i*360/12+7.5,12 if i%2==0 else 37) for i in range(12)],
        "orbit": [(i*360/args.orbit_views+2.0,25) for i in range(args.orbit_views)],
    }
    summaries = []
    selected_images, labels = [], []
    focal = 0.5*args.size/math.tan(math.radians(args.fov)/2)
    for split, views in plans.items():
        (folder/split).mkdir(exist_ok=True)
        frames = []
        for i,(az,el) in enumerate(views):
            pose = camera_pose(az,el,radius)
            relative = f"{split}/{i:03d}.png"
            frames.append({"file_path":relative,"transform_matrix":pose.tolist(),"azimuth":float(az),"elevation":float(el)})
            if split != "orbit" or args.render_orbit:
                target=folder/relative
                if args.overwrite or not target.exists():
                    rgba = render(geometry,pose,args.size,args.fov)
                    Image.fromarray(rgba).save(target)
                elif split == "train" and i%max(1,args.views_per_ring//4)==0:
                    rgba=np.asarray(Image.open(target).convert("RGBA"))
                else:
                    rgba=None
                if split=="train" and i%max(1,args.views_per_ring//4)==0:
                    selected_images.append(rgba)
                    labels.append(f"train {i:03d}: {az:.0f}/{el:.0f}")
        transforms = {"camera_angle_x":math.radians(args.fov),"w":args.size,"h":args.size,
                      "fl_x":focal,"fl_y":focal,"cx":args.size/2,"cy":args.size/2,
                      "camera_model":"PINHOLE","frames":frames,
                      "camera_convention":"OpenGL camera-to-world: right +X, up +Y, look -Z",
                      "images_rendered":split!="orbit" or args.render_orbit}
        (folder/f"transforms_{split}.json").write_text(json.dumps(transforms,indent=2)+"\n")
        summaries.append({"split":split,"views":len(views),"images_rendered":transforms["images_rendered"]})
        print(f"{uid}: {split} {len(views)} views",flush=True)
    contact_sheet(selected_images,labels,folder/"contact_sheet.jpg",cols=4)
    stats.update({"seconds":time.time()-started,"image_size":args.size,"fov_degrees":args.fov,"camera_radius":radius,"splits":summaries})
    (folder/"render_stats.json").write_text(json.dumps(stats,indent=2)+"\n")
    print(json.dumps({"uid":uid,"seconds":round(stats["seconds"],2),"triangles":stats["triangles"]}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--uid",help="Optional single object UID")
    parser.add_argument("--size",type=int,default=384)
    parser.add_argument("--views-per-ring",type=int,default=24)
    parser.add_argument("--elevations",type=float,nargs="+",default=[0,25,50])
    parser.add_argument("--radius",type=float,default=3.0)
    parser.add_argument("--fov",type=float,default=40.0)
    parser.add_argument("--orbit-views",type=int,default=60)
    parser.add_argument("--texture-limit",type=int,default=2048)
    parser.add_argument("--render-orbit",action="store_true",help="Also render reference orbit PNGs; default orbit JSON supplies poses only")
    parser.add_argument("--preview",action="store_true",help="Render only an eight-view preview sheet")
    parser.add_argument("--overwrite",action="store_true")
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text())
    objects=manifest["objects"]
    if args.uid:
        objects=[item for item in objects if item["uid"]==args.uid]
        if not objects:
            parser.error(f"UID {args.uid} is not in manifest")
    args.output.mkdir(parents=True,exist_ok=True)
    for item in objects:
        build_dataset(item,args.output,args)


if __name__=="__main__":
    main()
