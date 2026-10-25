"""SfM-guided SAM prompts before saving training masks; no trained model is used."""
from pathlib import Path
import importlib.util
import hashlib
import os
import numpy as np
import cv2
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scripts.mask_generation.label_colmap_points_viewer import read_images_bin, read_points3D_bin

VERSION = "sfm_sam_instance_v1"
IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def dense_components(xyz, radius, min_points=5):
    tree = cKDTree(xyz)
    core = tree.query_ball_point(xyz, radius, return_length=True) >= min_points
    labels = np.full(len(xyz), -1, dtype=int)
    if not core.any():
        raise RuntimeError("实例约束没有足够的三维稠密支持")
    ids = np.flatnonzero(core)
    ct = cKDTree(xyz[core])
    pairs = ct.query_pairs(radius, output_type='ndarray')
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(ids), len(ids)))
    _, cl = connected_components(graph, directed=False)
    labels[ids] = cl
    distance, nearest = ct.query(xyz[~core])
    labels[~core] = np.where(distance <= radius, cl[nearest], -1)
    return labels


def sample_prompts(xy, count=24):
    """Deterministic spatial coverage instead of clustering prompts on one leaf."""
    xy = np.asarray(xy, np.float32)
    if len(xy) <= count:
        return xy
    chosen = [int(np.argmin(np.linalg.norm(xy - np.median(xy, axis=0), axis=1)))]
    distances = np.full(len(xy), np.inf)
    for _ in range(count - 1):
        distances = np.minimum(distances, np.sum((xy - xy[chosen[-1]]) ** 2, axis=1))
        chosen.append(int(np.argmax(distances)))
    return xy[chosen]


def build_support(scene, images, labels, init_idx):
    model = Path(scene) / 'sparse/0'
    if not (model / 'images.bin').is_file():
        model = Path(scene) / 'sparse'
    records = {r.name: r for r in read_images_bin(model / 'images.bin').values()}
    points = read_points3D_bin(model / 'points3D.bin')
    ids = np.array(sorted(points), dtype=np.int64)
    xyz = np.array([points[int(i)].xyz for i in ids])
    votes = np.zeros((len(ids), 4), dtype=np.int32)
    observations = {}
    for index in sorted(labels):
        image = records[images[index].name]
        xy = np.rint(image.xys).astype(int)
        h, w = labels[index].shape
        valid = (image.point3D_ids >= 0) & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        xy = xy[valid]; rows = np.searchsorted(ids, image.point3D_ids[valid])
        if not np.array_equal(ids[rows], image.point3D_ids[valid]):
            raise RuntimeError('COLMAP track references a missing point')
        np.add.at(votes, (rows, labels[index][xy[:, 1], xy[:, 0]]), 1)
        observations[index] = (rows, xy)
    total = votes.sum(1)
    cube = (votes[:, 2] >= 3) & (votes[:, 2] >= .6 * total)
    plant = (votes[:, 1] >= 3) & (votes[:, 1] >= .55 * total)
    if cube.sum() < 10 or plant.sum() < 30:
        raise RuntimeError('单株约束缺少跨视角植株/魔方支持，禁止保存未核验掩码 '
                           f'(cube={int(cube.sum())}/10, plant={int(plant.sum())}/30)')
    scale = float(np.linalg.norm(np.percentile(xyz[cube], 90, axis=0) - np.percentile(xyz[cube], 10, axis=0)) / np.sqrt(3))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError('魔方尺度不合法')
    radius = .8 * scale
    # Bound the one tracked cube in each view, including partly occluded views.
    # This removes stray ID-2 pixels without inventing a hidden cube surface.
    spec=importlib.util.spec_from_file_location('guard_colmap',Path(__file__).resolve().parents[2]/'scene/colmap_loader.py')
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    poses={r.name:r for r in loader.read_extrinsics_binary(str(model/'images.bin')).values()}
    cameras=loader.read_intrinsics_binary(str(model/'cameras.bin'))
    cube_xyz=xyz[cube]
    cube_xyz=cube_xyz[np.linalg.norm(cube_xyz-np.median(cube_xyz,axis=0),axis=1)<3*scale]
    cube_regions={}
    for index in observations:
        pose=poses[images[index].name];camera=cameras[pose.camera_id]
        if camera.model=='PINHOLE':fx,fy,cx,cy=camera.params
        elif camera.model=='SIMPLE_PINHOLE':fx,cx,cy=camera.params;fy=fx
        else:raise RuntimeError('实例约束要求去畸变的PINHOLE图像')
        p=cube_xyz@pose.qvec2rotmat().T+pose.tvec;p=p[p[:,2]>0]
        if not len(p):raise RuntimeError('魔方位于相机后方，模型/图像不匹配')
        uv=np.column_stack((fx*p[:,0]/p[:,2]+cx,fy*p[:,1]/p[:,2]+cy))
        lo=uv.min(0);hi=uv.max(0);margin=np.maximum((hi-lo)*.15,6)
        h,w=labels[index].shape
        box=np.r_[np.maximum(lo-margin,0),np.minimum(hi+margin,[w-1,h-1])]
        cube_regions[index]=box
    pi = np.flatnonzero(plant)
    components = dense_components(xyz[plant], radius)
    counts = np.bincount(components[components >= 0])
    order = np.argsort(-counts, kind='stable'); main = int(order[0])
    if len(order) > 1 and counts[main] < 1.5 * counts[order[1]]:
        raise RuntimeError('多个相近规模植株实例，请指定更清晰的初始化帧 '
                           f'(largest={int(counts[main])}, second={int(counts[order[1]])})')
    tree = cKDTree(xyz[pi[components == main]])
    accepted = components == main
    component_report = []
    for c in order:
        gap = float(tree.query(xyz[pi[components == c]])[0].min())
        keep = gap <= 3 * radius
        accepted |= (components == c) & keep
        component_report.append({'count': int(counts[c]), 'gap': gap, 'kept': bool(keep)})
    target = np.zeros(len(ids), bool); target[pi[accepted]] = True
    # Distractors can leak into only a few frames, so majority voting alone
    # misses them. Any repeatedly masked track far from the target is a negative.
    target_tree=cKDTree(xyz[target])
    neighbour=(votes[:,1]>=2)&(target_tree.query(xyz)[0]>4*radius)
    rows, xy = observations[init_idx]
    initial_plant = labels[init_idx][xy[:, 1], xy[:, 0]] == 1
    if int((target[rows] & initial_plant).sum()) < 10:
        raise RuntimeError('三维主实例与初始化目标不一致 '
                           f'(shared_tracks={int((target[rows] & initial_plant).sum())}/10)')
    for index in observations:
        observations[index]=(*observations[index],cube_regions[index])
    return observations, target, neighbour, {'version': VERSION, 'implementation_sha256': IMPLEMENTATION_SHA256, 'cube_scale': scale,
        'radius': radius, 'target_support': int(target.sum()), 'neighbour_support': int(neighbour.sum()),
        'components': component_report, 'model_directory': str(model)}


def refine_training_masks(scene, images, frame_results, init_idx, predictor):
    observations, target, neighbour, report = build_support(scene, images, frame_results, init_idx)
    recover_prompts = os.environ.get('MASK_PROMPT_RECOVERY', '0') == '1'
    report['prompt_recovery'] = recover_prompts
    report['frames'] = []
    report['low_support_policy'] = 'retain_sam2_propagation'
    report['low_support_frames'] = []
    for index in sorted(frame_results):
        previous = frame_results[index]
        rows, xy, cube_box = observations[index]
        positive = xy[target[rows]]
        negative = xy[neighbour[rows]]
        if len(positive) < 6:
            # Sparse tracks in one view do not invalidate the already propagated
            # SAM2 mask. Keep its plant labels instead of making an unreliable
            # tight crop, dropping a training view, or marking the plant background.
            # The scene-level identity checks and final 3D cleanup still apply.
            output = previous.copy()
            cube_region = np.zeros_like(previous, dtype=bool)
            x1, y1, x2, y2 = np.rint(cube_box).astype(int)
            if x1 <= x2 and y1 <= y2:
                cube_region[y1:y2+1, x1:x2+1] = True
            output[(previous == 2) & ~cube_region] = 0
            frame_results[index] = output
            report['low_support_frames'].append(images[index].name)
            report['frames'].append({'image': images[index].name,
                'mode': 'sam2_propagation_low_sfm_support',
                'observed_target_support': int(len(positive)),
                'positive_points': 0, 'negative_points': 0,
                'positive_hit': None, 'negative_hit': None,
                'plant_pixels_before': int((previous == 1).sum()),
                'plant_pixels_after': int((output == 1).sum()),
                'rubik_pixels_before': int((previous == 2).sum()),
                'rubik_pixels_after': int((output == 2).sum())})
            print('[Instance guard warning]', images[index].name,
                f'目标SfM支持={len(positive)}<6；保留SAM2传播掩码，跳过该帧SfM细化', flush=True)
            continue
        h, w = previous.shape
        # Isolated SfM mismatches can land on the target stem. A distractor
        # negative needs a local group of at least three observed features.
        if len(negative):
            local = cKDTree(negative).query_ball_point(negative, .04 * np.hypot(h, w), return_length=True)
            negative = negative[local >= 3]
        low = positive.min(0); high = positive.max(0)
        margin = np.maximum((high - low) * .12, 12)
        box = np.r_[np.maximum(low - margin, 0), np.minimum(high + margin, [w - 1, h - 1])].astype(np.float32)
        pos = sample_prompts(positive, 6); neg = sample_prompts(negative, 16)
        x1,y1,x2,y2=np.rint(box).astype(int)
        offset=np.array([x1,y1],np.float32)
        neg_input=neg[(neg[:,0]>=x1)&(neg[:,0]<=x2)&(neg[:,1]>=y1)&(neg[:,1]<=y2)]
        coords = np.vstack((pos, neg_input)).astype(np.float32)-offset
        point_labels = np.r_[np.ones(len(pos)), np.zeros(len(neg_input))].astype(np.int32)
        rgb = cv2.cvtColor(cv2.imread(str(images[index])), cv2.COLOR_BGR2RGB)
        predictor.set_image(rgb[y1:y2+1,x1:x2+1])
        prior = cv2.resize((previous[y1:y2+1,x1:x2+1] == 1).astype(np.float32), (256, 256), interpolation=cv2.INTER_LINEAR) * 8 - 4
        mask_input=prior[None]
        for attempt in range(3):
            masks, scores, low_res = predictor.predict(point_coords=coords, point_labels=point_labels,
                box=np.array([0,0,x2-x1,y2-y1],np.float32), mask_input=mask_input, multimask_output=True)
            candidates=[]
            for j,m in enumerate(masks):
                full=np.zeros_like(previous,dtype=bool);full[y1:y2+1,x1:x2+1]=np.asarray(m,bool);m=full
                ph=float(m[pos[:,1].astype(int),pos[:,0].astype(int)].mean())
                nh=float(m[neg[:,1].astype(int),neg[:,0].astype(int)].mean()) if len(neg) else 0.
                overlap=float(np.count_nonzero(m & (previous==1))/max(1,np.count_nonzero(m | (previous==1))))
                candidates.append((2*ph-3*nh+.2*overlap,j,ph,nh))
            _,j,positive_hit,negative_hit=max(candidates)
            plant_mask=np.zeros_like(previous,dtype=bool);plant_mask[y1:y2+1,x1:x2+1]=np.asarray(masks[j],bool)
            if positive_hit>=.8 and negative_hit<=.1:break
            mask_input=low_res[j:j+1]
            wrong=neg[plant_mask[neg[:,1].astype(int),neg[:,0].astype(int)]] if len(neg) else neg
            wrong=wrong[(wrong[:,0]>=x1)&(wrong[:,0]<=x2)&(wrong[:,1]>=y1)&(wrong[:,1]<=y2)]
            coords=np.vstack((coords,wrong-offset));point_labels=np.r_[point_labels,np.zeros(len(wrong),np.int32)]
            if recover_prompts:
                # Correct both kinds of failed prompt, without changing the
                # original positive/negative acceptance thresholds.
                missed=positive[~plant_mask[positive[:,1].astype(int),positive[:,0].astype(int)]]
                missed=sample_prompts(missed,24)
                coords=np.vstack((coords,missed-offset))
                point_labels=np.r_[point_labels,np.ones(len(missed),np.int32)]
        if positive_hit < .8 or negative_hit > .1:
            debug=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            tint=debug.copy();tint[plant_mask]=(0,220,0);debug=cv2.addWeighted(debug,.65,tint,.35,0)
            for x,y in pos:cv2.circle(debug,(int(x),int(y)),8,(0,255,0),2)
            for x,y in neg:cv2.circle(debug,(int(x),int(y)),8,(0,0,255),2)
            cv2.rectangle(debug,tuple(box[:2].astype(int)),tuple(box[2:].astype(int)),(255,0,0),2)
            cv2.imwrite(str(Path(scene)/'instance_guard_failure.jpg'),debug)
            raise RuntimeError(f'{images[index].name}: SAM实例提示未满足 positive={positive_hit:.3f}, negative={negative_hit:.3f}')
        output = np.zeros_like(previous)
        output[plant_mask] = 1
        # Preserve the already tracked, single physical cube, including occlusion.
        cube_region=np.zeros_like(previous,dtype=bool)
        x1,y1,x2,y2=np.rint(cube_box).astype(int)
        if x1<=x2 and y1<=y2:cube_region[y1:y2+1,x1:x2+1]=True
        output[(previous == 2)&cube_region] = 2
        frame_results[index] = output
        report['frames'].append({'image': images[index].name, 'positive_points': len(pos), 'negative_points': len(neg),
            'mode': 'sfm_sam_refined', 'observed_target_support': int(len(positive)),
            'positive_hit': positive_hit, 'negative_hit': negative_hit,
            'plant_pixels_before': int((previous == 1).sum()), 'plant_pixels_after': int((output == 1).sum()),
            'rubik_pixels_before': int((previous==2).sum()), 'rubik_pixels_after':int((output==2).sum())})
        print('[Instance guard]', images[index].name, report['frames'][-1], flush=True)
    return report
