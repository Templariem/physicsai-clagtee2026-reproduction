"""Illustrative queries for the five Figure 5 frames without a probe.

The geometry rule is fixed in data/figure5_query_policy.json before
inference. No fitting, scoring, annotation editing or fallback labels.
"""
from pathlib import Path
import argparse
import datetime
import dataclasses
import gc
import hashlib
import json
import os
import sys

from .runtime import ROOT as BASE, load_backbone
RUN=BASE/'reference'
DATA=BASE/'data'
import cv2
import matplotlib.tri as mtri
import numpy as np
from scipy.interpolate import interp1d
import torch
from transformers import AutoImageProcessor, AutoModel, IJepaModel
from . import core as m
from .residual import VisualResidualMLP
from .calibration import voltage as curve, decay_derivative


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=BASE/'runs/figures')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    policy_path = BASE/'data/figure5_query_policy.json'
    policy = json.loads(policy_path.read_text())
    assert policy['policy'] == 'virtual_query' and policy['fixed_before_inference']
    names = policy['frames_without_query']
    annotations = json.loads((DATA/'annotations.json').read_text())
    valid = json.loads((DATA/'regression_annotations.json').read_text())
    assert not set(names) & set(valid)
    sim = np.load(DATA/'fem_prior.npz')
    triangles = [t for e in sim['elements'] for t in [[e[0], e[1], e[2]], [e[0], e[2], e[3]]]]
    mesh = mtri.Triangulation(sim['nodes'][:, 0], sim['nodes'][:, 1], triangles)
    fem = mtri.LinearTriInterpolator(mesh, sim['U_abs'])
    samples, records = [], []
    for column, name in zip(policy['columns_without_query'], names):
        image = cv2.cvtColor(cv2.imread(str(DATA/'frames' / name)), cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        conductors = [np.asarray(p['points'], np.int32) for p in annotations[name]
                      if p['class'] in ['esfera', 'bobina_cobre']]
        sphere = next(np.asarray(p['points'], np.int32) for p in annotations[name] if p['class'] == 'esfera')
        moments = cv2.moments(sphere)
        cx, cy = int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00'])
        radius = int(np.mean(np.linalg.norm(sphere - [cx, cy], axis=1)))
        sign = 1 if w - 1 - cx >= cx else -1
        qx, qy = cx + sign * 2 * radius, cy
        assert 0 <= qx < w and 0 <= qy < h
        distances = [-cv2.pointPolygonTest(p, (float(qx), float(qy)), True) for p in conductors]
        assert min(distances) > 0, name
        scale = 4.5/(2.0 * radius)
        distance = min(distances) * scale
        radial = np.hypot(qx-cx, qy-cy) * scale
        fv = fem(radial/100., .1225)
        assert not np.ma.is_masked(fv) and np.isfinite(float(fv))
        derivative = float(decay_derivative(distance))
        reference = float(curve(distance))
        physical = np.asarray([float(fv), derivative, radial, qx/w, qy/h], np.float32)
        # NaN explicitly prevents confusing a display reference with a label.
        samples.append(m.Sample(name, image, m.video_id(name), physical, float('nan'), radial, qx/w, qy/h))
        records.append({'frame': name, 'column': column, 'query_kind': 'virtual',
                        'sphere_xy_radius_px': [cx, cy, radius], 'query_xy_px': [qx, qy],
                        'image_wh_px': [w, h], 'distance_to_conductor_cm': distance,
                        'physical_features': physical.tolist(), 'g_reference_V': reference,
                        'reference_is_measurement': False, 'used_for_fitting_or_scoring': False})

    archive = np.load(RUN / 'results/sample_features.npz')
    indices = [list(archive['frame']).index(n) for n in names]
    centered = [dataclasses.replace(s, x_norm=.5, y_norm=.5) for s in samples]
    arrays = {'frames': np.asarray(names), 'physical': np.stack([s.phys for s in samples])}
    models = {}
    specs = [
        ('DINOv3', 'facebook/dinov3-vits16-pretrain-lvd1689m', '114c1379950215c8b35dfcd4e90a5c251dde0d32',
         'features_dinov3_s16_aug14.npz', 'checkpoint_dinov3_residual_final.pt'),
        ('I-JEPA', 'facebook/ijepa_vith14_1k', 'f157467ea509bc356ff9f61fd3c0d840eec5e04e',
         'features_ijepa_h14_aug14.npz', 'checkpoint_ijepa_residual_fixed.pt'),
    ]
    for name, identifier, revision, cache_name, ck_name in specs:
        print(f'Computing {name} at five fixed virtual queries on CPU...', flush=True)
        processor = AutoImageProcessor.from_pretrained(identifier, revision=revision, local_files_only=True)
        cls = AutoModel if name == 'DINOv3' else IJepaModel
        backbone = cls.from_pretrained(identifier, revision=revision, local_files_only=True).eval()
        raw, controls = [], []
        with torch.inference_mode():
            for i, sample in enumerate(samples):
                output = backbone(**processor(images=[sample.image_rgb.copy()], return_tensors='pt'))
                if name == 'DINOv3':
                    global_feature = output.pooler_output.numpy()
                    tokens, grid = output.last_hidden_state[:, 5:, :], 14
                else:
                    tokens, grid = output.last_hidden_state, 16
                    global_feature = tokens.mean(dim=1).numpy()
                raw.append(np.concatenate([global_feature, m.weighted_local(tokens, samples, [i], grid)], axis=1)[0])
                controls.append(np.concatenate([global_feature, m.weighted_local(tokens, centered, [i], grid)], axis=1)[0])
        raw = np.asarray(raw)
        original = np.load(BASE/'reference/features' / cache_name)['clean'][indices]
        difference = float(np.max(np.abs(np.asarray(controls)-original)))
        assert difference < .005, (name, difference)
        ck_path = RUN/'checkpoints' / ck_name
        ck = torch.load(ck_path, map_location='cpu', weights_only=False)
        x = torch.tensor(ck['physics_scaler'].transform(arrays['physical']), dtype=torch.float32)
        visual = ck['visual_scaler'].transform(ck['pca'].transform(raw))
        members = []
        with torch.inference_mode():
            for state in ck['physics_state_dicts']:
                head = m.PhysicsAdapter(5).eval(); head.load_state_dict(state)
                members.append(head(x).numpy().ravel())
            head = VisualResidualMLP(128).eval(); head.load_state_dict(ck['residual_state_dict'])
            residual = head(torch.tensor(visual, dtype=torch.float32)).numpy().ravel()
        physical_mean = np.mean(members, axis=0)
        prediction = physical_mean + ck['gate'] * residual
        assert np.all(np.isfinite(prediction))
        for i, record in enumerate(records):
            record[name+'_prediction_V'] = float(prediction[i])
            record['physical_prediction_V'] = float(physical_mean[i])
        arrays[name+'_raw'] = raw
        models[name] = {'checkpoint': identifier, 'revision': revision, 'head_sha256': sha(ck_path),
                        'center_pooling_control_max_abs_difference_from_archived_features': difference}
        print(name, 'predictions:', prediction.tolist(), 'center-control max difference:', difference, flush=True)
        del backbone, output, tokens, head
        gc.collect()
    np.savez_compressed(args.output/'figure5_virtual_query_features.npz', **arrays)
    result = {'date': datetime.date.today().isoformat(), 'device': 'cpu', 'torch_threads': 4,
              'policy_sha256': sha(policy_path), 'inference_script_sha256': sha(__file__),
              'training': False, 'metrics_recomputed': False, 'models': models, 'queries': records,
              'calibration': 'five-LED fitted power law only',
              'normalization': 'The renderer uses prediction/P(query), not the empirical voltage reference.'}
    (args.output/'figure5_virtual_query_predictions.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
