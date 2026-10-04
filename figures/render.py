"""Replot voltage displays using the author-approved shared 0.1--8 V scale.

No training or encoder inference in this renderer. Virtual-query predictions
are supplied by physicsai.virtual_queries. RGB/layout come from the submitted
figure; latent tiles come from physicsai.latents using the official encoders.
Scalar profiles use the fitted law from the five documented LEDs only.
Hybrid profiles are normalized by the displayed base profile at their query,
so the scalar voltage at that location matches the model prediction.
Interpolation precedes color lookup, then 60% map + 40% RGB reveals the coil.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch

from voltage_display import VOLTAGE_NORM, VOLTAGE_CMAP, VOLTAGE_TICKS
from five_led_figures import calibration_plot

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / 'data'
RESULTS = BASE / 'reference/features'
CURATED = BASE / 'reference'
OUT = BASE / 'runs/figures'
OUT.mkdir(parents=True,exist_ok=True)
MAP_OPACITY = 0.60
sys.path.insert(0,str(BASE))
from physicsai.core import PhysicsAdapter
from physicsai.residual import VisualResidualMLP
from physicsai.calibration import voltage as calibrated_voltage
from physicsai.latent_palette import apply_dino_palette


def selected_predictions(audit, selected, virtual_queries):
    predictions = {}
    valid_names = set(json.loads((DATA/'regression_annotations.json').read_text()))
    valid_positions = [k for k, index in enumerate(selected) if str(audit['frame'][index]) in valid_names]
    valid_selected = selected[valid_positions]
    model_dir = CURATED/'checkpoints'
    for name, cache, checkpoint, archive, column in [
        ('DINOv3', 'features_dinov3_s16_aug14.npz',
         'checkpoint_dinov3_residual_final.pt', 'dinov3_final_holdout_predictions.csv', 'dinov3_residual_v'),
        ('I-JEPA', 'features_ijepa_h14_aug14.npz',
         'checkpoint_ijepa_residual_fixed.pt', 'ijepa_fixed_equation_predictions.csv', 'ijepa_residual_v'),
    ]:
        ck = torch.load(model_dir / checkpoint, map_location='cpu', weights_only=False)
        clean = np.load(RESULTS / cache)['clean'][valid_selected]
        visual = ck['visual_scaler'].transform(ck['pca'].transform(clean))
        physical = ck['physics_scaler'].transform(audit['physical'][valid_selected])
        members = []
        with torch.inference_mode():
            for state in ck['physics_state_dicts']:
                model = PhysicsAdapter(5).eval()
                model.load_state_dict(state)
                members.append(model(torch.tensor(physical, dtype=torch.float32)).numpy().ravel())
            head = VisualResidualMLP(128).eval()
            head.load_state_dict(ck['residual_state_dict'])
            prediction = np.mean(members, axis=0) + ck['gate'] * head(
                torch.tensor(visual, dtype=torch.float32)).numpy().ravel()
        archived = {r['frame']: float(r[name + '_corrected']) for r in csv.DictReader((CURATED / 'results/holdout_predictions.csv').open())}
        for k, index in enumerate(valid_selected):
            if str(audit['frame'][index]) in archived:
                prediction[k] = archived[str(audit['frame'][index])]
        full_prediction = np.full(len(selected), np.nan)
        full_prediction[valid_positions] = prediction
        for k, index in enumerate(selected):
            frame = str(audit['frame'][index])
            if frame in virtual_queries:
                assert frame not in valid_names
                full_prediction[k] = virtual_queries[frame][name + '_prediction_V']
        assert np.all(np.isfinite(full_prediction))
        predictions[name] = full_prediction
    return predictions


def bilinear_at(values, xy):
    """Sample the scalar display grid at a continuous image-query position."""
    x, y = xy
    height, width = values.shape
    assert 0 <= x <= width-1 and 0 <= y <= height-1
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0+1, width-1), min(y0+1, height-1)
    dx, dy = x-x0, y-y0
    return float((1-dy)*((1-dx)*values[y0,x0]+dx*values[y0,x1])
                 + dy*((1-dx)*values[y1,x0]+dx*values[y1,x1]))


def draw_query_marker(panel, xy, virtual):
    """Annotate the query without changing its position or scalar voltage."""
    marker = tuple(round(v) for v in xy)
    for color, thickness in [((0, 0, 0), 6), ((255, 255, 255), 3)]:
        if virtual:
            cv2.drawMarker(panel, marker, color, cv2.MARKER_CROSS, 21, thickness, cv2.LINE_AA)
        else:
            cv2.circle(panel, marker, 10, color, thickness, cv2.LINE_AA)


def generate_mosaic():
    audit = np.load(CURATED / 'results/sample_features.npz')
    selected = []
    for video in range(1, 10):
        indices = np.flatnonzero(audit['video'] == video)
        if video in [5, 8, 9]:
            selected.extend([indices[len(indices)//3], indices[2*len(indices)//3]])
        else:
            selected.append(indices[len(indices)//2])
    selected = np.asarray(selected)
    virtual_path=OUT/'figure5_virtual_query_predictions.json'
    if not virtual_path.exists():virtual_path=BASE/'reference/figures/figure5_virtual_query_predictions.json'
    virtual_record = json.loads(virtual_path.read_text())
    policy_path = BASE/'data/figure5_query_policy.json'
    assert virtual_record['policy_sha256'] == hashlib.sha256(policy_path.read_bytes()).hexdigest()
    virtual_queries = {q['frame']: q for q in virtual_record['queries']}
    predictions = selected_predictions(audit, selected, virtual_queries)
    annotations = json.loads((DATA/'annotations.json').read_text())

    source_path = BASE/'data/figure5_original.png'
    original = np.asarray(Image.open(source_path).convert('RGBA'))
    # Exact opaque mosaic extent in the submitted PNG; labels lie outside it.
    assert original.shape == (1966, 3185, 4)
    source = original[3:1861, 377:3181, :3].copy()
    assert np.all(original[3:1861, 377:3181, 3] == 255)
    mosaic = source.copy()
    latent_path = OUT/'latent_tiles.npz'
    if not latent_path.exists():latent_path=BASE/'reference/figures/latent_tiles.npz'
    if not latent_path.exists():
        raise FileNotFoundError('Run python -m physicsai.latents --device auto first.')
    latent = np.load(latent_path)
    height, width = source.shape[:2]
    row_edges = np.rint(np.linspace(0, height, 9)).astype(int)
    changed = np.zeros((height, width), dtype=bool)
    records = []
    scalar_panels = {}
    # Original mosaic contains 12 x 224 px tiles and two 8 px separators.
    for k, index in enumerate(selected):
        frame = str(audit['frame'][index])
        im = np.asarray(Image.open(DATA/'frames' / frame).convert('RGB'))
        h, w = im.shape[:2]
        polygon = next(p for p in annotations[frame] if p['class'] == 'esfera')
        points = np.asarray(polygon['points'], dtype=np.int32)
        moments = cv2.moments(points)
        cx, cy = int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00'])
        # Use the same integer sphere radius and surface geometry as labels.
        radius = int(np.mean(np.linalg.norm(points - [cx, cy], axis=1)))
        cm_per_px = 4.5/(2.0*radius)
        conductors = [np.asarray(p['points'], dtype=np.int32) for p in annotations[frame]
                      if p['class'] in ('esfera', 'bobina_cobre')]
        x0 = round((k*224 + (k//4)*8)*width/2704)
        x1 = round((k*224 + (k//4)*8 + 224)*width/2704)
        for latent_row,key in [(2,'dinov3_raw'),(3,'ijepa_raw'),(4,'dinov3_pca128'),(5,'ijepa_pca128')]:
            lo,hi=row_edges[latent_row:latent_row+2]
            tile=latent[key][k]
            if key.startswith('dinov3_'):
                tile=apply_dino_palette(tile)
            mosaic[lo:hi,x0:x1]=cv2.resize(tile,(x1-x0,hi-lo),interpolation=cv2.INTER_CUBIC)
            changed[lo:hi,x0:x1]=True
        virtual_query = virtual_queries.get(frame)
        if virtual_query:
            query = virtual_query['query_xy_px']
        else:
            # Match the integer probe centroid used to extract model inputs.
            probes = [p for p in annotations[frame] if p['class']=='diodo_mano']
            assert len(probes)==1
            probe_moments = cv2.moments(np.asarray(probes[0]['points'], dtype=np.int32))
            query = [int(probe_moments['m10']/probe_moments['m00']),
                     int(probe_moments['m01']/probe_moments['m00'])]
            assert np.allclose(np.asarray(query)/[w,h], audit['physical'][index,3:], atol=1e-7)
        for row, name in [(1, None), (6, 'DINOv3'), (7, 'I-JEPA')]:
            low, high = row_edges[row:row+2]
            # Evaluate at displayed pixel centres in original-image coordinates.
            yy, xx = np.mgrid[:high-low, :x1-x0]
            coords = np.column_stack([((xx+.5)*w/(x1-x0)-.5).ravel(),
                                      ((yy+.5)*h/(high-low)-.5).ravel()])
            surface_px = np.full(len(coords), np.inf)
            for conductor in conductors:
                distances = np.fromiter((-cv2.pointPolygonTest(conductor, tuple(point), True)
                                         for point in coords), dtype=float, count=len(coords))
                surface_px = np.minimum(surface_px, distances)
            surface_cm = surface_px.reshape(high-low, x1-x0)*cm_per_px
            air = surface_cm > 0
            base_display = calibrated_voltage(np.maximum(surface_cm, .5*cm_per_px))
            query_display = [(query[0]+.5)*(x1-x0)/w-.5,
                             (query[1]+.5)*(high-low)/h-.5]
            profile_at_query = bilinear_at(base_display, query_display)
            prediction = float(predictions[name][k]) if name else None
            multiplier = prediction/profile_at_query if name else 1.0
            assert np.isfinite(multiplier)
            # Saturation is a display operation, never a label-generation rule.
            displayed = np.clip(base_display*multiplier, .1, 8.)
            if name:
                assert .1 <= prediction <= 8.
                # Check the actual saved map, including clipping/resampling.
                assert abs(bilinear_at(displayed, query_display)-prediction) < 1e-9
            rgba = VOLTAGE_CMAP(VOLTAGE_NORM(displayed), bytes=True)
            photo = cv2.resize(im, (x1-x0, high-low), interpolation=cv2.INTER_AREA)
            panel = cv2.addWeighted(
                rgba[:, :, :3], MAP_OPACITY, photo, 1.0-MAP_OPACITY, 0)
            panel[~air] = photo[~air]
            if row in (6, 7):
                draw_query_marker(panel, query_display, virtual=bool(virtual_query))
            displayed[~air] = np.nan  # No surrogate voltage inside conductors.
            mosaic[low:high, x0:x1] = panel
            changed[low:high, x0:x1] = True
            scalar_panels[f'row{row+1}_column{k+1}'] = displayed
            record = {'row': row+1, 'column': k+1, 'frame': frame,
                      'multiplier': float(multiplier),
                      'min_V': float(np.nanmin(displayed)), 'max_V': float(np.nanmax(displayed))}
            if name:
                record.update(query_xy_px=query, query_xy_display=query_display,
                              prediction_V=prediction, profile_at_query_V=profile_at_query,
                              displayed_at_query_V=bilinear_at(displayed, query_display),
                              normalization='prediction / displayed base-profile value at query',
                              query_kind='virtual' if virtual_query else 'annotated_probe',
                              marker='white cross with black outline' if virtual_query else 'white circle with black outline')
            if virtual_query and row in (6, 7):
                record.update(status='virtual_query', query_xy_px=virtual_query['query_xy_px'],
                              g_reference_V=virtual_query['g_reference_V'],
                              g_reference_used_for_normalization=False, reference_is_measurement=False,
                              used_for_fitting_or_scoring=False, marker='white cross with black outline')
            records.append(record)
    assert np.array_equal(mosaic[~changed], source[~changed])
    # A 1-pixel guard handles antialiasing at the resampled source row edges.
    # Latent/PCA rows now come from the same official backbones as the metrics.
    np.savez_compressed(OUT/'figure5_display_values.npz', **scalar_panels)
    Image.fromarray(mosaic).save(OUT / 'fig5_shared_voltage_mosaic.png')
    layout = render_mosaic(mosaic)
    return {'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
            'unchanged_rgb_pixels': True, 'latent_source': 'official checkpoints, regenerated spatial PCA displays',
            'dinov3_display_palette': 'fixed green/orange/purple RGB transform; reference/figures/dino_display_palette.json; visualization only',
            'source_mosaic_extent_xyxy': [377, 3, 3181, 1861],
            'display_saturation_V': [.1, 8.], 'labels_or_predictions_clipped': False,
            'profile_calibration': 'power-law fit to the five documented LED pairs only',
            'surface_regularization': 'half one source-image pixel outside conductor contours; visualization only',
            'distance_convention': 'nearest signed conductor-polygon distance and integer sphere scale, identical to regression labels',
            'conductor_interiors': 'masked; original photo remains visible, scalar arrays store NaN',
            'hybrid_profile_normalization': 'P(x) * prediction(q) / P(q), then saturate colors at 0.1--8 V; P(q) uses the same bilinear display grid',
            'scalar_interpolation': 'evaluate the surface-distance law at displayed pixel centres; bilinear sampling only for query normalization',
            'RGB_blending': True, 'map_opacity': MAP_OPACITY,
            'photo_opacity': 1.0-MAP_OPACITY, 'colorbar_before_blending': True,
            'query_marker_legend': {'rows': [7, 8], 'circle': 'annotated probe location',
                                   'cross': 'virtual query excluded from fitting and scoring'},
            'layout': layout, 'panels': records}


def render_mosaic(mosaic):
    """Lay out the archived mosaic without repeating any numerical inference."""
    height, width = mosaic.shape[:2]
    # The archived mosaic has gaps after columns 4 and 8. Join columns 1--8
    # into one development group, removing only the first blank separator.
    first_gap = (round(4*224*width/2704), round((4*224+8)*width/2704))
    removed_width = first_gap[1]-first_gap[0]
    display = np.concatenate([mosaic[:, :first_gap[0]], mosaic[:, first_gap[1]:]], axis=1)
    development_end = round((8*224+8)*width/2704)-removed_width
    holdout_start = round((8*224+16)*width/2704)-removed_width
    display_width = display.shape[1]
    groups = [
        (development_end/(2*display_width), 'Training and cross-validation (Videos 1-7)'),
        ((holdout_start+display_width)/(2*display_width), 'Holdout (Videos 8-9)'),
    ]
    figure_width, figure_height = 15.0, 8.1
    label_fontsize = 14.5
    fig = plt.figure(figsize=(figure_width, figure_height))
    left, bottom, plot_width = 2.0/figure_width, .35/figure_height, 11.5/figure_width
    plot_height = plot_width*figure_width*height/width/figure_height
    ax = fig.add_axes([left, bottom, plot_width, plot_height])
    ax.imshow(display, interpolation='none', aspect='auto')
    ax.axis('off')
    labels = ['RGB', 'Empirical\nvoltage (V)', 'DINOv3 latent\n(unitless)',
              'I-JEPA latent\n(unitless)', 'DINOv3 PCA-128\n(unitless)',
              'I-JEPA PCA-128\n(unitless)', 'DINOv3 scaled\nprofile (V)',
              'I-JEPA scaled\nprofile (V)']
    for row, label in enumerate(labels):
        fig.text(left-.065/figure_width, bottom + plot_height*(1-(row+.5)/8), label,
                 ha='right', va='center', fontsize=label_fontsize, fontweight='bold',
                 linespacing=1.05, multialignment='right')
    for center, label in groups:
        fig.text(left+plot_width*center, .14/figure_height, label,
                 ha='center', va='center', fontsize=label_fontsize, fontweight='bold')
    bar_height = 6.8/figure_height
    cax = fig.add_axes([13.68/figure_width, bottom+(plot_height-bar_height)/2,
                       .16/figure_width, bar_height])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=VOLTAGE_NORM, cmap=VOLTAGE_CMAP),
                     cax=cax, orientation='vertical', ticks=VOLTAGE_TICKS, extend='max')
    cb.ax.tick_params(labelsize=12, pad=3)
    cb.set_label('Surrogate voltage (V)\nBefore RGB blending; rows 2, 7, 8',
                 fontsize=12, labelpad=7)
    fig.savefig(OUT / 'fig5_shared_voltage_scale.pdf')
    fig.savefig(OUT/'fig5_shared_voltage_scale.png', dpi=180)
    plt.close(fig)
    return {'figure_size_inches': [figure_width, figure_height],
            'colorbar': 'vertical, right of mosaic', 'row_label_fontsize_pt': label_fontsize,
            'row_label_fontweight': 'bold', 'row_label_alignment': 'right',
            'set_label_fontsize_pt': label_fontsize, 'set_label_fontweight': 'bold',
            'set_labels': [label for _, label in groups], 'group_columns': [[1, 8], [9, 12]],
            'removed_source_separator_x': list(first_gap),
            'remaining_group_separator_after_column': 8,
            'row_label_gap_inches': .065}


if __name__ == '__main__':
    rows = list(csv.DictReader((CURATED / 'results/holdout_predictions.csv').open()))
    y = np.asarray([float(r['target_V']) for r in rows])
    predictions = {name:np.asarray([float(r[name+'_corrected']) for r in rows]) for name in ['Physical','DINOv3','I-JEPA']}
    assert len(y) == 36
    metrics = {r['Model']:{k:float(r[k]) for k in ['MSE_V2','MAE_V','R2']} for r in csv.DictReader((CURATED/'results/holdout_comparison.csv').open()) if r['Stage']=='Surface_distance_retrained_36'}
    within = np.asarray([r['within_interval']=='True' for r in rows])
    figure4 = calibration_plot(y, predictions, within)
    result = {'date': '2026-10-03', 'range_V': [.1, 8.], 'normalization': 'linear, shared',
              'colormap': 'turbo', 'per_panel_autoscaling': False,
              'fig4_above_range': 'saturated at 8 V; upper colorbar extension',
              'mapping_samples_rgb': {str(v): list(VOLTAGE_CMAP(VOLTAGE_NORM(v), bytes=True)[:3])
                                      for v in VOLTAGE_TICKS},
              'figure4': figure4, 'figure5': generate_mosaic(), 'curated_holdout_metrics': metrics,
              'missing_query_policy':'Virtual geometry-defined queries provide illustrative hybrid profiles in five frames; crosses mark their positions. These frames remain excluded from fitting and scoring.'}
    (OUT/'figure_color_consistency.json').write_text(json.dumps(result, indent=2, default=int))
    print('Figures 4 and 5 regenerated: shared Turbo 0.1--8 V; RGB preserved and official-model latent/PCA rows regenerated.')
