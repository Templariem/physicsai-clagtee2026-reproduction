# Reproducing the experiments

The default README workflow uses the published checkpoints. It does not retrain them. The following optional workflow is for an independent training replication and can take many hours on the RTX 5060 Ti. Results need not be bitwise identical across library/hardware versions. Preserve the shipped snapshot and compare your new results separately.

## Independent training copy

First download `dataset models physical-training` with `tools/download_assets.py` and obtain the official pinned encoders. Then:

```sh
python tools/create_training_workspace.py ../physicsai-training-replica
export PHYSICS_ENCODERS="$(pwd)/encoders"
cd ../physicsai-training-replica
python physical/scripts/entrenar.py
python scripts/preparar_residuales.py
python scripts/entrenar.py --model dinov3 --arm frozen
python scripts/entrenar.py --model dinov3 --arm partial
python scripts/entrenar.py --model dinov3 --arm full
python scripts/entrenar.py --model dinov3 --arm residual
python scripts/entrenar.py --model ijepa --arm frozen
python scripts/entrenar.py --model ijepa --arm partial
python scripts/entrenar.py --model ijepa --arm full
python scripts/entrenar.py --model ijepa --arm residual
```

`create_training_workspace.py` refuses an existing destination, copies prepared research inputs, and omits learned weights, geometry predictions, results and frozen-run guards. It retains the fixed no-ruler FEM numerical prior and training-only mean target map. The physical ensemble trains first; the next step fits PCA on training RGB only and builds physical geometry predictions with that ensemble. Dense and residual methods then follow the shipped optimizer, augmentation, weighted sampling and validation stopping code. Physical training uses the optional cached query samples; it does not use RGB augmentation. The six dense RGB variants share geometric/photometric augmentations. Do not change a split, select an epoch, tune hyperparameters or choose a model using test errors.

Original frozen checkpoint/configuration hashes remain in the published snapshot. `validation/packaging_changes.json` records portable path substitutions. Raw provenance retains the original acquisition/configuration names and historical paths solely for traceability; the runtime uses relative image/mask/teacher paths. `config/protocolo.json` records the training protocol as captured, including planning notes that are not additional paper experiments. The paper's measured inference protocol is `reference/inference_homelab.json`.

## FEM target regeneration

The accepted dataset includes reviewed masks and exact normalized P1 targets. `experiments/field/scripts/preparar_docente.py` and `docente_base.py` implement geometry registration, grounded ruler, the fixed LED fit and monotone amplitude transfer. The solver copy in `experiments/field/solver/fem_regla.py` is the precise P1 teacher used for training. The standalone `fem/ruler/fem_regla.py` additionally offers Q4 and is regression-tested against that implementation.

To inspect or regenerate the 12 displayed teacher meshes without changing any targets:

```sh
python tools/export_figure_meshes.py runs/rebuilt_meshes
```

The exporter solves each recorded scene and asserts agreement with its saved teacher values and mesh counts. It is not a surrogate illustration of a mesh. To rebuild the complete accepted teacher set in an independent workspace, run `python tools/rebuild_teachers.py ../physicsai-teacher-replica`. This refuses an existing destination and copies only the inputs needed for target generation. Mesh generation can vary across Gmsh versions; the shipped target arrays and meshes define the reported experiment.

## Figure inputs from checkpoints

The figure renderer accepts the included, checked numerical inputs. For an independent reconstruction of potential maps and PCA panels:

```sh
python tools/export_figure_fields.py runs/rebuilt_fields
python tools/export_figure_panels.py runs/rebuilt_panels
python tools/export_figure_meshes.py runs/rebuilt_meshes
```

The field exporter restores the same four display checkpoints and fixed physical map. The panel exporter uses the preregistered display cases and training-only PCA fit pool; one basis per backbone is fitted jointly to frozen/partial/full training representations and reused for the residual display. PCA colors are unitless and cannot be compared physically across backbones. The fixed DINO color transform and percentile bounds are recorded. The three commands write separate directories and do not replace publication inputs. Figure 5 computes gradients before cropping, preserves conductor edges, and uses one train-derived field scale across methods. The renderer's explicit input paths can be pointed to these rebuilt directories in a separate working copy.

## Mapping to the paper

| Paper item | Implementation / source |
|---|---|
| Figure 1; Table I | `fem/run_verification.py --element P1` |
| Figure 2 | `fem/run_benchmarks.py` |
| Figure 3 | `reference/figures/figure3.png` |
| Figure 4; Table II | P1 coil prior, `physicsai/calibration.py`, `figures/figure4.py` |
| Figure 5 | `figures/figure5.py`, `reference/figure_inputs/` |
| Table III accuracy | `tools/evaluate.py`, `experiments/field/resultados/tabla_comparacion.csv` |
| Table III size/latency | `tools/benchmark.py`, `reference/inference_homelab.json` |
| P1/Q4 convergence | `fem/ruler/verify.py` |

All electrical-potential prediction errors use the FEM-derived LED-equivalent target, not independently measured voltages. Dataset sources, the grounded ruler, the video-level diagnostic evaluation and the geometry advantage are stated in the README and paper.
