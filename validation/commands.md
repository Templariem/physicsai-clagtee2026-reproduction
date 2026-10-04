# Commands used for the Mac/CUDA check

The source directory was copied to the author's homelab over SSH. Two authorized official model snapshots were copied separately to its private Hugging Face cache, outside the repository. No token was transferred.

## Clean Linux installation

```sh
python3 -m venv ~/physicsai_clagtee2026_validation/.venv
~/physicsai_clagtee2026_validation/.venv/bin/python -m pip install torch==2.13.0 torchvision==0.28.0 transformers==5.14.1 numpy==2.5.1 scipy==1.18.0 scikit-learn==1.9.0 pandas==3.0.3 opencv-python==5.0.0.93 Pillow==12.3.0 matplotlib==3.11.0 gmsh==4.15.2 openpyxl huggingface-hub==1.24.0
~/physicsai_clagtee2026_validation/.venv/bin/python -m pip check
```

The installed top-level versions are those in `requirements.txt` and optional `requirements-mesh.txt`; `cuda_requirements_freeze.txt` records the resolved transitives. The default Linux PyTorch wheel supplied CUDA 13.0 and `sm_120`; `torch.cuda.is_available()` and an actual CUDA matrix product were checked.

## Experiment commands

The commands below used the selected environment's `python` and ran from the package root. `HF_HUB_CACHE` pointed to the authorized private cache on Linux.

```sh
python -m physicsai.evaluate --device cuda --output validation/cuda_saved_features
python -m physicsai.train --device cuda --output runs/cuda_retrained_cached
python -m physicsai.features --device cuda --offline --output runs/cuda_features
python -m physicsai.evaluate --device cuda --features runs/cuda_features --output validation/cuda_fresh_features
python -m physicsai.train --device cuda --features runs/cuda_features --output runs/cuda_retrained_fresh
python -m physicsai.vision_only --device cuda --features runs/cuda_features --output runs/cuda_vision_retrained
python -m physicsai.verify
MPLBACKEND=Agg python fem/exp_3.py --output-dir runs/fem3 --skip-mesh-plots
python -m physicsai.benchmark --device cuda --output validation/cuda_timing
```

On the Mac, `--device mps` was used for saved-head evaluation, a fresh `--clean-only` extraction of all frames, re-evaluation, complete hybrid training from the paper features, all ten vision-only models, and the five-pass benchmark. The new latent/PCA tiles were extracted with MPS; virtual-query predictions were recomputed on CPU. Experiments I–III, five-LED sensitivity, all computational figures and the data/fold audit were also executed.

Training uses the paper's epochs and seed schedule without a fresh search. CUDA and MPS run sequentially through each model; timing started after our training jobs completed. Cross-platform retraining differences were retained without tuning to the holdout.
