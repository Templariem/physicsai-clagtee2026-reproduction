# CLAGTEE 2026 — complete paper reproduction v1.0

Public release `clagtee-2026-full-v1.0`, approved by the first author on October 4, 2026. This version pins the complete reproduction package used for the revised paper.

- Credits all seven paper authors in manuscript order, with Giovanni Cocca-Guardia identified as first author.
- Contains only the revised paper's FEM and AI experiments, 271 input images, annotations, meshes, five-LED calibration, fixed splits, trained adapters/PCA/scalers and original frozen-feature arrays.
- Includes official, pinned backbone download instructions; no pretrained DINOv3/I-JEPA weights or credentials are redistributed.
- Supports CPU, Apple MPS and NVIDIA CUDA through a device option.
- Tested on the original M4 Pro MacBook (48 GB) and a clean Python environment on RTX 5060 Ti (16 GB). Fresh-feature CUDA inference differs from the paper by at most 7.424e-7 V. Cross-platform training differences, including the more variable vision-only diagnostic, are retained in `validation/REPORT.md`.
- The new timing check gives 79.89/12.95 FPS on MPS and 115.11/25.26 FPS on CUDA for DINOv3/I-JEPA respectively. The paper's original timing results remain unchanged.
- Regenerates Figure 5's latent/PCA displays using the actual official encoders. Voltage arrays, predictions, data splits and reported numerical tables are unchanged.
- Restores the author's preferred green/orange/purple display palette in both DINOv3 rows using one documented RGB transform. The official DINOv3 features and all other figure rows remain unchanged.
- The README shows experiment coverage, saved-head agreement, adapter-retraining metrics and inference timings for the MacBook M4 Pro and RTX 5060 Ti.
- The extended private implementation and the older prose-only adapter guide are excluded. The existing public FEM repository/release is unchanged.
- Archives the exact training protocol whose hash is recorded in the saved hybrid checkpoints; the package protocol adds portability metadata without changing training settings.

The repository's source archive contains the experimental code and data; the release assets provide scientific input hashes and the platform comparison. The experimental photographs for Figure 3 are included in `data/figure3_led_setup.png`. See `README.md` for commands. The public history starts with the reviewed reproduction package; manuscript files and earlier private development history are retained separately in private author archives.
