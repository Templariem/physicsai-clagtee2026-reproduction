# License scope

Copyright (c) 2026 Giovanni Cocca-Guardia and contributors.

The authors' software and software documentation (`fem/`, `experiments/field/scripts/`, `experiments/field/solver/`, `experiments/field/physical/scripts/`, `experiments/field/physical/solver/`, `figures/`, `tools/`, `physicsai/`) use the MIT license in `LICENSE`.

The authors' rights in experimental frames, reviewed masks, FEM meshes and numerical targets, train/validation/test metadata, calibration, predictions, numerical figure inputs, results and validation records use CC BY 4.0 (`LICENSE-DATA`). Credit Giovanni Cocca-Guardia, this repository/release URL and the license; indicate changes. This grant covers only rights held by the authors, not third-party rights, depicted people, trademarks or unprotectable facts. Cite the complete paper author list for the research.

## Models

Author-created physical heads, dense decoders, residual adapters and PCA artifacts are covered by CC BY 4.0 only to the extent of the authors' rights. **This does not override the encoder licenses.** The DINOv3 partial/full checkpoints contain modified Meta encoder parameters and are distributed under the DINOv3 license in `licenses/DINOv3-LICENSE.md`; retain that license and attribution. The I-JEPA partial/full checkpoints contain modified Meta I-JEPA parameters and retain **CC BY-NC 4.0**, reproduced in `licenses/IJEPA-LICENSE.txt`. Their modification is fine-tuning for dense LED-equivalent electric-potential prediction. No MIT or CC BY statement here grants commercial rights to I-JEPA-derived materials.

The model archive includes both upstream license texts. Official pretrained backbones are downloaded separately from pinned official repositories; access and use require compliance with their terms. Frozen/residual uses also remain subject to the applicable upstream model terms. See `THIRD_PARTY_NOTICES.md`.

## Publication figures and article

The article text, LaTeX/PDF and prepared publication figure compositions are excluded from the MIT and CC BY grants. `reference/figures/*` and prepared RGB/PCA panel compositions in `reference/figure_inputs/panels/*` are supplied to inspect/reproduce the experiment, without an additional reuse grant for those compositions. Publication rights remain reserved except where separate permission or applicable law permits reuse. This does not remove the stated license from underlying numerical data and author-owned frames. No manuscript or camera application is included.
