# Licensing and attribution

Effective October 4, 2026. This file identifies the material covered by each
license. The MIT license at the repository root applies to the authors' software
and accompanying software documentation, not to every file in this repository.

## Software — MIT

Copyright (c) 2026 Giovanni Cocca-Guardia and contributors.

The authors' Python source in `fem/`, `physicsai/`, `figures/` and `validation/`,
the installation requirements, `.gitignore`, `protocol.json`, and the authors'
Markdown software documentation are licensed under [MIT](LICENSE). Embedded
publication figures and externally linked material retain their separate status
below. Standard license texts retain their own terms.

## Data and research artifacts — CC BY 4.0

Copyright (c) 2026 Giovanni Cocca-Guardia.

The author's rights in the following material are licensed under the
[Creative Commons Attribution 4.0 International license](LICENSE-DATA):

| Material | Repository paths |
|---|---|
| Original experimental frame dataset | `data/frames/*.jpg` |
| Annotations and query policy | `data/annotations.json`, `data/regression_annotations.json`, `data/figure5_query_policy.json` |
| FEM input and meshes | `data/fem_prior.npz`, `fem/gmsh_meshes/*.npz` |
| Calibration, numerical results, splits and provenance | `reference/fem/`, `reference/results/`, `reference/provenance/` |
| Author-created feature caches, adapters, PCA/scalers and numerical display artifacts | `reference/features/`, `reference/checkpoints/`, and the `.json`/`.npz` files in `reference/figures/`; see the model conditions below |
| Validation records and checksums | Numerical records, environment lists and logs in `validation/`, and `MANIFEST.sha256`; Python and Markdown files use MIT as specified above |

CC BY 4.0 permits sharing and adaptation, including commercial reuse, of the
author-owned material it covers, with attribution, a license link, and an
indication of changes. It grants only rights the licensor holds. It does not
license third-party material, trademarks, or rights in people depicted in images,
and does not create exclusive rights over facts or other unprotected material.
The canonical license is <https://creativecommons.org/licenses/by/4.0/>.

For dataset attribution, retain the credit **Giovanni Cocca-Guardia**, the
repository/release URL, and the CC BY 4.0 notice. When citing the associated
research, use the paper's complete author list in the order shown in the README.
An academic citation is requested in addition to the legal notices; this request
does not add a condition to MIT or CC BY 4.0.

### Model conditions

The saved adapters were trained by the authors with frozen visual encoders; the
backbone weights are not included. CC BY 4.0 covers only the authors' rights in
the shipped research artifacts. It does not relicense Meta's DINOv3 or I-JEPA
materials or override any applicable upstream conditions. Downloading and using
the backbones requires compliance with their official licenses, including
DINOv3's access conditions and I-JEPA's noncommercial terms. A permissive license
for our scripts is not a commercial-use license for those backbones. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Article and publication figures — excluded

The article text, LaTeX/PDF, and prepared publication figure compositions are
excluded from the MIT and CC BY 4.0 grants above. In this repository, the excluded
figure files are:

- `assets/fig2.png`
- `assets/fig5.png`
- `data/figure3_led_setup.png`
- `data/figure5_original.png`

They are provided to display or reproduce the reported experiment; no additional
reuse license is granted for these figure compositions here. Rights are reserved
except where a separate permission or applicable law allows reuse. Their
copyright and reuse conditions are handled separately with the article's
publication agreement. These exclusions do not remove CC BY 4.0 from the
underlying frames and numerical data explicitly licensed above.

## Applicability to archived releases

These grants also apply, with the same scope and exclusions, to the covered
files in the original `clagtee-2026-full-v1.0` release, from the effective date
above. That tag and its scientific files are unchanged. The
`clagtee-2026-full-v1.0.1` release packages these license notices alongside the
same code, data, figures, trained artifacts and results. No scientific result or
model has been updated by this licensing release.
