# Third-party materials

The repository licenses cover the authors' contributions described in
[LICENSING.md](LICENSING.md). They do not replace licenses or access conditions
of external software, pretrained models or other third-party material.

## Frozen visual encoders

The paper uses **DINOv3**, provided by Meta, and **I-JEPA**, provided by Meta.
The pretrained backbone weights are not redistributed in this repository.

| Encoder | Official checkpoint used in the paper | Upstream terms |
|---|---|---|
| DINOv3 ViT-S/16 | [facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m/tree/114c1379950215c8b35dfcd4e90a5c251dde0d32) | [DINOv3 license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) and official checkpoint access conditions |
| I-JEPA ViT-H/14 | [facebook/ijepa_vith14_1k](https://huggingface.co/facebook/ijepa_vith14_1k/tree/f157467ea509bc356ff9f61fd3c0d840eec5e04e) | [Official model card](https://huggingface.co/facebook/ijepa_vith14_1k) identifies CC BY-NC 4.0; see [upstream license](https://github.com/facebookresearch/ijepa/blob/main/LICENSE) |

Request any required authorization through the official provider and use your own
account. The CC BY 4.0 grant for the authors' data and artifacts does not grant
rights to Meta's materials. In particular, the I-JEPA backbone's noncommercial
terms are not replaced by the MIT license for our scripts. The saved feature
arrays, PCA/scalers and small trained adapters are not copies of the frozen
backbone weights; their inclusion does not remove any applicable upstream
conditions.

## Software dependencies

Packages installed from `requirements.txt` and `requirements-mesh.txt` retain
their respective licenses. Dependencies are installed separately; their source
code is not relicensed by this repository. Gmsh is an external mesh-generation
dependency; the authors' supplied meshes are covered by the data license.

## Attribution

Keep any upstream attribution and license notices when redistributing material
that requires them. The publication's references acknowledge the visual models
and numerical methods. No endorsement by Meta, IEEE or the conference is implied.
