# Explicit 2D FEM solver — CLAGTEE 2026

Python implementation accompanying **A Physics-AI Framework for Video-Assisted Electric Potential Mapping Using 2D FEM: A Tesla Coil Case Study**.

Authors: Giovanni Cocca-Guardia, Gabriel Hermosilla, Hernán Mella, Gabriel Olmos, Manuel Silva, Lisa Soto, Sebastian Quiñones. Pontificia Universidad Católica de Valparaíso, Chile.

**P1 triangles and Q4 quadrilaterals are both implemented and selectable.** The solver explicitly forms gradients, element stiffness/mass matrices and sparse assemblies; SciPy solves the systems. Q4 uses 2×2 Gauss integration. Gmsh generates meshes. Connectivity selects P1 (three nodes) or Q4 (four nodes). The reported paper targets/figures use P1.

```sh
python -m pip install -r requirements.txt
python run_verification.py --element P1
python run_verification.py --element Q4
python run_benchmarks.py
python ruler/verify.py
python ruler/solve.py --help
```

`run_verification.py` reproduces electrostatic, magnetostatic, harmonic and transient verification on three meshes, plus the coil prior and five LED samples. Outputs are under `results/P1/` or `results/Q4/`. P1 verification meshes are formed by splitting each supplied quadrilateral along local diagonal 0–2 and assembling new triangle matrices; Q4 retains quadrilateral connectivity. Both use the same nodes, material tags and boundary values.

`run_benchmarks.py` solves and plots the electric dipole, permanent magnet, parallel wires and Faraday induction with P1. `benchmark_meshes/` supplies the four tagged input meshes. Outputs are under `results/benchmarks/`.

`ruler/` contains the interchangeable planar coil/conductor electrostatic solver. Its independent tests check affine patches, manufactured harmonic convergence, Q4 interpolation/gradients, charge balance and regression against the P1 teacher implementation. All study ruler targets use the grounded-conductor case. Equal requested mesh size does not imply equal node counts across element families.

The complete reviewed dataset, neural experiments, trained checkpoints and paper figures/tables are in [the reproduction package](https://github.com/Templariem/physicsai-clagtee2026-reproduction/releases/tag/clagtee-2026-full-v2.0). Neither repository includes a camera application. Validation records are in `validation/`; execution was checked on the Linux homelab. No models are trained by these FEM commands.

Code is MIT (`LICENSE`); author-owned meshes and numerical records are CC BY 4.0 (`LICENSE-DATA`), attributed to Giovanni Cocca-Guardia. Third-party dependencies retain their licenses. Gmsh is installed separately. The article and prepared publication compositions are outside these grants.
