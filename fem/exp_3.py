"""Experiment III: planar FEM prior and five-LED surrogate calibration.

Drop-in replacement for exp_3.py in Templariem/maxwell_solver_fem_2d.
Requires the repository's solver_fem_2d.py and
gmsh_meshes/exp_3_mesh_calibration.npz; no AI code or weights are used.

The FEM problem is Cartesian, with complex conductivity sigma + j*omega*epsilon.
Its contour amplitudes are remapped using ONE least-squares power law fitted
to five manually observed activation distances and color-assigned LED voltages.
These voltages are surrogate assignments, not measurements of local potential.
The resulting scalar grid matches the construction of revised Figure 4(a).
Figure 4(b), which reports the separate AI experiment, is outside this script.

Run from the repository: python exp_3.py
Optional: python exp_3.py --skip-mesh-plots --output-dir results
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
import numpy as np
from scipy.interpolate import interp1d

from solver_fem_2d import resolver_sistema_maestro


BASE = Path(__file__).resolve().parent
DISTANCE_CM = np.array([5.0, 6.0, 6.5, 12.0, 17.0])
ASSIGNED_V = np.array([3.3, 3.1, 2.5, 2.1, 2.0])
LED_NAMES = ("White", "Blue", "Green", "Yellow", "Red")
LED_COLORS = ("white", "#4297ff", "#31c459", "#ffd43b", "#f04e4e")
R_SPHERE_M, Y_SPHERE_M = 0.0225, 0.1225
R_COIL_M, H_COIL_M = 0.015, 0.10
DOMAIN_M, SOURCE_V = 5.0, 45000.0
FREQUENCY_HZ, EPSILON_F_M = 876000.0, 8.854e-12
CONDUCTIVITY_S_M = {1: 1e-15, 2: 5.8e7, 3: 3.5e7}
DISPLAY_RANGE_V = (0.1, 8.0)
DISPLAY_TICKS_V = [0.1, 1, 2, 3, 4, 5, 6, 7, 8]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.titlesize": 10, "axes.labelsize": 9,
                     "pdf.fonttype": 42, "ps.fonttype": 42})


def fit_calibration():
    """OLS: ln(V/1 V) = b - gamma*ln(d/1 cm), using only the five LED pairs."""
    x, y = np.log(DISTANCE_CM), np.log(ASSIGNED_V)
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope*x + intercept)
    r2 = 1.0 - np.sum(residual**2)/np.sum((y-y.mean())**2)
    return float(np.exp(intercept)), float(-slope), float(r2)


def surrogate_voltage(distance_cm, amplitude_v, exponent):
    """No voltage clipping or invented endpoint anchors; undefined at d <= 0."""
    distance = np.asarray(distance_cm, dtype=float)
    return amplitude_v*np.power(np.where(distance > 0, distance, np.nan), -exponent)


def boundary_conditions(nodes):
    """Same node selection as the published FEM experiment; other fluxes are zero."""
    x, y = nodes.T
    ground = (np.abs(x-DOMAIN_M) < 0.002) | (np.abs(y-DOMAIN_M) < 0.002)
    active = ((x < 0.002) & (y > Y_SPHERE_M-R_SPHERE_M-0.002)
              & (y < Y_SPHERE_M+R_SPHERE_M+0.002))
    values = {int(i): 0.0 for i in np.flatnonzero(ground)}
    values.update({int(i): SOURCE_V for i in np.flatnonzero(active)})
    if not values or not np.any(active):
        raise ValueError("The mesh does not contain the required ground/terminal nodes.")
    return values, ground, active


def solve_prior(nodes, elements, materials, boundaries):
    omega = 2*np.pi*FREQUENCY_HZ
    conductivity = {m: sigma + 1j*omega*EPSILON_F_M
                    for m, sigma in CONDUCTIVITY_S_M.items()}
    zero = {m: 0 for m in CONDUCTIVITY_S_M}
    # 'static' selects an algebraic solve. Complex conductivity contains the
    # frequency dependence; this is not an electrostatic or Az wave solve.
    return resolver_sistema_maestro(nodes, elements, materials, conductivity,
                                   zero, zero, zero, boundaries, "static")


def remap_prior(nodes, elements, magnitude, amplitude_v, exponent):
    triangles = np.vstack([elements[:, [0, 1, 2]], elements[:, [0, 2, 3]]])
    interpolator = mtri.LinearTriInterpolator(
        mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles), magnitude)
    radial_x = np.linspace(R_SPHERE_M+0.0001, DOMAIN_M-0.1, 1000)
    radial_u = np.asarray(interpolator(radial_x, np.full_like(radial_x, Y_SPHERE_M)))
    radial_distance = (radial_x-R_SPHERE_M)*100
    if not np.all(np.isfinite(radial_u)):
        raise ValueError("The mesh must cover the calibration profile.")
    order = np.argsort(radial_u)
    # Endpoint assignment concerns inversion of the finite FEM profile only;
    # it does not add calibration observations or clip surrogate voltages.
    equivalent_distance = interp1d(
        radial_u[order], radial_distance[order], bounds_error=False,
        fill_value=(radial_distance[-1], radial_distance[0]))
    x, y = np.meshgrid(np.linspace(0, 0.30, 400), np.linspace(0, 0.30, 400))
    fem_grid = interpolator(x, y)
    distance = equivalent_distance(np.asarray(fem_grid))
    conductor = ((np.hypot(x, y-Y_SPHERE_M) <= R_SPHERE_M)
                 | ((x <= R_COIL_M) & (y <= H_COIL_M)))
    mask = conductor | np.ma.getmaskarray(fem_grid) | ~np.isfinite(distance)
    voltage = surrogate_voltage(distance, amplitude_v, exponent)
    voltage = np.where(mask, np.nan, voltage)
    inside = (~mask & (distance >= DISTANCE_CM.min()) & (distance <= DISTANCE_CM.max()))
    arrays = {"x_m": x, "y_m": y, "fem_magnitude_V": np.asarray(fem_grid),
              "equivalent_surface_distance_cm": distance,
              "surrogate_voltage_V": voltage, "conductor_mask": conductor,
              "valid_map_mask": ~mask, "within_calibration_interval": inside,
              "extrapolated": ~mask & ~inside,
              "profile_x_m": radial_x, "profile_FEM_V": radial_u,
              "profile_surface_distance_cm": radial_distance}
    return interpolator, arrays


def plot_map(arrays, output):
    """Match Figure 4(a)'s grid, contours, LED markers, and shared voltage scale."""
    x, y = arrays["x_m"], arrays["y_m"]
    voltage = np.ma.masked_invalid(arrays["surrogate_voltage_V"])
    fig, ax = plt.subplots(figsize=(5.6, 4.5), constrained_layout=True)
    colors = ax.pcolormesh(x*100, y*100, voltage, shading="nearest", rasterized=True,
                          cmap="turbo", norm=Normalize(*DISPLAY_RANGE_V, clip=True))
    for distance, assigned, color in zip(DISTANCE_CM, ASSIGNED_V, LED_COLORS):
        ax.contour(x*100, y*100, voltage, levels=[assigned], colors=[color], linewidths=1.2)
        ax.scatter(distance+R_SPHERE_M*100, Y_SPHERE_M*100,
                   c=color, edgecolors="black", s=30, zorder=5)
    theta = np.linspace(-np.pi/2, np.pi/2, 100)
    ax.fill(R_SPHERE_M*100*np.cos(theta), (Y_SPHERE_M+R_SPHERE_M*np.sin(theta))*100,
            color="silver", edgecolor="black")
    ax.fill([0, R_COIL_M*100, R_COIL_M*100, 0], [0, 0, H_COIL_M*100, H_COIL_M*100],
            color="#bb8647", edgecolor="black")
    ax.set(xlim=(0, 30), ylim=(0, 30), xlabel="x (cm)", ylabel="y (cm)",
           title="FEM contours with five-LED fitted remapping")
    ax.set_aspect("equal")
    fig.colorbar(colors, ax=ax, label="Calibration-derived voltage (V)",
                 shrink=0.85, ticks=DISPLAY_TICKS_V, extend="max")
    fig.savefig(output/"exp_3_led_map.png", dpi=300)
    fig.savefig(output/"exp_3_led_map.pdf")
    plt.close(fig)


def plot_mesh(nodes, elements, materials, ground, active, folder):
    """Retain mesh illustrations, using collections instead of one artist per cell."""
    folder.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    material_colors = {1: "white", 2: "#cd7f32", 3: "silver"}
    ax.add_collection(PolyCollection(nodes[elements],
                      facecolors=[material_colors[int(m)] for m in materials],
                      edgecolors="black", linewidths=0.3, alpha=0.6))
    ax.set(xlim=(0, 0.4), ylim=(0, 0.4), xlabel="x (m)", ylabel="y (m)")
    ax.set_aspect("equal"); fig.tight_layout()
    fig.savefig(folder/"exp_3_mesh_calibration.png", dpi=250); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_collection(PolyCollection(nodes[elements], facecolors="white",
                                    edgecolors="gray", linewidths=0.2, alpha=0.5))
    natural = ((np.abs(nodes[:, 0]) < 1e-4) | (np.abs(nodes[:, 1]) < 1e-4)) & ~ground & ~active
    for selected, color, label in [(active, "red", "Terminal: 45 kV (assumed)"),
                                    (ground, "blue", "Ground: 0 V"),
                                    (natural, "green", "Zero natural flux")]:
        ax.scatter(*nodes[selected].T, c=color, s=8, label=label)
    ax.set(xlim=(-0.1, 5.1), ylim=(-0.1, 5.1), xlabel="x (m)", ylabel="y (m)")
    ax.set_aspect("equal"); ax.legend(loc="upper right", fontsize=8); fig.tight_layout()
    fig.savefig(folder/"exp_3_mesh_calibration_fronteras.png", dpi=250); plt.close(fig)


def export_tables(interpolator, amplitude, exponent, output):
    records = []
    for name, distance, assigned in zip(LED_NAMES, DISTANCE_CM, ASSIGNED_V):
        fem_v = float(interpolator(R_SPHERE_M+distance/100, Y_SPHERE_M))
        records.append({"LED_Color": name, "Measured_activation_distance_cm": float(distance),
                        "Assigned_LED_voltage_V": float(assigned), "FEM_magnitude_kV": fem_v/1000,
                        "LED_to_FEM_ratio": assigned/fem_v, "1e5_LED_to_FEM_ratio": 1e5*assigned/fem_v,
                        "Fitted_surrogate_at_observed_distance_V": float(surrogate_voltage(distance, amplitude, exponent)),
                        "Fitted_surface_distance_at_assigned_voltage_cm": float((amplitude/assigned)**(1/exponent))})
    with (output/"exp_3_led_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    # Both packages are already listed in the public repository requirements.
    import pandas as pd
    pd.DataFrame(records).to_excel(output/"exp_3_led_table.xlsx", index=False)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=BASE/"gmsh_meshes/exp_3_mesh_calibration.npz")
    parser.add_argument("--output-dir", type=Path, default=BASE/"results")
    parser.add_argument("--skip-mesh-plots", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    with np.load(args.mesh, allow_pickle=False) as data:
        nodes, elements, materials = data["nodes"], data["elements"], data["materials"]
    boundaries, ground, active = boundary_conditions(nodes)
    print("Cartesian FEM: {} nodes, {} Q4 elements".format(len(nodes), len(elements)), flush=True)
    solution = solve_prior(nodes, elements, materials, boundaries)
    if not np.all(np.isfinite(solution)):
        raise RuntimeError("Non-finite FEM solution.")
    amplitude, exponent, log_r2 = fit_calibration()
    interpolator, arrays = remap_prior(nodes, elements, np.abs(solution), amplitude, exponent)
    plot_map(arrays, output)
    if not args.skip_mesh_plots:
        plot_mesh(nodes, elements, materials, ground, active, output.parent/"mesh_images")
    records = export_tables(interpolator, amplitude, exponent, output)
    np.savez_compressed(output/"exp_3_fem_solution.npz", nodes=nodes, elements=elements,
                        materials=materials, U=solution, U_abs=np.abs(solution))
    np.savez_compressed(output/"exp_3_calibrated_grid.npz", **arrays,
                        amplitude_V=amplitude, exponent=exponent, log_R2=log_r2)
    metadata = {"calibration_pairs": 5, "amplitude_V": amplitude, "exponent": exponent,
                "log_space_R2": log_r2, "law": "V(d)=A*(d/1 cm)^(-gamma)",
                "distance_reference": "Sphere surface along y=0.1225 m for the FEM profile",
                "calibration_interval_cm": [5.0, 17.0],
                "outside_interval": "Extrapolated surrogate, not an independent observation",
                "measurement_provenance": "Manual LED activation distances; color-assigned forward voltages; 2 cm leg separation",
                "uncertainty": "No retained repeats, threshold tolerances or controlled operator/environmental records",
                "FEM_formulation": "Cartesian 2D, -div[(sigma+j*omega*epsilon)*grad(V)]=0; no cylindrical weighting",
                "excitation": {"amplitude_V": SOURCE_V, "frequency_Hz": FREQUENCY_HZ,
                               "basis": "45 kV approximate 3 MV/m times 1.5 cm breakdown estimate; frequency assumed"},
                "epsilon_F_per_m": EPSILON_F_M, "conductivity_S_per_m_by_material": CONDUCTIVITY_S_M,
                "boundary_conditions": "45 kV for x<0.002 m, 0.098<y<0.147 m; zero at x=5/y=5 m; other fluxes zero; no separate winding-base ground",
                "remapping": "FEM magnitude -> equivalent surface distance on 1000-point radial profile -> five-LED power law",
                "display_range_V": list(DISPLAY_RANGE_V), "colormap": "turbo, linear",
                "data_voltage_clipping": False, "marker_meaning": "Observed activation locations; contours indicate assigned voltages and need not pass through markers",
                "mesh_sha256": hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "nodes": len(nodes), "Q4_elements": len(elements), "LED_records": records}
    (output/"exp_3_calibration.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Five-LED fit: A={:.9f} V, gamma={:.9f}, log-space R2={:.9f}".format(amplitude, exponent, log_r2))
    print("Outputs: {}".format(output))


if __name__ == "__main__":
    main()
