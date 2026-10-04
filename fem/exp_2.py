import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
import matplotlib.tri as mtri

from solver_fem_2d import resolver_sistema_maestro

plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'figure.titlesize': 18,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

base_dir = os.path.dirname(os.path.abspath(__file__))
meshs_dir = os.path.join(base_dir, "gmsh_meshes")
out_dir = os.path.join(base_dir, "results")
os.makedirs(meshs_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)
sigma_al = 3.5e7
sigma_cu = 5.8e7

def cargar_mesh_exp2(name):
    path = os.path.join(meshs_dir, f"exp_2_mesh_{name}.npz")
    data = np.load(path)
    return data['nodes'], data['elements'], data['materials']

def plot_1r3c(nodes, elements, materials, U, Ex, Ey, X_g, Y_g, title_main, filename, cmap, u_label, vec_label, plot_type='electric'):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor('white')

    # 1. Mesh
    ax = axs[0]
    ax.set_title("Mesh and Physical Domains", pad=12, fontweight='bold')

    # Define a color map for materials
    mat_colors = {1: 'white', 2: '#ffcccc', 3: '#ccccff', 4: '#ccffcc'}

    for i, el in enumerate(elements):
        mat = materials[i]
        color = mat_colors.get(mat, 'lightgray')
        pts = nodes[el]
        poly = plt.Polygon(pts, fill=True, facecolor=color, edgecolor='black', linewidth=0.2, alpha=0.7)
        ax.add_patch(poly)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    # 2. Potential Map
    ax = axs[1]
    ax.set_title("Potential / Scalar Excitation", pad=12, fontweight='bold')
    triangles = []
    for el in elements:
        triangles.extend([[el[0], el[1], el[2]], [el[0], el[2], el[3]]])
    cf = ax.tricontourf(nodes[:, 0], nodes[:, 1], triangles, U, levels=50, cmap=cmap)
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label=u_label)

    # Draw material outlines in subplot 2 (including the big conductor)
    for i, el in enumerate(elements):
        if materials[i] > 1:
            pts = nodes[el]
            poly = plt.Polygon(pts, fill=False, edgecolor='black', linewidth=0.3, alpha=0.3)
            ax.add_patch(poly)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_axis_off()

    # 3. Field Streamlines
    ax = axs[2]
    ax.set_title("Vector Field Lines", pad=12, fontweight='bold')

    # Plot potential background lightly
    ax.tricontourf(nodes[:, 0], nodes[:, 1], triangles, U, levels=20, cmap=cmap, alpha=0.15)

    if plot_type == 'electric':
        # Electric field lines are orthogonal to potential, use streamplot
        strm = ax.streamplot(X_g, Y_g, Ex, Ey, color='black', density=1.5, linewidth=1.0, arrowsize=1.2)
    elif plot_type == 'magnetic':
        # Magnetic field lines in 2D ARE the contour lines of Az
        # This guarantees perfect visualization of field lines without streamplot artifacts
        interp = mtri.LinearTriInterpolator(mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles), U)
        Z = interp(X_g, Y_g)
        ax.contour(X_g, Y_g, Z, levels=25, colors='black', linewidths=1.0)

    # Draw material outlines in subplot 3
    for i, el in enumerate(elements):
        if materials[i] > 1:
            pts = nodes[el]
            poly = plt.Polygon(pts, fill=False, edgecolor='black', linewidth=0.3, alpha=0.3)
            ax.add_patch(poly)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_axis_off()

    fig.suptitle(title_main, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Result saved at {filename}")

def solve_dipole(nodes, elements, materials):
    print("Solving Electric Dipole...")
    dir_borde = {}
    for i, (x, y) in enumerate(nodes):
        if abs(x) > 1.98 or abs(y) > 1.98:
            dir_borde[i] = 0.0
        elif np.sqrt((x + 0.5)**2 + y**2) < 0.21:
            dir_borde[i] = 10.0
        elif np.sqrt((x - 0.5)**2 + y**2) < 0.21:
            dir_borde[i] = -10.0

    prop_K = {1: 1.0}
    U = resolver_sistema_maestro(nodes, elements, materials, prop_K, {}, {}, {}, dir_borde, 'static')

    # Interpolate for streamlines
    xg = np.linspace(-1.95, 1.95, 150)
    yg = np.linspace(-1.95, 1.95, 150)
    X, Y = np.meshgrid(xg, yg)

    triangles = []
    for el in elements:
        triangles.extend([[el[0], el[1], el[2]], [el[0], el[2], el[3]]])
    tri_mesh = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)
    interp = mtri.LinearTriInterpolator(tri_mesh, U)

    Z = interp(X, Y)
    Z = np.nan_to_num(Z, nan=0.0)
    dy, dx = np.gradient(Z, yg[1] - yg[0], xg[1] - xg[0])

    Ex = -dx
    Ey = -dy

    # Vector field inside conductor is naturally 0 due to constant potential.
    # No need to manually mask it, let streamplot handle it naturally.

    plot_1r3c(nodes, elements, materials, U, Ex, Ey, X, Y,
              "Electric Dipole: Field of two Conducting Spheres (+10V / -10V)",
              os.path.join(out_dir, "exp_2_electrostatics_dipole.png"),
              'RdBu_r', "Potential V (V)", "Electric Field E", plot_type='electric')

def solve_magnet(nodes, elements, materials):
    print("Solving Horizontal Magnet...")
    # Magnet coordinates: x in [-0.6, 0.6], y in [-0.2, 0.2]
    # Re-assign material IDs:
    # Material 2: Top half of magnet (y >= 0) -> Jsz = +500
    # Material 3: Bottom half of magnet (y < 0) -> Jsz = -500
    for i, el in enumerate(elements):
        pts = nodes[el]
        xc, yc = np.mean(pts[:, 0]), np.mean(pts[:, 1])
        if abs(xc) < 0.601 and abs(yc) < 0.201:
            if yc >= 0.0:
                materials[i] = 2
            else:
                materials[i] = 3

    dir_borde = {}
    for i, (x, y) in enumerate(nodes):
        if abs(x) > 1.98 or abs(y) > 1.98:
            dir_borde[i] = 0.0

    prop_K = {1: 1.0, 2: 1.0, 3: 1.0}
    fuentes = {1: 0.0, 2: 500.0, 3: -500.0}

    A = resolver_sistema_maestro(nodes, elements, materials, prop_K, {}, {}, fuentes, dir_borde, 'static')

    xg = np.linspace(-1.95, 1.95, 150)
    yg = np.linspace(-1.95, 1.95, 150)
    X, Y = np.meshgrid(xg, yg)

    triangles = []
    for el in elements:
        triangles.extend([[el[0], el[1], el[2]], [el[0], el[2], el[3]]])
    tri_mesh = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)
    interp = mtri.LinearTriInterpolator(tri_mesh, A)

    Z = interp(X, Y)
    Z = np.nan_to_num(Z, nan=0.0)
    dy, dx = np.gradient(Z, yg[1] - yg[0], xg[1] - xg[0])

    Bx = dy
    By = -dx

    plot_1r3c(nodes, elements, materials, A, Bx, By, X, Y,
              "Horizontal Permanent Magnet: Magnetic Field induced by Equivalent Magnetization",
              os.path.join(out_dir, "exp_2_magnetostatics_magnet.png"),
              'viridis', "Potential Az (Wb/m)", "Magnetic Induction B", plot_type='magnetic')

def solve_cables(nodes, elements, materials):
    print("Solving Current Cables...")
    dir_borde = {}
    for i, (x, y) in enumerate(nodes):
        if abs(x) > 1.98 or abs(y) > 1.98:
            dir_borde[i] = 0.0

    # Materials already assigned at mesh time via physical groups

    prop_K = {1: 1.0, 2: 1.0, 3: 1.0}
    fuentes = {1: 0.0, 2: 800.0, 3: -800.0}

    A = resolver_sistema_maestro(nodes, elements, materials, prop_K, {}, {}, fuentes, dir_borde, 'static')

    xg = np.linspace(-1.95, 1.95, 150)
    yg = np.linspace(-1.95, 1.95, 150)
    X, Y = np.meshgrid(xg, yg)

    triangles = []
    for el in elements:
        triangles.extend([[el[0], el[1], el[2]], [el[0], el[2], el[3]]])
    tri_mesh = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)
    interp = mtri.LinearTriInterpolator(tri_mesh, A)

    Z = interp(X, Y)
    Z = np.nan_to_num(Z, nan=0.0)
    dy, dx = np.gradient(Z, yg[1] - yg[0], xg[1] - xg[0])

    Bx = dy
    By = -dx

    # Magnetic field exists inside the conductor, no masking needed.

    plot_1r3c(nodes, elements, materials, A, Bx, By, X, Y,
              "Parallel Cables: Magnetic Field of two Current Conductors (+I / -I)",
              os.path.join(out_dir, "exp_2_magnetostatics_cables.png"),
              'coolwarm', "Potential Az (Wb/m)", "Magnetic Induction B", plot_type='magnetic')

def solve_faraday(nodes, elements, materials):
    print("Solving Faraday's Law (Eddy Currents Induction)...")
    # Materials already assigned at mesh time via physical groups:
    # 1=Air, 2=Wire+I, 3=Wire-I, 4=Aluminum cylinder
    dir_borde = {}
    for i, (x, y) in enumerate(nodes):
        if abs(x) > 1.98 or abs(y) > 1.98:
            dir_borde[i] = 0.0

    # AC parameters (f = 50Hz, omega = 314.16)
    w_ac = 2 * np.pi * 50.0
    # Permeability of free space (for proper scaling: ∇·(1/μ)∇Az + jωσAz = -Js)
    mu0 = 4 * np.pi * 1e-7
    prop_K = {1: 1.0/mu0, 2: 1.0/mu0, 3: 1.0/mu0, 4: 1.0/mu0}
    # Only the aluminum cylinder has eddy-current conductivity.
    # Source wires carry imposed J via fuentes, not via eddy currents.
    prop_C = {
        1: 0.0,
        2: 0.0,
        3: 0.0,
        4: sigma_al
    }
    prop_M = {k: 0.0 for k in [1, 2, 3, 4]}
    # Current density in the source wires (A/m²)
    fuentes = {1: 0.0, 2: 2000.0, 3: -2000.0, 4: 0.0}

    # Solve Harmonic AC: (K + jωC)Az = F
    A_complex = resolver_sistema_maestro(nodes, elements, materials, prop_K, prop_C, prop_M, fuentes, dir_borde, 'harmonic', kwargs={'omega': w_ac})

    # We plot the amplitude of the total current density:
    # J = fuentes + j*omega*sigma * A
    J_tot = np.zeros(len(nodes), dtype=float)
    for i, mat in enumerate(materials):
        nodes_el = elements[i]
        if mat == 2:
            J_tot[nodes_el] = 2000.0
        elif mat == 3:
            J_tot[nodes_el] = 2000.0 # amplitude
        elif mat == 4:
            # induced current density: J = w * sigma * |A|
            J_tot[nodes_el] = w_ac * sigma_al * np.abs(A_complex[nodes_el])

    # Smooth a bit or just plot
    # Interpolate
    xg = np.linspace(-1.95, 1.95, 150)
    yg = np.linspace(-1.95, 1.95, 150)
    X, Y = np.meshgrid(xg, yg)

    triangles = []
    for el in elements:
        triangles.extend([[el[0], el[1], el[2]], [el[0], el[2], el[3]]])
    tri_mesh = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)

    interp_J = mtri.LinearTriInterpolator(tri_mesh, J_tot)
    J_grid = interp_J(X, Y)
    J_grid = np.nan_to_num(J_grid, nan=0.0)

    # Magnetic Field lines from real part of Az at phi=0
    interp_A = mtri.LinearTriInterpolator(tri_mesh, np.real(A_complex))
    A_grid = interp_A(X, Y)
    A_grid = np.nan_to_num(A_grid, nan=0.0)

    dy, dx = np.gradient(A_grid, yg[1] - yg[0], xg[1] - xg[0])
    Bx = dy
    By = -dx

    # Magnetic field exists inside sources, no masking needed.

    # Static plot at phase 0
    plot_1r3c(nodes, elements, materials, np.real(A_complex), Bx, By, X, Y,
              "Faraday's Law: Harmonic Induction Field and Interaction with Secondary Cylinder",
              os.path.join(out_dir, "exp_2_faraday_induction.png"),
              'coolwarm', "Real Potential Az (Wb/m)", "Magnetic Induction Lines B", plot_type='magnetic')

if __name__ == "__main__":
    nodes_d, elements_d, materials_d = cargar_mesh_exp2("dipole")
    nodes_i, elements_i, materials_i = cargar_mesh_exp2("magnet")
    nodes_c, elements_c, materials_c = cargar_mesh_exp2("cables")
    nodes_f, elements_f, materials_f = cargar_mesh_exp2("faraday")

    solve_dipole(nodes_d, elements_d, materials_d)
    solve_magnet(nodes_i, elements_i, materials_i)
    solve_cables(nodes_c, elements_c, materials_c)
    solve_faraday(nodes_f, elements_f, materials_f)

    print("Experiment 2 completed successfully!")
