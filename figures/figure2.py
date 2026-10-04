import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
import matplotlib.tri as mtri
import sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'fem'))
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
mallas_dir = str(BASE/"fem/gmsh_meshes")
out_dir = str(BASE/"runs/figures")
os.makedirs(mallas_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)
sigma_al = 3.5e7
sigma_cu = 5.8e7

def cargar_malla_exp2(name):
    path = os.path.join(mallas_dir, f"exp_2_mesh_{name}.npz")
    data = np.load(path)
    return data['nodes'], data['elements'], data['materials']

def solve_dipolo(nodes, elements, materials):
    print("Resolviendo Dipolo Eléctrico...")
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
    xg = np.linspace(-1.95, 1.95, 150); yg = np.linspace(-1.95, 1.95, 150)
    X, Y = np.meshgrid(xg, yg)
    triangles = []
    for el in elements: triangles.extend([[el[0], el[1], el[2]], [el[0], el[2], el[3]]])
    tri_mesh = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)
    interp = mtri.LinearTriInterpolator(tri_mesh, U)
    Z = interp(X, Y)
    Z = np.nan_to_num(Z, nan=0.0)
    dy, dx = np.gradient(Z, yg[1] - yg[0], xg[1] - xg[0])
    return (nodes, elements, materials, triangles, U, -dx, -dy, X, Y,
            "Electric Dipole", 'RdBu_r', "Potential V (V)", 'electric')

def solve_iman(nodes, elements, materials):
    print("Resolviendo Imán Horizontal...")
    for i, el in enumerate(elements):
        pts = nodes[el]
        xc, yc = np.mean(pts[:, 0]), np.mean(pts[:, 1])
        if abs(xc) < 0.601 and abs(yc) < 0.201:
            if yc >= 0.0: materials[i] = 2
            else: materials[i] = 3
    dir_borde = {}
    for i, (x, y) in enumerate(nodes):
        if abs(x) > 1.98 or abs(y) > 1.98: dir_borde[i] = 0.0
    prop_K = {1: 1.0, 2: 1.0, 3: 1.0}
    fuentes = {1: 0.0, 2: 500.0, 3: -500.0}
    A = resolver_sistema_maestro(nodes, elements, materials, prop_K, {}, {}, fuentes, dir_borde, 'static')
    xg = np.linspace(-1.95, 1.95, 150); yg = np.linspace(-1.95, 1.95, 150)
    X, Y = np.meshgrid(xg, yg)
    triangles = []
    for el in elements: triangles.extend([[el[0], el[1], el[2]], [el[0], el[2], el[3]]])
    tri_mesh = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)
    interp = mtri.LinearTriInterpolator(tri_mesh, A)
    Z = interp(X, Y)
    Z = np.nan_to_num(Z, nan=0.0)
    dy, dx = np.gradient(Z, yg[1] - yg[0], xg[1] - xg[0])
    return (nodes, elements, materials, triangles, A, dy, -dx, X, Y,
            "Permanent Magnet", 'viridis', "Potential Az (Wb/m)", 'magnetic')

def solve_cables(nodes, elements, materials):
    print("Resolviendo Cables de Corriente...")
    dir_borde = {}
    for i, (x, y) in enumerate(nodes):
        if abs(x) > 1.98 or abs(y) > 1.98: dir_borde[i] = 0.0
    prop_K = {1: 1.0, 2: 1.0, 3: 1.0}
    fuentes = {1: 0.0, 2: 800.0, 3: -800.0}
    A = resolver_sistema_maestro(nodes, elements, materials, prop_K, {}, {}, fuentes, dir_borde, 'static')
    xg = np.linspace(-1.95, 1.95, 150); yg = np.linspace(-1.95, 1.95, 150)
    X, Y = np.meshgrid(xg, yg)
    triangles = []
    for el in elements: triangles.extend([[el[0], el[1], el[2]], [el[0], el[2], el[3]]])
    tri_mesh = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)
    interp = mtri.LinearTriInterpolator(tri_mesh, A)
    Z = interp(X, Y)
    Z = np.nan_to_num(Z, nan=0.0)
    dy, dx = np.gradient(Z, yg[1] - yg[0], xg[1] - xg[0])
    return (nodes, elements, materials, triangles, A, dy, -dx, X, Y,
            "Parallel Wires", 'coolwarm', "Potential Az (Wb/m)", 'magnetic')

def solve_faraday(nodes, elements, materials):
    print("Resolviendo Ley de Faraday...")
    dir_borde = {}
    for i, (x, y) in enumerate(nodes):
        if abs(x) > 1.98 or abs(y) > 1.98: dir_borde[i] = 0.0
    w_ac = 2 * np.pi * 50.0
    mu0 = 4 * np.pi * 1e-7
    prop_K = {1: 1.0/mu0, 2: 1.0/mu0, 3: 1.0/mu0, 4: 1.0/mu0}
    prop_C = {1: 0.0, 2: 0.0, 3: 0.0, 4: sigma_al}
    prop_M = {k: 0.0 for k in [1, 2, 3, 4]}
    fuentes = {1: 0.0, 2: 2000.0, 3: -2000.0, 4: 0.0}
    A_complex = resolver_sistema_maestro(nodes, elements, materials, prop_K, prop_C, prop_M, fuentes, dir_borde, 'harmonic', kwargs={'omega': w_ac})
    xg = np.linspace(-1.95, 1.95, 150); yg = np.linspace(-1.95, 1.95, 150)
    X, Y = np.meshgrid(xg, yg)
    triangles = []
    for el in elements: triangles.extend([[el[0], el[1], el[2]], [el[0], el[2], el[3]]])
    tri_mesh = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)
    interp_A = mtri.LinearTriInterpolator(tri_mesh, np.real(A_complex))
    A_grid = interp_A(X, Y)
    A_grid = np.nan_to_num(A_grid, nan=0.0)
    dy, dx = np.gradient(A_grid, yg[1] - yg[0], xg[1] - xg[0])
    return (nodes, elements, materials, triangles, np.real(A_complex), dy, -dx, X, Y,
            "Faraday Induction", 'coolwarm', "Real Potential Az", 'magnetic')

if __name__ == "__main__":
    d1 = solve_dipolo(*cargar_malla_exp2("dipole"))
    d2 = solve_iman(*cargar_malla_exp2("magnet"))
    d3 = solve_cables(*cargar_malla_exp2("cables"))
    d4 = solve_faraday(*cargar_malla_exp2("faraday"))

    data_list = [d1, d2, d3, d4]

    # 1. Plot all potentials
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor('white')
    for ax, (nodes, elements, materials, triangles, U, Vx, Vy, X, Y, title, cmap, clabel, ptype) in zip(axs, data_list):
        ax.set_title(title, pad=12, fontweight='bold')
        cf = ax.tricontourf(nodes[:, 0], nodes[:, 1], triangles, U, levels=50, cmap=cmap)
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label=clabel)
        for i, el in enumerate(elements):
            if materials[i] > 1:
                pts = nodes[el]
                poly = plt.Polygon(pts, fill=False, edgecolor='black', linewidth=0.3, alpha=0.3)
                ax.add_patch(poly)
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_aspect('equal')
        ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "exp_2_potentials.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # 2. Plot all vector fields
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor('white')
    for ax, (nodes, elements, materials, triangles, U, Vx, Vy, X, Y, title, cmap, clabel, ptype) in zip(axs, data_list):
        ax.set_title(title, pad=12, fontweight='bold')
        ax.tricontourf(nodes[:, 0], nodes[:, 1], triangles, U, levels=20, cmap=cmap, alpha=0.15)
        if ptype == 'electric':
            ax.streamplot(X, Y, Vx, Vy, color='black', density=1.5, linewidth=1.0, arrowsize=1.2)
        elif ptype == 'magnetic':
            tri_mesh = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)
            interp = mtri.LinearTriInterpolator(tri_mesh, U)
            Z = interp(X, Y)
            ax.contour(X, Y, Z, levels=25, colors='black', linewidths=1.0)
        for i, el in enumerate(elements):
            if materials[i] > 1:
                pts = nodes[el]
                poly = plt.Polygon(pts, fill=False, edgecolor='black', linewidth=0.3, alpha=0.3)
                ax.add_patch(poly)
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_aspect('equal')
        ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "exp_2_vector_fields.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # Assemble the two paper rows. Source solution fields are recomputed above.
    from PIL import Image
    top=Image.open(os.path.join(out_dir,'exp_2_potentials.png')).convert('RGB')
    bottom=Image.open(os.path.join(out_dir,'exp_2_vector_fields.png')).convert('RGB')
    width=max(top.width,bottom.width)
    canvas=Image.new('RGB',(width,top.height+bottom.height),'white')
    canvas.paste(top,((width-top.width)//2,0));canvas.paste(bottom,((width-bottom.width)//2,top.height))
    canvas.save(os.path.join(out_dir,'exp_2_potentials_and_vector_fields.png'))
