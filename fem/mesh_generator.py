import numpy as np
import os
import gmsh
import matplotlib.pyplot as plt

# Path setup
base_dir = os.path.dirname(os.path.abspath(__file__))
meshs_dir = os.path.join(base_dir, "gmsh_meshes")
images_dir = os.path.join(base_dir, "mesh_images")
os.makedirs(meshs_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

def gmsh_to_npz(mesh_name):
    """Extract mesh from gmsh. Uses physical groups for material assignment if defined."""
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    nodes = node_coords.reshape(-1, 3)[:, :2]

    node_idx_map = {tag: idx for idx, tag in enumerate(node_tags)}
    pgs = gmsh.model.getPhysicalGroups(2)

    elements = []
    materials_list = []

    if pgs:
        for dim, pg_tag in pgs:
            entity_tags = gmsh.model.getEntitiesForPhysicalGroup(dim, pg_tag)
            for entity_tag in entity_tags:
                elem_types, elem_tags_list, elem_node_tags_list = gmsh.model.mesh.getElements(dim, entity_tag)
                for t, tags, n_tags in zip(elem_types, elem_tags_list, elem_node_tags_list):
                    if t == 3:  # 4-node quad
                        n_tags_reshaped = n_tags.reshape(-1, 4)
                        for el in n_tags_reshaped:
                            elements.append([node_idx_map[node_id] for node_id in el])
                            materials_list.append(pg_tag)
    else:
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)
        for t, tags, n_tags in zip(elem_types, elem_tags, elem_node_tags):
            if t == 3:
                n_tags_reshaped = n_tags.reshape(-1, 4)
                for el in n_tags_reshaped:
                    elements.append([node_idx_map[node_id] for node_id in el])
                    materials_list.append(1)

    elements = np.array(elements)
    materials = np.array(materials_list, dtype=int)

    out_path = os.path.join(meshs_dir, f"{mesh_name}.npz")
    np.savez(out_path, nodes=nodes, elements=elements, materials=materials)
    print(f"Mesh {mesh_name} saved at {out_path} ({len(nodes)} nodes, {len(elements)} elements)")
    return nodes, elements, materials

# ==============================================================================
# EXPERIMENT 1: Mesh generation for the plate with hole (Q4)
# ==============================================================================
def generar_mesh_placa_agujero_q4(L, H, R, nx, ny, estrategia):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("mesh")

    tol = 1e-4

    p1 = gmsh.model.geo.addPoint(0, H/2 - R, 0)
    p2 = gmsh.model.geo.addPoint(0, 0, 0)
    p3 = gmsh.model.geo.addPoint(L, 0, 0)
    p4 = gmsh.model.geo.addPoint(L, H, 0)
    p5 = gmsh.model.geo.addPoint(0, H, 0)
    p6 = gmsh.model.geo.addPoint(0, H/2 + R, 0)
    p_arc = gmsh.model.geo.addPoint(R, H/2, 0)
    p0 = gmsh.model.geo.addPoint(0, H/2, 0)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p5)
    l5 = gmsh.model.geo.addLine(p5, p6)
    c6a = gmsh.model.geo.addCircleArc(p6, p0, p_arc)
    c6b = gmsh.model.geo.addCircleArc(p_arc, p0, p1)

    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4, l5, c6a, c6b])
    s = gmsh.model.geo.addPlaneSurface([cl])

    gmsh.option.setNumber("Mesh.Algorithm", 8)

    if estrategia == "polar":
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min(L,H)/nx)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", min(L,H)/nx)

        for l in [l1, l5]: gmsh.model.geo.mesh.setTransfiniteCurve(l, (nx//2) + 1)
        for l in [c6a, c6b]: gmsh.model.geo.mesh.setTransfiniteCurve(l, nx + 1)
        for l in [l2, l4]: gmsh.model.geo.mesh.setTransfiniteCurve(l, int(nx*1.5) + 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(l3, nx + 1)

    elif estrategia == "multibloque":
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min(L,H)/(nx*0.6))
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", min(L,H)/(nx*0.6))
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)

    elif estrategia == "hibrida":
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min(L,H)/(nx*1.5))
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", min(L,H)/(nx*0.3))
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    nodes_raw = np.array(nodeCoords).reshape(-1, 3)[:, :2]
    tag2idx = {tag: i for i, tag in enumerate(nodeTags)}

    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim=2)
    elements_raw = []

    for eType, eTags, eNodeTags in zip(elemTypes, elemTags, elemNodeTags):
        if eType == 3:
            n_elems = len(eTags)
            eNodeTags = np.array(eNodeTags).reshape(n_elems, 4)
            for el_nodes in eNodeTags:
                p1_n, p2_n, p3_n, p4_n = [nodes_raw[tag2idx[t]] for t in el_nodes]
                area = (p2_n[0]-p1_n[0])*(p3_n[1]-p1_n[1]) - (p2_n[1]-p1_n[1])*(p3_n[0]-p1_n[0])
                if area < 0:
                    elements_raw.append([tag2idx[el_nodes[0]], tag2idx[el_nodes[3]], tag2idx[el_nodes[2]], tag2idx[el_nodes[1]]])
                else:
                    elements_raw.append([tag2idx[t] for t in el_nodes])

    active_nodes = set()
    for el in elements_raw:
        active_nodes.update(el)

    active_nodes_list = sorted(list(active_nodes))
    node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(active_nodes_list)}

    nodes = nodes_raw[active_nodes_list]
    elements = []
    for el in elements_raw:
        elements.append([node_map[n] for n in el])
    elements = np.array(elements)

    materials = np.ones(len(elements), dtype=int)

    bobina, esfera, izq_sup, inferior, superior, derecho = [], [], [], [], [], []
    for i, (x, y) in enumerate(nodes):
        r = np.sqrt(x**2 + (y - H/2)**2)

        if abs(r - R) < tol and x >= -tol:
            esfera.append(i)
        elif abs(x) < tol:
            if y <= H/2 - R + tol:
                bobina.append(i)
            elif y >= H/2 + R - tol:
                izq_sup.append(i)

        if abs(y - H) < tol: superior.append(i)
        if abs(x - L) < tol: derecho.append(i)
        if abs(y) < tol: inferior.append(i)

    boundaries = {
        'bobina': bobina,
        'esfera': esfera,
        'izq_sup': izq_sup,
        'superior': superior,
        'derecho': derecho,
        'inferior': inferior
    }

    gmsh.finalize()
    return nodes, elements, materials, boundaries

# ==============================================================================
# EXPERIMENT 2: Classical mesh generation
# ==============================================================================
def generar_mesh_dipole():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("dipole")
    box = gmsh.model.occ.addRectangle(-2.0, -2.0, 0.0, 4.0, 4.0)
    c1 = gmsh.model.occ.addDisk(-0.5, 0.0, 0.0, 0.2, 0.2)
    c2 = gmsh.model.occ.addDisk(0.5, 0.0, 0.0, 0.2, 0.2)
    gmsh.model.occ.cut([(2, box)], [(2, c1), (2, c2)])
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.08)
    surfaces = gmsh.model.getEntities(2)
    for s in surfaces:
        gmsh.model.mesh.setRecombine(2, s[1])
    gmsh.model.mesh.generate(2)
    gmsh_to_npz("exp_2_mesh_dipole")
    gmsh.finalize()

def generar_mesh_magnet():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("magnet")
    box = gmsh.model.occ.addRectangle(-2.0, -2.0, 0.0, 4.0, 4.0)
    magnet = gmsh.model.occ.addRectangle(-0.6, -0.2, 0.0, 1.2, 0.4)
    gmsh.model.occ.fragment([(2, box)], [(2, magnet)])
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.08)
    surfaces = gmsh.model.getEntities(2)
    for s in surfaces:
        gmsh.model.mesh.setRecombine(2, s[1])
    gmsh.model.mesh.generate(2)
    gmsh_to_npz("exp_2_mesh_magnet")
    gmsh.finalize()

def generar_mesh_cables():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("cables")
    box = gmsh.model.occ.addRectangle(-2.0, -2.0, 0.0, 4.0, 4.0)
    c1 = gmsh.model.occ.addDisk(-0.5, 0.0, 0.0, 0.1, 0.1)
    c2 = gmsh.model.occ.addDisk(0.5, 0.0, 0.0, 0.1, 0.1)
    gmsh.model.occ.fragment([(2, box)], [(2, c1), (2, c2)])
    gmsh.model.occ.synchronize()
    surfaces = gmsh.model.getEntities(2)
    air_tags, wire1_tags, wire2_tags = [], [], []
    for dim, tag in surfaces:
        cx, cy, _ = gmsh.model.occ.getCenterOfMass(dim, tag)
        if np.sqrt((cx + 0.5)**2 + cy**2) < 0.11:
            wire1_tags.append(tag)
        elif np.sqrt((cx - 0.5)**2 + cy**2) < 0.11:
            wire2_tags.append(tag)
        else:
            air_tags.append(tag)
    gmsh.model.addPhysicalGroup(2, air_tags, 1)
    gmsh.model.addPhysicalGroup(2, wire1_tags, 2)
    gmsh.model.addPhysicalGroup(2, wire2_tags, 3)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.06)
    for s in surfaces:
        gmsh.model.mesh.setRecombine(2, s[1])
    gmsh.model.mesh.generate(2)
    gmsh_to_npz("exp_2_mesh_cables")
    gmsh.finalize()

def generar_mesh_faraday():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("faraday")
    box = gmsh.model.occ.addRectangle(-2.0, -2.0, 0.0, 4.0, 4.0)
    c1 = gmsh.model.occ.addDisk(-0.6, 0.3, 0.0, 0.08, 0.08)
    c2 = gmsh.model.occ.addDisk(-0.6, -0.3, 0.0, 0.08, 0.08)
    # Enlarge the cylinder to radius 0.45 centered at 0.5 for better induction coupling and visibility
    c3 = gmsh.model.occ.addDisk(0.5, 0.0, 0.0, 0.45, 0.45)
    gmsh.model.occ.fragment([(2, box)], [(2, c1), (2, c2), (2, c3)])
    gmsh.model.occ.synchronize()
    surfaces = gmsh.model.getEntities(2)
    air_tags, w1_tags, w2_tags, cyl_tags = [], [], [], []
    for dim, tag in surfaces:
        cx, cy, _ = gmsh.model.occ.getCenterOfMass(dim, tag)
        if np.sqrt((cx + 0.6)**2 + (cy - 0.3)**2) < 0.09:
            w1_tags.append(tag)
        elif np.sqrt((cx + 0.6)**2 + (cy + 0.3)**2) < 0.09:
            w2_tags.append(tag)
        elif np.sqrt((cx - 0.5)**2 + cy**2) < 0.46:
            cyl_tags.append(tag)
        else:
            air_tags.append(tag)
    gmsh.model.addPhysicalGroup(2, air_tags, 1)
    gmsh.model.addPhysicalGroup(2, w1_tags, 2)
    gmsh.model.addPhysicalGroup(2, w2_tags, 3)
    gmsh.model.addPhysicalGroup(2, cyl_tags, 4)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.07)
    for s in surfaces:
        gmsh.model.mesh.setRecombine(2, s[1])
    gmsh.model.mesh.generate(2)
    gmsh_to_npz("exp_2_mesh_faraday")
    gmsh.finalize()

# ==============================================================================
# EXPERIMENT 3: Calibration mesh generation (Tesla coil)
# ==============================================================================
def generar_mesh_calibracion():
    # Model parameters for calibracion
    h_coil = 0.10; r_coil = 0.015; r_sphere = 0.0225
    y_sphere = h_coil + r_sphere
    L_domain = 5.0; H_domain = 5.0

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("mapa_calibrado")

    s_air = gmsh.model.occ.addRectangle(0, 0, 0, L_domain, H_domain)
    s_coil = gmsh.model.occ.addRectangle(0, 0, 0, r_coil, h_coil)
    s_sphere = gmsh.model.occ.addDisk(0, y_sphere, 0, r_sphere, r_sphere)

    gmsh.model.occ.fragment([(2, s_air)], [(2, s_coil), (2, s_sphere)])
    gmsh.model.occ.synchronize()

    surfaces = gmsh.model.getEntities(2)
    to_rm = [(d,t) for d,t in surfaces if gmsh.model.occ.getCenterOfMass(d,t)[0] < -1e-6]
    if to_rm:
        gmsh.model.occ.remove(to_rm, recursive=True)
        gmsh.model.occ.synchronize()

    surfaces = gmsh.model.getEntities(2)
    mat_coil, mat_sphere, mat_air = [], [], []
    for dim, tag in surfaces:
        cx, cy, _ = gmsh.model.occ.getCenterOfMass(dim, tag)
        d_sph = np.sqrt(cx**2 + (cy - y_sphere)**2)
        if d_sph < r_sphere - 1e-4:
            mat_sphere.append(tag)
        elif cx < r_coil + 1e-4 and -1e-4 < cy < h_coil + 1e-4 and d_sph > r_sphere:
            mat_coil.append(tag)
        else:
            mat_air.append(tag)

    gmsh.model.addPhysicalGroup(2, mat_air, 1)
    if mat_coil: gmsh.model.addPhysicalGroup(2, mat_coil, 2)
    if mat_sphere: gmsh.model.addPhysicalGroup(2, mat_sphere, 3)

    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.0005)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1.0)

    phys_surfs = mat_sphere + mat_coil
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "SurfacesList", phys_surfs)
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", 0.001)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", 1.0)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.05)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 3.0)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    gmsh.model.mesh.generate(2)

    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    nodes_raw = np.array(nodeCoords).reshape(-1, 3)[:, :2]
    tag2idx = {tag: i for i, tag in enumerate(nodeTags)}

    elements_raw, materials_list = [], []
    for dim, tag in surfaces:
        pgs = gmsh.model.getPhysicalGroupsForEntity(dim, tag)
        if not pgs: continue
        mat_id = pgs[0]
        for eType, _, eNT in zip(*gmsh.model.mesh.getElements(dim, tag)):
            if eType == 3: # Quadrilaterals
                nE = len(eNT) // 4
                eNT = np.array(eNT).reshape(nE, 4)
                for en in eNT:
                    c = [nodes_raw[tag2idx[t]] for t in en]
                    area = (c[1][0]-c[0][0])*(c[2][1]-c[0][1]) - (c[1][1]-c[0][1])*(c[2][0]-c[0][0])
                    if area < 0:
                        elements_raw.append([tag2idx[en[0]], tag2idx[en[3]], tag2idx[en[2]], tag2idx[en[1]]])
                    else:
                        elements_raw.append([tag2idx[t] for t in en])
                    materials_list.append(mat_id)

    active = sorted(set(n for el in elements_raw for n in el))
    nmap = {old: new for new, old in enumerate(active)}
    nodes = nodes_raw[active]
    elements = np.array([[nmap[n] for n in el] for el in elements_raw])
    materials = np.array(materials_list)

    out_path = os.path.join(meshs_dir, "exp_3_mesh_calibration.npz")
    np.savez(out_path, nodes=nodes, elements=elements, materials=materials)
    print(f"Mesh exp_3_mesh_calibration saved at {out_path} ({len(nodes)} nodes, {len(elements)} elements)")
    gmsh.finalize()


def graficar_mesh_exp1(nodes, elements, filename, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('white')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    for el in elements:
        pts = nodes[el]
        poly = plt.Polygon(pts, fill=False, edgecolor='black', linewidth=0.4)
        ax.add_patch(poly)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Mesh image saved at {filename}")


if __name__ == "__main__":
    print("=========================================================")
    print("CENTRALIZED MESH GENERATOR (GMSH)")
    print("=========================================================")

    # 1. Generar Exp 1
    L, H, R = 1.0, 1.0, 0.2
    nx, ny = 20, 20
    estrategias_exp1 = {
        'multibloque': 'exp_1_mesh_blossom',
        'polar': 'exp_1_mesh_transfinite',
        'hibrida': 'exp_1_mesh_subdivision'
    }
    for estr, name in estrategias_exp1.items():
        print(f"Generating Exp 1 mesh ({estr}) with Gmsh...")
        n, e, m, b = generar_mesh_placa_agujero_q4(L, H, R, nx, ny, estr)
        filename = os.path.join(meshs_dir, f"{name}.npz")
        np.savez(filename,
                 nodes=n,
                 elements=e,
                 materials=m,
                 bobina=b['bobina'],
                 esfera=b['esfera'],
                 izq_sup=b['izq_sup'],
                 superior=b['superior'],
                 derecho=b['derecho'],
                 inferior=b['inferior'])
        print(f"  Saved at {filename}. Nodes: {len(n)}, Elements: {len(e)}")

        # Graficar y guardar imagen de la mesh
        img_filename = os.path.join(images_dir, f"{name}.png")
        tit_label = name.replace("exp_1_mesh_", "").capitalize()
        graficar_mesh_exp1(n, e, img_filename, None)

    # 2. Generar Exp 2
    print("\nGenerating Exp 2 meshes...")
    generar_mesh_dipole()
    generar_mesh_magnet()
    generar_mesh_cables()
    generar_mesh_faraday()

    # 3. Generar Exp 3
    print("\nGenerating Exp 3 mesh...")
    generar_mesh_calibracion()

    print("\nAll meshes successfully generated in gmsh_meshes/!")
    print("=========================================================")
