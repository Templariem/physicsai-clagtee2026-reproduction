import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
import matplotlib.tri as mtri
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time

from solver_fem_2d import resolver_sistema_maestro, ensamblar_sistema_maestro

plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'figure.titlesize': 18,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': '#aaaaaa'
})

def cargar_mesh(filename):
    data = np.load(filename)
    boundaries = {
        'bobina': list(data['bobina']),
        'esfera': list(data['esfera']),
        'izq_sup': list(data['izq_sup']),
        'superior': list(data['superior']),
        'derecho': list(data['derecho']),
        'inferior': list(data['inferior'])
    }
    return data['nodes'], data['elements'], data['materials'], boundaries

def generar_triangulaciones(e):
    triangles = []
    for el in e:
        triangles.extend([[el[0], el[1], el[2]], [el[0], el[2], el[3]]])
    return np.array(triangles)


# ==============================================================================
# SECCIÓN 1: RESOLUCIÓN NUMÉRICA Y VISUALIZACIÓN FEM
# ==============================================================================


# ==============================================================================
# SECCIÓN 2: VALIDACIÓN ANALÍTICA Y CONVERGENCIA L2
# ==============================================================================

def generar_figures_completas():
    estrategias = ['subdivision', 'blossom', 'transfinite']
    nombres_estr = ['Subdivision', 'Blossom', 'Transfinite']
    base_dir = os.path.dirname(os.path.abspath(__file__))
    meshs_dir = os.path.join(base_dir, 'gmsh_meshes')
    out_dir = os.path.join(base_dir, 'results')
    os.makedirs(out_dir, exist_ok=True)

    # 1. Cargar meshs
    meshs = {}
    for estr in estrategias:
        meshs[estr] = cargar_mesh(os.path.join(meshs_dir, f"exp_1_mesh_{estr}.npz"))

    # Estructura para guardar soluciones
    # soluciones[estr] = { 'es': (fem, exact, err), 'ms': ..., 'ac': ..., 'td': ... }
    soluciones = {}

    # Línea de corte 1D en y = 0.5, de x = 0.201 a 0.999
    x_line = np.linspace(0.201, 0.999, 100)
    y_line = np.full_like(x_line, 0.5)

    for estr in estrategias:
        n, e, m, b = meshs[estr]

        # --- ELECTROSTÁTICO (ES) ---
        dir_borde_es = {}
        for b_name, node_list in b.items():
            for nd in node_list:
                x_n, y_n = n[nd, 0], n[nd, 1]
                r_n = np.sqrt(x_n**2 + (y_n - 0.5)**2)
                r_n = max(r_n, 0.2)
                dir_borde_es[nd] = 1000.0 * np.log(5.0 / r_n) / np.log(5.0 / 0.2)

        U_fem_es = resolver_sistema_maestro(n, e, m, {1: 1.0}, {}, {}, {1: 0.0}, dir_borde_es, 'static')
        r_all = np.sqrt(n[:, 0]**2 + (n[:, 1] - 0.5)**2)
        U_exact_es = 1000.0 * np.log(5.0 / r_all) / np.log(5.0 / 0.2)
        err_es = np.abs(U_fem_es - U_exact_es)

        # --- MAGNETOSTÁTICO (MS) ---
        dir_borde_ms = {}
        for b_name, node_list in b.items():
            for nd in node_list:
                x_n, y_n = n[nd, 0], n[nd, 1]
                dir_borde_ms[nd] = x_n**2 - y_n**2

        U_fem_ms = resolver_sistema_maestro(n, e, m, {1: 1.0}, {}, {}, {1: 0.0}, dir_borde_ms, 'static')
        U_exact_ms = n[:, 0]**2 - n[:, 1]**2
        err_ms = np.abs(U_fem_ms - U_exact_ms)

        # --- ARMÓNICO (AC) ---
        omega_ac = 1.0
        sigma_ac = 2.0
        k_val = 1.0 + 1j

        dir_borde_ac = {}
        for b_name, node_list in b.items():
            for nd in node_list:
                x_n, y_n = n[nd, 0], n[nd, 1]
                dir_borde_ac[nd] = np.exp(-k_val * x_n)

        U_fem_ac = resolver_sistema_maestro(n, e, m, {1: 1.0}, {1: sigma_ac}, {}, {1: 0.0}, dir_borde_ac, 'harmonic', kwargs={'omega': omega_ac})
        # Trabajamos con la parte real del potencial fasorial complejo
        U_exact_ac = np.exp(-k_val * n[:, 0])
        err_ac = np.abs(U_fem_ac - U_exact_ac)

        # --- TRANSITORIO (TD) ---
        omega_td = 5.0
        dt = 0.02
        num_pasos = 40
        t_final = dt * num_pasos # 0.8s

        def f_wave(tau):
            return np.where(tau <= 0.0, 0.0, np.sin(omega_td * tau)**3)

        dir_borde_td = {}
        for b_name, node_list in b.items():
            for nd in node_list:
                x_n, y_n = n[nd, 0], n[nd, 1]
                dir_borde_td[nd] = lambda t, x_val=x_n: float(f_wave(t - x_val))

        U_hist = resolver_sistema_maestro(n, e, m, {1: 1.0}, {}, {1: 1.0}, {1: 0.0}, dir_borde_td, 'transient', kwargs={'dt': dt, 'num_pasos': num_pasos})
        U_fem_td = U_hist[-1]
        U_exact_td = f_wave(t_final - n[:, 0])
        err_td = np.abs(U_fem_td - U_exact_td)

        soluciones[estr] = {
            'es': (U_fem_es, U_exact_es, err_es),
            'ms': (U_fem_ms, U_exact_ms, err_ms),
            'ac': (np.real(U_fem_ac), np.real(U_exact_ac), err_ac), # guardamos la parte real para graficar
            'td': (U_fem_td, U_exact_td, err_td)
        }

    # =========================================================================
    # FIGURA 1: 2x2 Comparación de Casos Analyticals (Campos Exactos en 2D)
    # Usamos la mesh estructurada Polar (transfinite) por ser la más fina
    # =========================================================================
    print("Generating Figure 1: Analytical Cases Comparison (2D)...")
    fig1, axs1 = plt.subplots(2, 2, figsize=(12, 11))
    fig1.patch.set_facecolor('white')

    n_p, e_p = meshs['transfinite'][0], meshs['transfinite'][1]
    tri_p = generar_triangulaciones(e_p)
    sol_p = soluciones['transfinite']

    # (0,0) Electrostático Analytical
    cf_es = axs1[0,0].tricontourf(n_p[:,0], n_p[:,1], tri_p, sol_p['es'][1], levels=50, cmap='magma', antialiased=True)
    axs1[0,0].set_title("Electrostatic Analytical $V_{\\mathrm{analit}}(r)$ (V)", fontsize=11, fontweight='bold')
    axs1[0,0].set_aspect('equal'); axs1[0,0].axis('off')
    fig1.colorbar(cf_es, ax=axs1[0,0], fraction=0.046, pad=0.04)

    # (0,1) Magnetostático Analytical
    cf_ms = axs1[0,1].tricontourf(n_p[:,0], n_p[:,1], tri_p, sol_p['ms'][1], levels=50, cmap='viridis', antialiased=True)
    axs1[0,1].set_title("Magnetostatic Analytical $A_{z,\\mathrm{analit}}(x,y)$ (Wb/m)", fontsize=11, fontweight='bold')
    axs1[0,1].set_aspect('equal'); axs1[0,1].axis('off')
    fig1.colorbar(cf_ms, ax=axs1[0,1], fraction=0.046, pad=0.04)

    # (1,0) Armónico Analytical (Parte Real)
    cf_ac = axs1[1,0].tricontourf(n_p[:,0], n_p[:,1], tri_p, sol_p['ac'][1], levels=50, cmap='inferno', antialiased=True)
    axs1[1,0].set_title("Harmonic AC Analytical $\\mathrm{Re}(A_{z,\\mathrm{analit}})$ (Wb/m)", fontsize=11, fontweight='bold')
    axs1[1,0].set_aspect('equal'); axs1[1,0].axis('off')
    fig1.colorbar(cf_ac, ax=axs1[1,0], fraction=0.046, pad=0.04)

    # (1,1) Transitorio Analytical (t = 0.8s)
    cf_td = axs1[1,1].tricontourf(n_p[:,0], n_p[:,1], tri_p, sol_p['td'][1], levels=50, cmap='twilight_shifted', antialiased=True)
    axs1[1,1].set_title("Transient TD Analytical $u_{\\mathrm{analit}}(x, t=0.8\\mathrm{s})$ (V)", fontsize=11, fontweight='bold')
    axs1[1,1].set_aspect('equal'); axs1[1,1].axis('off')
    fig1.colorbar(cf_td, ax=axs1[1,1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig1_path = os.path.join(out_dir, "exp_1_analytical_cases_2d.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved at {fig1_path}")

    # =========================================================================
    # FIGURA 2: 2x2 Comparación de Precisión de Meshs (Líneas 1D)
    # =========================================================================
    print("Generating Figure 2: 1D Precision Comparison (Meshes vs Exact)...")
    fig2, axs2 = plt.subplots(2, 2, figsize=(13, 10))
    fig2.patch.set_facecolor('white')

    colors = ['#d62728', '#1f77b4', '#2ca02c']
    markers = ['o', 's', '^']

    # Definir soluciones analíticas en la línea de corte y = 0.5
    y_cut_es = 1000.0 * np.log(5.0 / x_line) / np.log(5.0 / 0.2)
    y_cut_ms = x_line**2 - 0.5**2
    y_cut_ac = np.real(np.exp(-(1.0 + 1j) * x_line))
    y_cut_td = np.where((0.8 - x_line) <= 0.0, 0.0, np.sin(5.0 * (0.8 - x_line))**3)

    regimes = [
        ('es', y_cut_es, "Electrostatic: Electric Potential $V(x)$ (V)", axs2[0,0]),
        ('ms', y_cut_ms, "Magnetostatic: Magnetic Potential $A_z(x)$ (Wb/m)", axs2[0,1]),
        ('ac', y_cut_ac, "Harmonic AC: Magnetic Potential $\\mathrm{Re}(A_z(x))$ (Wb/m)", axs2[1,0]),
        ('td', y_cut_td, "Transient TD: Electric Potential $u(x, t=0.8\\mathrm{s})$ (V)", axs2[1,1])
    ]

    for reg_key, exact_line, title, ax in regimes:
        # Trazar analítico
        ax.plot(x_line, exact_line, label='Analytical', color='black', lw=1.5, zorder=1)

        # Trazar cada mesh interpolada
        for m_idx, estr in enumerate(estrategias):
            n, e, m, b = meshs[estr]
            fem_data = soluciones[estr][reg_key][0] # potencial FEM

            triangles = generar_triangulaciones(e)
            tri_mesh = mtri.Triangulation(n[:, 0], n[:, 1], triangles)
            interp = mtri.LinearTriInterpolator(tri_mesh, fem_data)
            fem_line = np.array([float(interp(px, py)) for px, py in zip(x_line, y_line)])

            ax.plot(x_line, fem_line, label=f'FEM: {nombres_estr[m_idx]}', color=colors[m_idx],
                    linestyle='None', marker=markers[m_idx], markersize=4, markevery=5, zorder=2)

        ax.set_xlabel('Radial position $x$ (m)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2_path = os.path.join(out_dir, "exp_1_precision_meshes_1d.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved at {fig2_path}")

    # =========================================================================
    # FIGURA 3: 4x3 Contraste de Error Absoluto 2D (4 Casos × 3 Meshs)
    # =========================================================================
    print("Generating Figure 3: 2D Absolute Error Contrast (4x3)...")
    fig3, axs3 = plt.subplots(4, 3, figsize=(15, 18))
    fig3.patch.set_facecolor('white')

    # Definición de filas (regímenes) y columnas (meshs)
    filas_reg = [
        ('es', "Electrostatic"),
        ('ms', "Magnetostatic"),
        ('ac', "Harmonic AC"),
        ('td', "Transient TD")
    ]

    for r_idx, (reg_key, reg_name) in enumerate(filas_reg):
        for c_idx, estr in enumerate(estrategias):
            n, e, m, b = meshs[estr]
            triangles = generar_triangulaciones(e)

            # Obtener el error absoluto local
            err_data = soluciones[estr][reg_key][2]
            max_err_val = np.max(err_data)

            # Establecer niveles y colorbar por fila para homogeneizar escalas de visualización
            levels_err = np.linspace(0, max_err_val, 50) if max_err_val > 1e-6 else 50
            cf = axs3[r_idx, c_idx].tricontourf(n[:, 0], n[:, 1], triangles, err_data, levels=levels_err, cmap='viridis', antialiased=True)

            axs3[r_idx, c_idx].set_aspect('equal')
            axs3[r_idx, c_idx].axis('off')

            # Títulos de columnas (sólo en la primera fila)
            if r_idx == 0:
                axs3[r_idx, c_idx].set_title(f"Mesh {nombres_estr[c_idx]}\n", fontsize=12, fontweight='bold')

            # Nombre del régimen a la izquierda (sólo en la primera columna)
            if c_idx == 0:
                axs3[r_idx, c_idx].text(-0.25, 0.5, reg_name, transform=axs3[r_idx, c_idx].transAxes,
                                       fontsize=12, fontweight='bold', rotation=90, va='center', ha='center')

            # Mostrar el error máximo local en cada panel
            axs3[r_idx, c_idx].text(0.95, 0.05, f"Max Err: {max_err_val:.2e}", transform=axs3[r_idx, c_idx].transAxes,
                                   fontsize=10, color='white', fontweight='bold',
                                   bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3'),
                                   ha='right', va='bottom')

            fig3.colorbar(cf, ax=axs3[r_idx, c_idx], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig3_path = os.path.join(out_dir, "exp_1_comparative_errors_2d.png")
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved at {fig3_path}")

    # Escribir tabla de convergencia CSV y Excel
    csv_path = os.path.join(out_dir, "exp_1_convergence_table.csv")
    xlsx_path = os.path.join(out_dir, "exp_1_convergence_table.xlsx")

    rows_convergencia = []
    for estr in estrategias:
        n, e, m, b = meshs[estr]
        U_fem_es, U_exact_es, err_es = soluciones[estr]['es']
        U_fem_ms, U_exact_ms, err_ms = soluciones[estr]['ms']
        U_fem_ac_real, U_exact_ac_real, err_ac = soluciones[estr]['ac']
        omega_ac = 1.0
        sigma_ac = 2.0
        k_val = 1.0 + 1j
        U_exact_ac_complex = np.exp(-k_val * n[:, 0])
        U_fem_td, U_exact_td, err_td = soluciones[estr]['td']

        l2_es = np.sqrt(np.sum(err_es**2) / np.sum(U_exact_es**2))
        l2_ms = np.sqrt(np.sum(err_ms**2) / np.sum(U_exact_ms**2))
        l2_ac = np.sqrt(np.sum(err_ac**2) / np.sum(np.abs(U_exact_ac_complex)**2))
        l2_td = np.sqrt(np.sum(err_td**2) / np.sum(U_exact_td**2))

        rows_convergencia.append({
            'Mesh': estr,
            'Nodos': len(n),
            'Elementos': len(e),
            'L2_Electrostatic': l2_es,
            'L2_Magnetostatic': l2_ms,
            'L2_Harmonic': l2_ac,
            'L2_Transient': l2_td
        })

    import pandas as pd
    df_conv = pd.DataFrame(rows_convergencia)
    df_conv.to_csv(csv_path, index=False)
    df_conv.to_excel(xlsx_path, index=False)
    print(f"Convergence table successfully saved at {csv_path} y {xlsx_path}")
    print("All figures and tables successfully generated!")

def f_wave(tau, omega=5.0):
    return np.where(tau <= 0.0, 0.0, np.sin(omega * tau)**3)



# ==============================================================================
# SECCIÓN 3: ANÁLISIS DE RENDIMIENTO COMPUTACIONAL
# ==============================================================================

def estimar_condicion_k(K_global, dir_borde):
    n_nodes = K_global.shape[0]
    boundary_nodes = set(dir_borde.keys())
    free_nodes = [i for i in range(n_nodes) if i not in boundary_nodes]

    # Extraer submatriz libre K_ff
    K_ff = K_global[free_nodes, :][:, free_nodes].tocsc()

    # Calcular norma 1 de K_ff
    norm_K = spla.norm(K_ff, 1)

    # Factorizar K_ff para resolver rápido en onenormest
    try:
        solve_factor = spla.factorized(K_ff)
        def solve_K(x):
            return solve_factor(x)
        K_inv_op = spla.LinearOperator((K_ff.shape[0], K_ff.shape[0]), matvec=solve_K, rmatvec=solve_K)
        norm_K_inv = spla.onenormest(K_inv_op)
        cond_K = norm_K * norm_K_inv
    except Exception as e:
        import traceback
        traceback.print_exc()
        cond_K = np.nan

    return cond_K

def analizar_rendimiento():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    meshs_dir = os.path.join(base_dir, "gmsh_meshes")
    out_dir = os.path.join(base_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    estrategias = ['blossom', 'transfinite', 'subdivision']
    titulos = ["Blossom", "Transfinite", "Subdivision"]

    csv_rows = []
    csv_rows.append("Mesh Strategy,Nodes,Elements,Assembly Time (s),Electrostatic Solve Time (s),Electrostatic Matrix Condition Number,Magnetostatic Solve Time (s),Magnetostatic Matrix Condition Number,Harmonic Solve Time (s),Transient Solve Time (40 steps) (s)")

    print("Starting performance and stability analysis...")

    for i, estr in enumerate(estrategias):
        mesh_path = os.path.join(meshs_dir, f"exp_1_mesh_{estr}.npz")
        nodes, elements, materials, boundaries = cargar_mesh(mesh_path)

        n_nodes = len(nodes)
        n_elements = len(elements)
        print(f"\nProcessing Mesh: {titulos[i]} ({n_nodes} nodes, {n_elements} elements)")

        # 1. Medir tiempo de ensamblaje
        prop_K = {1: 1.0}; prop_C = {1: 0.1}; prop_M = {1: 0.05}
        fuentes = {1: 10.0}

        t0 = time.time()
        K_global, C_global, M_global, F_base = ensamblar_sistema_maestro(
            nodes, elements, materials, prop_K, prop_C, prop_M, fuentes, 'transient'
        )
        t_assembly = time.time() - t0
        print(f"  Assembly time: {t_assembly:.4f} s")

        # 2. Régimen Electrostático
        dir_borde_est = {nodo: 1000.0 for nodo in boundaries['bobina'] + boundaries['esfera']}
        dir_borde_est.update({nodo: 0.0 for nodo in boundaries['superior'] + boundaries['derecho']})

        # Estimar condición de K_ff en electrostática
        cond_est = estimar_condicion_k(K_global, dir_borde_est)

        t0 = time.time()
        _ = resolver_sistema_maestro(nodes, elements, materials, prop_K, {}, {}, fuentes, dir_borde_est, 'static')
        t_solve_est = time.time() - t0
        print(f"  Electrostatics solved in {t_solve_est:.4f} s, K_ff Condition: {cond_est:.2e}")

        # 3. Régimen Magnetostático
        dir_borde_mag = {nodo: 0.0 for nodo in boundaries['superior'] + boundaries['derecho'] + boundaries['inferior'] + boundaries['izq_sup']}
        dir_borde_mag.update({nodo: 5.0 for nodo in boundaries['bobina'] + boundaries['esfera']})

        # Estimar condición de K_ff en magnetostática
        cond_mag = estimar_condicion_k(K_global, dir_borde_mag)

        t0 = time.time()
        _ = resolver_sistema_maestro(nodes, elements, materials, prop_K, {}, {}, fuentes, dir_borde_mag, 'static')
        t_solve_mag = time.time() - t0
        print(f"  Magnetostatics solved in {t_solve_mag:.4f} s, K_ff Condition: {cond_mag:.2e}")

        # 4. Régimen Armónico
        dir_borde_arm = {nodo: 0.0 for nodo in boundaries['superior'] + boundaries['derecho']}
        dir_borde_arm.update({nodo: 100.0 for nodo in boundaries['bobina'] + boundaries['esfera']})

        t0 = time.time()
        _ = resolver_sistema_maestro(nodes, elements, materials, prop_K, prop_C, {}, fuentes, dir_borde_arm, 'harmonic', kwargs={'omega': 314.16})
        t_solve_arm = time.time() - t0
        print(f"  Harmonic solved in {t_solve_arm:.4f} s")

        # 5. Régimen Transitorio
        dir_borde_tran = {nodo: 0.0 for nodo in boundaries['superior'] + boundaries['derecho']}
        def boundary_func(t):
            return 100.0 * np.sin(2 * np.pi * 5 * t) * np.exp(-2*t)
        dir_borde_tran.update({nodo: boundary_func for nodo in boundaries['bobina'] + boundaries['esfera']})

        t0 = time.time()
        _ = resolver_sistema_maestro(nodes, elements, materials, prop_K, prop_C, prop_M, fuentes, dir_borde_tran, 'transient', kwargs={'dt': 0.02, 'num_pasos': 40})
        t_solve_tran = time.time() - t0
        print(f"  Transient (40 steps) solved in {t_solve_tran:.4f} s")

        # Formatear condicionales para CSV
        cond_est_str = f"{cond_est:.4e}" if not np.isnan(cond_est) else "N/A"
        cond_mag_str = f"{cond_mag:.4e}" if not np.isnan(cond_mag) else "N/A"

        csv_rows.append(f"{titulos[i]},{n_nodes},{n_elements},{t_assembly:.6f},{t_solve_est:.6f},{cond_est_str},{t_solve_mag:.6f},{cond_mag_str},{t_solve_arm:.6f},{t_solve_tran:.6f}")

    # Guardar en archivo CSV y Excel
    csv_path = os.path.join(out_dir, "exp_1_solver_performance.csv")
    xlsx_path = os.path.join(out_dir, "exp_1_solver_performance.xlsx")
    with open(csv_path, 'w') as f_out:
        for row in csv_rows:
            f_out.write(row + "\n")

    # Convertir a pandas DataFrame y guardar en Excel
    import pandas as pd
    rows_parsed = [r.split(",") for r in csv_rows]
    df_rend = pd.DataFrame(rows_parsed[1:], columns=rows_parsed[0])
    df_rend.to_excel(xlsx_path, index=False)
    print(f"\nPerformance analysis successfully saved at: {csv_path} y {xlsx_path}")

if __name__ == "__main__":
    generar_figures_completas()
    analizar_rendimiento()
