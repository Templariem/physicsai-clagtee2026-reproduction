import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def obtener_cuadratura(precision):
    """
    Devuelve los puntos (q) y pesos (w) de Gauss-Legendre para un cuadrilátero [-1, 1]^2.
    """
    if precision == 1:
        q = np.array([[0.0, 0.0]])
        w = np.array([4.0])
    elif precision == 2:
        p = 1.0 / np.sqrt(3.0)
        q = np.array([
            [-p, -p],
            [ p, -p],
            [ p,  p],
            [-p,  p]
        ])
        w = np.array([1.0, 1.0, 1.0, 1.0])
    elif precision == 3:
        p = np.sqrt(3.0 / 5.0)
        w1 = 5.0 / 9.0; w2 = 8.0 / 9.0
        pts_1d = [-p, 0.0, p]
        w_1d = [w1, w2, w1]

        q, w = [], []
        for i in range(3):
            for j in range(3):
                q.append([pts_1d[i], pts_1d[j]])
                w.append(w_1d[i] * w_1d[j])
        q = np.array(q)
        w = np.array(w)
    elif precision == 4:
        p1 = np.sqrt((3.0 - 2.0*np.sqrt(6.0/5.0)) / 7.0)
        p2 = np.sqrt((3.0 + 2.0*np.sqrt(6.0/5.0)) / 7.0)
        w1 = (18.0 + np.sqrt(30.0)) / 36.0
        w2 = (18.0 - np.sqrt(30.0)) / 36.0

        pts_1d = [-p2, -p1, p1, p2]
        w_1d = [w2, w1, w1, w2]

        q, w = [], []
        for i in range(4):
            for j in range(4):
                q.append([pts_1d[i], pts_1d[j]])
                w.append(w_1d[i] * w_1d[j])
        q = np.array(q)
        w = np.array(w)
    else:
        raise ValueError("Precision not supported. Use 1, 2, 3 or 4.")

    return q, w

def funciones_forma_y_derivadas(xi, eta):
    N = 0.25 * np.array([
        (1 - xi) * (1 - eta),
        (1 + xi) * (1 - eta),
        (1 + xi) * (1 + eta),
        (1 - xi) * (1 + eta)
    ])
    dN_dloc = 0.25 * np.array([
        [-(1 - eta),  (1 - eta), (1 + eta), -(1 + eta)],
        [-(1 - xi),  -(1 + xi),  (1 + xi),  (1 - xi) ]
    ])
    return N, dN_dloc

def calcular_matrices_locales(coords_elemento, c_K, c_C, c_M, f_val, precision=2):
    """
    Calcula las matrices locales Ke, Ce, Me y el vector fe para un cuadrilátero Q4.
    Maneja aritmética compleja si las propiedades son complejas.
    """
    q, w = obtener_cuadratura(precision)

    dtype_use = complex if isinstance(c_K, complex) or isinstance(c_C, complex) or isinstance(c_M, complex) or isinstance(f_val, complex) else float

    Ke = np.zeros((4, 4), dtype=dtype_use)
    Ce = np.zeros((4, 4), dtype=dtype_use)
    Me = np.zeros((4, 4), dtype=dtype_use)
    fe = np.zeros(4, dtype=dtype_use)

    for k in range(len(w)):
        xi, eta = q[k]
        peso = w[k]

        N, dN_dloc = funciones_forma_y_derivadas(xi, eta)

        J = np.dot(dN_dloc, coords_elemento)
        detJ = np.linalg.det(J)

        if detJ <= 0:
            raise ValueError(f"Negative or zero Jacobian! detJ={detJ}")

        J_inv = np.linalg.inv(J)
        dN_dglob = np.dot(J_inv, dN_dloc)

        # Integral para Ke (Gradientes)
        Ke += c_K * np.dot(dN_dglob.T, dN_dglob) * detJ * peso

        # Integral para Ce y Me (Funciones de forma)
        N_mat = N.reshape(1, 4)
        N_N = np.dot(N_mat.T, N_mat)
        Ce += c_C * N_N * detJ * peso
        Me += c_M * N_N * detJ * peso

        # Integral para vector de cargas fe
        fe += f_val * N * detJ * peso

    return Ke, Ce, Me, fe
def ensamblar_sistema_maestro(nodes, elements, materials, prop_K, prop_C, prop_M, fuentes, tipo_analisis='static'):
    n_nodes = len(nodes)
    I, J = [], []
    V_K, V_C, V_M, V_F = [], [], [], []

    # Evaluar si necesitamos sistema complejo de entrada
    any_complex = (tipo_analisis == 'harmonic' or
                   any(isinstance(v, complex) for v in prop_K.values()) or
                   any(isinstance(v, complex) for v in prop_C.values()) or
                   any(isinstance(v, complex) for v in prop_M.values()) or
                   any(isinstance(v, complex) for v in fuentes.values()))

    dtype_use = complex if any_complex else float
    F_base = np.zeros(n_nodes, dtype=dtype_use)

    for idx, el in enumerate(elements):
        coords_elemento = nodes[el]
        mat_id = materials[idx]

        c_K = prop_K.get(mat_id, 0.0)
        c_C = prop_C.get(mat_id, 0.0)
        c_M = prop_M.get(mat_id, 0.0)
        f_val = fuentes.get(mat_id, 0.0)

        Ke, Ce, Me, fe = calcular_matrices_locales(coords_elemento, c_K, c_C, c_M, f_val, precision=2)

        for i in range(4):
            F_base[el[i]] += fe[i]
            for j in range(4):
                I.append(el[i])
                J.append(el[j])
                V_K.append(Ke[i, j])
                V_C.append(Ce[i, j])
                V_M.append(Me[i, j])

    K_global = sp.coo_matrix((V_K, (I, J)), shape=(n_nodes, n_nodes), dtype=dtype_use).tocsr()
    C_global = sp.coo_matrix((V_C, (I, J)), shape=(n_nodes, n_nodes), dtype=dtype_use).tocsr()
    M_global = sp.coo_matrix((V_M, (I, J)), shape=(n_nodes, n_nodes), dtype=dtype_use).tocsr()

    return K_global, C_global, M_global, F_base
def resolver_sistema_maestro(nodes, elements, materials, prop_K, prop_C, prop_M, fuentes, dir_borde, tipo_analisis, kwargs=None):
    """
    Ensamblador maestro generalizado.

    - prop_K, prop_C, prop_M, fuentes: diccionarios con {ID_material: valor}
    - dir_borde: diccionario {nodo_global: valor_fijado}
    - tipo_analisis: 'static', 'harmonic', 'transient'
    - kwargs: parámetros específicos (ej. 'omega' para armónico, 'dt' y 'num_pasos' para transient)
    """
    if kwargs is None:
        kwargs = {}

    n_nodes = len(nodes)
    I, J = [], []
    V_K, V_C, V_M, V_F = [], [], [], []

    # Evaluar si necesitamos sistema complejo de entrada
    any_complex = (tipo_analisis == 'harmonic' or
                   any(isinstance(v, complex) for v in prop_K.values()) or
                   any(isinstance(v, complex) for v in prop_C.values()) or
                   any(isinstance(v, complex) for v in prop_M.values()) or
                   any(isinstance(v, complex) for v in fuentes.values()))

    dtype_use = complex if any_complex else float
    F_base = np.zeros(n_nodes, dtype=dtype_use)

    print("Assembling Global Matrices...")
    K_global, C_global, M_global, F_base = ensamblar_sistema_maestro(
        nodes, elements, materials, prop_K, prop_C, prop_M, fuentes, tipo_analisis
    )

    # Evaluar si necesitamos sistema complejo de entrada
    any_complex = (tipo_analisis == 'harmonic' or
                   any(isinstance(v, complex) for v in prop_K.values()) or
                   any(isinstance(v, complex) for v in prop_C.values()) or
                   any(isinstance(v, complex) for v in prop_M.values()) or
                   any(isinstance(v, complex) for v in fuentes.values()))

    dtype_use = complex if any_complex else float

    BIG_NUMBER = 1e15

    print(f"Solving Analysis: {tipo_analisis}")
    if tipo_analisis == 'static':
        A = K_global.copy()
        F = F_base.copy()
        for nodo, valor in dir_borde.items():
            A[nodo, nodo] += BIG_NUMBER
            F[nodo] += BIG_NUMBER * valor
        return spla.spsolve(A, F)

    elif tipo_analisis == 'harmonic':
        omega = kwargs.get('omega', 1.0)
        A = K_global + 1j * omega * C_global - (omega**2) * M_global
        F = F_base.copy()
        for nodo, valor in dir_borde.items():
            A[nodo, nodo] += BIG_NUMBER
            F[nodo] += BIG_NUMBER * valor
        return spla.spsolve(A, F)

    elif tipo_analisis == 'transient':
        dt = kwargs.get('dt', 0.1)
        num_pasos = kwargs.get('num_pasos', 10)
        f_historial = kwargs.get('f_historial', None)

        # Algoritmo de integración de Newmark-beta
        beta = 0.25
        gamma = 0.5

        a0 = 1.0 / (beta * dt**2)
        a1 = gamma / (beta * dt)
        a2 = 1.0 / (beta * dt)
        a3 = 1.0 / (2.0 * beta) - 1.0
        a4 = gamma / beta - 1.0
        a5 = dt / 2.0 * (gamma / beta - 2.0)
        a6 = dt * (1.0 - gamma)
        a7 = gamma * dt

        A_eff = K_global + a0 * M_global + a1 * C_global

        # Penalización para condiciones de borde (asumimos constantes en el tiempo para simplicidad)
        for nodo in dir_borde.keys():
            A_eff[nodo, nodo] += BIG_NUMBER

        U = np.zeros(n_nodes, dtype=dtype_use)
        U_dot = np.zeros(n_nodes, dtype=dtype_use)
        U_ddot = np.zeros(n_nodes, dtype=dtype_use)

        U_history = [U.copy()]

        for step in range(1, num_pasos + 1):
            t = step * dt
            if f_historial is not None:
                F_t = f_historial(t, F_base)
            else:
                F_t = F_base.copy()

            F_eff = F_t + M_global.dot(a0 * U + a2 * U_dot + a3 * U_ddot) + C_global.dot(a1 * U + a4 * U_dot + a5 * U_ddot)

            # Aplicar BC en F_eff
            for nodo, valor in dir_borde.items():
                if callable(valor):
                    F_eff[nodo] = valor(t) * BIG_NUMBER
                else:
                    F_eff[nodo] = valor * BIG_NUMBER

            U_next = spla.spsolve(A_eff, F_eff)

            U_ddot_next = a0 * (U_next - U) - a2 * U_dot - a3 * U_ddot
            U_dot_next = U_dot + a6 * U_ddot + a7 * U_ddot_next

            U = U_next
            U_dot = U_dot_next
            U_ddot = U_ddot_next

            U_history.append(U.copy())

            if step % max(1, num_pasos // 10) == 0:
                print(f"  Progress: Step {step}/{num_pasos} (t = {t:.4f})")

        return np.array(U_history)

    else:
        raise ValueError(f"Unsupported analysis type: {tipo_analisis}")
