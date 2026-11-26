"""Intervalos por bootstrap para proyecciones locales aumentadas con rezagos.

Sigue el procedimiento que recomiendan Montiel Olea y Plagborg-Møller (2021): un bootstrap
recursivo «wild» sobre un VAR, con intervalo percentil-t.

    1. Se estima por MCO un VAR(p) con las variables de la proyección local y se corrige el sesgo
       de muestra finita de sus coeficientes (Pope, 1990), como propone Kilian (1998).
    2. Con ese VAR se calcula la respuesta que la proyección local estima. Es la «verdad» del
       mundo simulado.
    3. Se generan B muestras: cada residuo del VAR se multiplica por una normal estándar
       (así se conserva su heterocedasticidad) y se recorre el VAR desde un bloque inicial tomado
       al azar de los datos.
    4. En cada muestra se vuelve a estimar la proyección local y se guarda
       t* = (b* - verdad) / ee*.
    5. El intervalo es [b - q_alto·ee, b - q_bajo·ee], con los cuantiles de t* en lugar de los
       de la normal.

El intervalo usual supone que (b - verdad) / ee es normal y está centrado en cero. El bootstrap
no lo supone: si en muestras como esta el estimador queda sistemáticamente por debajo de la
verdad, los cuantiles de t* lo recogen y el intervalo se desplaza.
"""
from __future__ import annotations

import numpy as np

from .estimador import ResultadoLP, proyeccion_local


# ---------- VAR ----------
def var_mco(Y: np.ndarray, p: int, validas: np.ndarray | None = None):
    """VAR(p) con constante, por MCO. Devuelve (c, A, U): A tiene forma (p, k, k) y U, (n, k).

    `validas` marca con True los periodos utilizables; una fila entra si ella y sus p rezagos lo son.
    """
    Y = np.asarray(Y, dtype=float).reshape(len(Y), -1)
    T, k = Y.shape
    X = np.column_stack([np.ones(T - p)] + [Y[p - j:T - j] for j in range(1, p + 1)])
    dep = Y[p:]
    if validas is not None:
        limpia = np.array([validas[t - p:t + 1].all() for t in range(p, T)])
        X, dep = X[limpia], dep[limpia]
    coef, *_ = np.linalg.lstsq(X, dep, rcond=None)
    A = np.stack([coef[1 + (j - 1) * k:1 + j * k].T for j in range(1, p + 1)])
    return coef[0], A, dep - X @ coef


def companera(A: np.ndarray) -> np.ndarray:
    """Forma compañera del VAR(p): un VAR(1) en el vector apilado de p rezagos."""
    p, k, _ = A.shape
    F = np.zeros((k * p, k * p))
    F[:k] = np.concatenate(list(A), axis=1)
    F[k:, :-k] = np.eye(k * (p - 1))
    return F


def radio(A: np.ndarray) -> float:
    """Mayor raíz en valor absoluto. El VAR es estacionario si es menor que uno."""
    return float(np.abs(np.linalg.eigvals(companera(A))).max())


def corregir_sesgo(A: np.ndarray, U: np.ndarray, T: int) -> np.ndarray:
    """Corrección de primer orden del sesgo de MCO en un VAR (Pope, 1990).

    El sesgo aproximado de la matriz compañera F es -b/T, con

        b = G [ (I - F')^-1 + F'(I - F'^2)^-1 + Σ λ (I - λF')^-1 ] Γ(0)^-1

    donde G es la covarianza de las innovaciones, Γ(0) la de los datos y λ recorre las raíces de F.
    En un AR(1) se reduce a la fórmula de Kendall: sesgo = -(1 + 3ρ)/T.

    Salvaguarda de Kilian (1998): si el VAR estimado no es estacionario, no se corrige; si la
    corrección lo vuelve no estacionario, se reduce hasta que deja de hacerlo.
    """
    p, k, _ = A.shape
    F = companera(A)
    raices = np.linalg.eigvals(F)
    if np.abs(raices).max() >= 1:
        return A
    m = k * p
    G = np.zeros((m, m))
    G[:k, :k] = U.T @ U / len(U)
    gamma0 = np.linalg.solve(np.eye(m * m) - np.kron(F, F), G.ravel()).reshape(m, m)
    identidad, Ft = np.eye(m), F.T
    suma = np.linalg.inv(identidad - Ft) + Ft @ np.linalg.inv(identidad - Ft @ Ft)
    suma = suma + sum(raiz * np.linalg.inv(identidad - raiz * Ft) for raiz in raices)
    sesgo = -np.real(G @ suma @ np.linalg.inv(gamma0)) / T
    for delta in np.arange(1.0, -0.005, -0.01):
        corregida = F - delta * sesgo
        if np.abs(np.linalg.eigvals(corregida)).max() < 1:
            break
    return np.stack([corregida[:k, j * k:(j + 1) * k] for j in range(p)])


def respuesta_var(A: np.ndarray, sigma: np.ndarray, respuesta: int, impulso: int, horizontes: int) -> np.ndarray:
    """Lo que estima la proyección local cuando los datos salen de este VAR.

    Es la respuesta de la variable `respuesta` a un aumento de una unidad en la parte de la
    innovación de `impulso` que no explican las variables ordenadas antes que ella (los controles
    contemporáneos de la proyección local van primero).
    """
    p, k, _ = A.shape
    P = np.linalg.cholesky(sigma)
    psi = [np.eye(k)]
    for h in range(1, horizontes + 1):
        psi.append(sum(A[j - 1] @ psi[h - j] for j in range(1, min(h, p) + 1)))
    return np.array([(m @ P)[respuesta, impulso] for m in psi]) / P[impulso, impulso]


def simular_lote(c: np.ndarray, A: np.ndarray, U: np.ndarray, Y: np.ndarray, B: int, rng: np.random.Generator,
                 inicios: np.ndarray | None = None) -> np.ndarray:
    """B muestras del VAR por bootstrap recursivo «wild». Devuelve un arreglo (B, p + n, k)."""
    Y = np.asarray(Y, dtype=float).reshape(len(Y), -1)
    p, k, _ = A.shape
    n = len(U)
    posibles = np.arange(len(Y) - p + 1) if inicios is None else np.asarray(inicios)
    inicio = rng.choice(posibles, size=B)
    sim = np.empty((B, p + n, k))
    for j in range(p):
        sim[:, j] = Y[inicio + j]
    eta = rng.standard_normal((B, n))
    for t in range(p, p + n):
        nuevo = c + eta[:, t - p, None] * U[t - p]
        for j in range(1, p + 1):
            nuevo = nuevo + sim[:, t - j] @ A[j - 1].T
        sim[:, t] = nuevo
    return sim


def ajustar_var(Y: np.ndarray, p: int, validas: np.ndarray | None = None, corregir: bool = True):
    """VAR(p) listo para simular: coeficientes (corregidos por sesgo si se pide), constante y residuos."""
    Y = np.asarray(Y, dtype=float).reshape(len(Y), -1)
    c, A, U = var_mco(Y, p, validas)
    if corregir:
        A = corregir_sesgo(A, U, len(U))
        T = len(Y)
        ajuste = sum(Y[p - j:T - j] @ A[j - 1].T for j in range(1, p + 1))
        resto = Y[p:] - ajuste
        if validas is not None:
            resto = resto[np.array([validas[t - p:t + 1].all() for t in range(p, T)])]
        c = resto.mean(axis=0)
        U = resto - c
    return c, A, U


# ---------- proyección local vectorizada (una serie sobre su propia innovación) ----------
def lp_propia_lote(Y: np.ndarray, horizontes: int, rezagos: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Proyección local aumentada de cada fila de Y (lote, T) sobre su propia innovación.

    Da lo mismo que `proyeccion_local(y, y, rezagos=rezagos, metodo="aumentada")` para cada serie,
    pero resuelve todo el lote a la vez. Devuelve (irf, ee), de forma (lote, horizontes + 1).
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    lote, T = Y.shape
    P = rezagos + 1
    irf, ee = np.full((lote, horizontes + 1), np.nan), np.full((lote, horizontes + 1), np.nan)
    for h in range(horizontes + 1):
        n = T - P - h
        k = P + 2
        if n <= k + 2:
            continue
        Z = np.stack([np.ones((lote, n))] + [Y[:, P - j:P - j + n] for j in range(P + 1)], axis=2)
        dep = Y[:, P + h:P + h + n]
        ZtZ = np.einsum("bni,bnj->bij", Z, Z)
        b = np.linalg.solve(ZtZ, np.einsum("bni,bn->bi", Z, dep)[..., None])[..., 0]
        u = dep - np.einsum("bni,bi->bn", Z, b)
        pan = np.linalg.inv(ZtZ)
        V = n / (n - k) * pan @ np.einsum("bni,bn,bnj->bij", Z, u ** 2, Z) @ pan
        irf[:, h], ee[:, h] = b[:, 1], np.sqrt(V[:, 1, 1])
    return irf, ee


# ---------- intervalos ----------
def percentil_t(irf: np.ndarray, ee: np.ndarray, t_estrella: np.ndarray, nivel: float = 0.90):
    """Intervalo percentil-t a partir de los estadísticos t* de las muestras simuladas (filas)."""
    alfa = 1 - nivel
    q_bajo = np.nanquantile(t_estrella, alfa / 2, axis=0)
    q_alto = np.nanquantile(t_estrella, 1 - alfa / 2, axis=0)
    return irf - q_alto * ee, irf - q_bajo * ee


def banda_bootstrap(y, x, controles=None, horizontes: int = 24, rezagos: int = 4, acumulada: bool = False,
                    excluir=None, nivel: float = 0.90, B: int = 999, semilla: int = 0, corregir: bool = True):
    """Proyección local aumentada con intervalo percentil-t por bootstrap.

    Devuelve (resultado, inferior, superior). Los argumentos son los de `proyeccion_local`.
    """
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    T = len(y)
    W = None if controles is None else np.asarray(controles, dtype=float).reshape(T, -1)
    propia = np.array_equal(y, x, equal_nan=True)
    p = rezagos + 1 if propia else rezagos
    if p < 1:
        raise ValueError("El bootstrap necesita al menos un rezago para construir el VAR.")
    res = proyeccion_local(y, x, W, horizontes, rezagos, "aumentada", acumulada, excluir)

    # Orden del VAR: controles contemporáneos, impulso y respuesta.
    columnas = ([] if W is None else list(W.T)) + [x] + ([] if propia else [y])
    Y = np.column_stack(columnas)
    m = 0 if W is None else W.shape[1]
    validas = ~np.isnan(Y).any(axis=1)
    if excluir is not None:
        validas &= ~np.asarray(excluir, dtype=bool)
    c, A, U = ajustar_var(Y, p, validas, corregir)
    verdad = respuesta_var(A, U.T @ U / len(U), m if propia else m + 1, m, horizontes)

    rng = np.random.default_rng(semilla)
    inicios = np.array([t for t in range(T - p + 1) if validas[t:t + p].all()])
    sim = simular_lote(c, A, U, Y, B, rng, inicios)
    t_estrella = np.full((B, horizontes + 1), np.nan)
    for b in range(B):
        s = sim[b]
        r = proyeccion_local(s[:, m] if propia else s[:, m + 1], s[:, m], None if W is None else s[:, :m],
                             horizontes, rezagos, "aumentada", acumulada)
        t_estrella[b] = (r.irf - verdad) / r.ee
    inferior, superior = percentil_t(res.irf, res.ee, t_estrella, nivel)
    return res, inferior, superior


def banda_bootstrap_propia(y, horizontes: int, rezagos: int = 0, nivel: float = 0.90, B: int = 499,
                           rng: np.random.Generator | None = None, corregir: bool = True):
    """Versión rápida de `banda_bootstrap` para una serie sobre su propia innovación, sin controles."""
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(0) if rng is None else rng
    irf, ee = (v[0] for v in lp_propia_lote(y[None, :], horizontes, rezagos))
    c, A, U = ajustar_var(y[:, None], rezagos + 1, None, corregir)
    verdad = respuesta_var(A, U.T @ U / len(U), 0, 0, horizontes)
    sim = simular_lote(c, A, U, y[:, None], B, rng)[:, :, 0]
    irf_b, ee_b = lp_propia_lote(sim, horizontes, rezagos)
    with np.errstate(invalid="ignore", divide="ignore"):       # en h = 0 la respuesta es 1 sin error: 0/0
        t_estrella = (irf_b - verdad) / ee_b
    inferior, superior = percentil_t(irf, ee, t_estrella, nivel)
    return ResultadoLP(np.arange(horizontes + 1), irf, ee, np.full(horizontes + 1, np.nan), "bootstrap"), inferior, superior
