"""Proyecciones locales (Jordà, 2005) con numpy.

Para cada horizonte h se estima por MCO una regresión distinta:

    y_{t+h} = a_h + b_h·x_t + c_h'·w_t + (rezagos de x, y, w) + u_{t+h}

y la función impulso-respuesta es la secuencia b_0, b_1, ..., b_H. `x` es el impulso (un choque
observado o la variable cuyo efecto interesa) y `w` son controles contemporáneos opcionales.

Dos formas de inferencia:

    "hac"        Errores de Newey-West con rezago de truncamiento h. Es la práctica tradicional:
                 el error de la regresión en h es una media móvil de orden h.
    "aumentada"  Se agrega un rezago más de cada variable y se usan errores robustos a
                 heterocedasticidad, sin corrección por autocorrelación (Montiel Olea y
                 Plagborg-Møller, 2021). La simulación de `montecarlo.py` muestra por qué conviene.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Z_NORMAL = {0.68: 0.9945, 0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}


@dataclass
class ResultadoLP:
    horizontes: np.ndarray
    irf: np.ndarray          # b_h
    ee: np.ndarray           # error estándar de b_h
    n: np.ndarray            # observaciones usadas en cada horizonte
    metodo: str

    def banda(self, nivel: float = 0.90) -> tuple[np.ndarray, np.ndarray]:
        z = Z_NORMAL[nivel]
        return self.irf - z * self.ee, self.irf + z * self.ee


def rezagar(s: np.ndarray, j: int) -> np.ndarray:
    """s_{t-j} alineado con t (j > 0) o s_{t+|j|} (j < 0); los extremos sin dato quedan en NaN."""
    out = np.full(len(s), np.nan)
    if j == 0:
        out[:] = s
    elif j > 0:
        out[j:] = s[:-j]
    else:
        out[:j] = s[-j:]
    return out


def covarianza_robusta(Z: np.ndarray, u: np.ndarray, rezagos_hac: int = 0) -> np.ndarray:
    """Matriz de covarianzas de MCO. Con `rezagos_hac = 0` es la de Eicker-Huber-White; con L > 0,
    la de Newey-West con núcleo de Bartlett. Ambas con la corrección n / (n - k)."""
    n, k = Z.shape
    pan = np.linalg.inv(Z.T @ Z)
    zu = Z * u[:, None]
    S = zu.T @ zu
    for lag in range(1, min(rezagos_hac, n - 1) + 1):
        G = zu[lag:].T @ zu[:-lag]
        S += (1 - lag / (rezagos_hac + 1)) * (G + G.T)
    return n / (n - k) * pan @ S @ pan


def proyeccion_local(y, x, controles=None, horizontes: int = 24, rezagos: int = 4, metodo: str = "aumentada",
                     acumulada: bool = False, excluir=None) -> ResultadoLP:
    """Respuesta de `y` a un aumento de una unidad en `x`, de 0 a `horizontes` periodos.

    y, x        series del mismo largo. Si son la misma, se estima la respuesta de y a su propia innovación.
    controles   matriz (T, m) de controles contemporáneos; entran en t y con los mismos rezagos.
    rezagos     rezagos de cada variable. Con metodo="aumentada" se agrega uno más.
    acumulada   si es True, la variable dependiente es y_{t+h} - y_{t-1}: sirve para variables en
                logaritmos, donde interesa el cambio acumulado del nivel.
    excluir     vector booleano (T). Se descartan las filas en que la variable dependiente, el
                impulso o alguno de sus rezagos cae en un periodo marcado.
    """
    if metodo not in ("aumentada", "hac"):
        raise ValueError("metodo debe ser 'aumentada' o 'hac'")
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    T = len(y)
    W = None if controles is None else np.asarray(controles, dtype=float).reshape(T, -1)
    P = rezagos + (metodo == "aumentada")
    series = [x] if np.array_equal(y, x, equal_nan=True) else [x, y]
    if W is not None:
        series += list(W.T)
    columnas = [np.ones(T), x, *([] if W is None else W.T)]
    columnas += [rezagar(s, j) for j in range(1, P + 1) for s in series]
    Z = np.column_stack(columnas)
    marcado = np.zeros(T, dtype=bool) if excluir is None else np.asarray(excluir, dtype=bool)
    # un periodo t queda fuera si él o alguno de sus P rezagos (o t-1, en la versión acumulada) está marcado
    atras = max(P, 1 if acumulada else 0)
    sucio = np.array([marcado[max(0, t - atras):t + 1].any() for t in range(T)])

    irf, ee, n = (np.full(horizontes + 1, np.nan) for _ in range(3))
    for h in range(horizontes + 1):
        dep = rezagar(y, -h) - (rezagar(y, 1) if acumulada else 0.0)
        futuro = rezagar(marcado.astype(float), -h)             # ¿está marcado el periodo t + h?
        ok = ~np.isnan(dep) & ~np.isnan(Z).any(axis=1) & ~sucio & (futuro == 0)
        Zh, dh = Z[ok], dep[ok]
        if len(dh) <= Z.shape[1] + 2:
            continue
        b, *_ = np.linalg.lstsq(Zh, dh, rcond=None)
        V = covarianza_robusta(Zh, dh - Zh @ b, 0 if metodo == "aumentada" else h)
        irf[h], ee[h], n[h] = b[1], np.sqrt(V[1, 1]), len(dh)
    return ResultadoLP(np.arange(horizontes + 1), irf, ee, n, metodo)
