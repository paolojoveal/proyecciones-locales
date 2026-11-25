"""Validación por Monte Carlo: ¿el estimador recupera la respuesta verdadera y sus intervalos
cubren lo que dicen cubrir?

El proceso generador es un AR(1), y_t = rho·y_{t-1} + e_t, cuya respuesta a su propia innovación
es conocida: rho^h. Se simulan muchas muestras y, en cada horizonte, se mide:

    sesgo        promedio de las estimaciones menos rho^h
    cobertura    fracción de muestras en que el intervalo de 90% contiene a rho^h

para las dos formas de inferencia de `estimador.py`. Un intervalo de 90% bien construido cubre
el valor verdadero cerca de 90% de las veces.

    python -m lp.montecarlo                       # 2 000 repeticiones (unos minutos)
    python -m lp.montecarlo --repeticiones 300    # versión rápida
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import estilo
from .estimador import proyeccion_local

RESULTADOS = Path(__file__).resolve().parents[1] / "resultados"
NIVEL = 0.90


def simular_ar1(rho: float, T: int, rng: np.random.Generator, quemado: int = 200) -> np.ndarray:
    e = rng.standard_normal(T + quemado)
    y = np.zeros(T + quemado)
    for t in range(1, T + quemado):
        y[t] = rho * y[t - 1] + e[t]
    return y[quemado:]


def experimento(rho: float, T: int = 240, horizontes: int = 36, repeticiones: int = 2000, semilla: int = 0) -> pd.DataFrame:
    """Sesgo y cobertura por horizonte para un AR(1) con persistencia `rho`."""
    rng = np.random.default_rng(semilla)
    verdad = rho ** np.arange(horizontes + 1)
    estimaciones = {m: np.empty((repeticiones, horizontes + 1)) for m in ("hac", "aumentada")}
    cubre = {m: np.empty((repeticiones, horizontes + 1), dtype=bool) for m in estimaciones}
    for r in range(repeticiones):
        y = simular_ar1(rho, T, rng)
        for metodo in estimaciones:
            res = proyeccion_local(y, y, horizontes=horizontes, rezagos=0, metodo=metodo)
            inferior, superior = res.banda(NIVEL)
            estimaciones[metodo][r] = res.irf
            cubre[metodo][r] = (inferior <= verdad) & (verdad <= superior)
    return pd.DataFrame({"verdadero": verdad,
                         "media_hac": estimaciones["hac"].mean(axis=0),
                         "media_aumentada": estimaciones["aumentada"].mean(axis=0),
                         "cobertura_hac": cubre["hac"].mean(axis=0),
                         "cobertura_aumentada": cubre["aumentada"].mean(axis=0)},
                        index=pd.RangeIndex(horizontes + 1, name="horizonte"))


def figura_cobertura(tablas: dict[float, pd.DataFrame], T: int, repeticiones: int, ruta: Path) -> None:
    fig, ejes = plt.subplots(1, len(tablas), figsize=(11, 4.4), sharey=True)
    for ax, (rho, t) in zip(np.atleast_1d(ejes), tablas.items()):
        t = t.iloc[1:]                                           # en h = 0 la respuesta es 1 por construcción
        ax.axhline(NIVEL * 100, color=estilo.TINTA_2, lw=1.2, ls=(0, (4, 3)))
        ax.plot(t.index, t["cobertura_hac"] * 100, color=estilo.NARANJA, label="Newey-West")
        ax.plot(t.index, t["cobertura_aumentada"] * 100, color=estilo.AZUL, label="Aumentada con un rezago")
        estilo.etiqueta(ax, t.index[2], NIVEL * 100, "Nominal: 90%", dx=0, dy=9, fontsize=8.5)
        ax.set_xlabel("Horizonte")
        ax.set_ylim(40, 100)
        estilo.formato_pct(ax)
        estilo.titulo(ax, f"Persistencia ρ = {rho}", "Cobertura efectiva del intervalo de 90%")
    np.atleast_1d(ejes)[0].legend(loc="lower left")
    estilo.guardar(fig, ruta, f"Simulación propia: AR(1), T = {T}, {repeticiones:,} repeticiones por panel.")


def figura_sesgo(t: pd.DataFrame, rho: float, T: int, repeticiones: int, ruta: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.plot(t.index, t["verdadero"], color=estilo.CONTEXTO, lw=3, label=f"Respuesta verdadera, {rho}^h")
    ax.plot(t.index, t["media_hac"], color=estilo.NARANJA, label="Promedio de las estimaciones, sin rezago adicional")
    ax.plot(t.index, t["media_aumentada"], color=estilo.AZUL, label="Promedio de las estimaciones, aumentada")
    ax.set_xlabel("Horizonte")
    estilo.titulo(ax, "El sesgo de muestra finita crece con el horizonte",
                  f"Respuesta de un AR(1) con ρ = {rho} a su propia innovación")
    ax.legend(loc="upper right")
    estilo.guardar(fig, ruta, f"Simulación propia: T = {T}, {repeticiones:,} repeticiones.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repeticiones", type=int, default=2000)
    ap.add_argument("--T", type=int, default=240, help="tamaño de muestra (240 = 20 años de datos mensuales)")
    args = ap.parse_args()
    estilo.aplicar()
    RESULTADOS.mkdir(exist_ok=True)
    tablas = {rho: experimento(rho, args.T, repeticiones=args.repeticiones) for rho in (0.5, 0.95)}
    for rho, t in tablas.items():
        t.round(4).to_csv(RESULTADOS / f"montecarlo_rho_{rho}.csv")
        print(f"\nrho = {rho}, T = {args.T}, {args.repeticiones:,} repeticiones")
        print(t.loc[[1, 6, 12, 24, 36]].round(3).to_string())
    figura_cobertura(tablas, args.T, args.repeticiones, RESULTADOS / "cobertura.png")
    figura_sesgo(tablas[0.95], 0.95, args.T, args.repeticiones, RESULTADOS / "sesgo.png")


if __name__ == "__main__":
    main()
