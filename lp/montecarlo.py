"""Validación por Monte Carlo: ¿el estimador recupera la respuesta verdadera y sus intervalos
cubren lo que dicen cubrir?

El proceso generador es un AR(1), y_t = rho·y_{t-1} + e_t, cuya respuesta a su propia innovación
es conocida: rho^h. Se simulan muchas muestras y, en cada horizonte, se mide:

    sesgo        promedio de las estimaciones menos rho^h
    cobertura    fracción de muestras en que el intervalo de 90% contiene a rho^h

para las dos formas de inferencia de `estimador.py` y para el intervalo por bootstrap de
`bootstrap.py`. Un intervalo de 90% bien construido cubre el valor verdadero cerca de 90% de las
veces. Las tres se evalúan sobre las mismas muestras.

    python -m lp.montecarlo                       # 2 000 repeticiones (un par de minutos)
    python -m lp.montecarlo --repeticiones 300    # versión rápida
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import estilo
from .bootstrap import banda_bootstrap_propia
from .estimador import proyeccion_local

RESULTADOS = Path(__file__).resolve().parents[1] / "resultados"
NIVEL = 0.90
TANDAS = 8          # las repeticiones se reparten en tandas con semillas propias: el resultado no depende de cuántos núcleos haya


def simular_ar1(rho: float, T: int, rng: np.random.Generator, quemado: int = 200) -> np.ndarray:
    e = rng.standard_normal(T + quemado)
    y = np.zeros(T + quemado)
    for t in range(1, T + quemado):
        y[t] = rho * y[t - 1] + e[t]
    return y[quemado:]


def _tanda(rho: float, T: int, horizontes: int, repeticiones: int, semilla, B: int) -> dict[str, np.ndarray]:
    """Una tanda de repeticiones. Cada muestra se evalúa con los tres métodos."""
    rng = np.random.default_rng(semilla)
    verdad = rho ** np.arange(horizontes + 1)
    forma = (repeticiones, horizontes + 1)
    out = {clave: np.full(forma, np.nan) for clave in
           ("irf_hac", "irf_aumentada", "cubre_hac", "cubre_aumentada", "cubre_bootstrap", "ancho_aumentada", "ancho_bootstrap")}
    for r in range(repeticiones):
        y = simular_ar1(rho, T, rng)
        for metodo in ("hac", "aumentada"):
            res = proyeccion_local(y, y, horizontes=horizontes, rezagos=0, metodo=metodo)
            inferior, superior = res.banda(NIVEL)
            out[f"irf_{metodo}"][r] = res.irf
            out[f"cubre_{metodo}"][r] = (inferior <= verdad) & (verdad <= superior)
        out["ancho_aumentada"][r] = superior - inferior
        if B:
            _, inferior, superior = banda_bootstrap_propia(y, horizontes, rezagos=0, nivel=NIVEL, B=B, rng=rng)
            out["cubre_bootstrap"][r] = (inferior <= verdad) & (verdad <= superior)
            out["ancho_bootstrap"][r] = superior - inferior
    return out


def experimento(rho: float, T: int = 240, horizontes: int = 36, repeticiones: int = 2000, semilla: int = 0,
                B: int = 0, procesos: int | None = 1) -> pd.DataFrame:
    """Sesgo y cobertura por horizonte para un AR(1) con persistencia `rho`.

    Con B > 0 se evalúa también el intervalo por bootstrap, con B muestras simuladas por repetición.
    `procesos` reparte las tandas entre núcleos (None usa los disponibles); no cambia el resultado.
    """
    semillas = np.random.SeedSequence([semilla, round(rho * 1000)]).spawn(TANDAS)
    tamanos = [len(x) for x in np.array_split(np.arange(repeticiones), TANDAS)]
    tareas = [(rho, T, horizontes, n, s, B) for n, s in zip(tamanos, semillas) if n]
    if procesos == 1:
        partes = [_tanda(*t) for t in tareas]
    else:
        with ProcessPoolExecutor(max_workers=procesos or min(TANDAS, os.cpu_count() or 1)) as grupo:
            partes = list(grupo.map(_tanda, *zip(*tareas)))
    junto = {clave: np.concatenate([parte[clave] for parte in partes]) for clave in partes[0]}
    tabla = pd.DataFrame({"verdadero": rho ** np.arange(horizontes + 1),
                          "media_hac": junto["irf_hac"].mean(axis=0),
                          "media_aumentada": junto["irf_aumentada"].mean(axis=0),
                          "cobertura_hac": junto["cubre_hac"].mean(axis=0),
                          "cobertura_aumentada": junto["cubre_aumentada"].mean(axis=0)},
                         index=pd.RangeIndex(horizontes + 1, name="horizonte"))
    if B:
        tabla["cobertura_bootstrap"] = junto["cubre_bootstrap"].mean(axis=0)
        tabla["ancho_aumentada"] = junto["ancho_aumentada"].mean(axis=0)
        tabla["ancho_bootstrap"] = junto["ancho_bootstrap"].mean(axis=0)
    return tabla


def figura_cobertura(tablas: dict[float, pd.DataFrame], T: int, repeticiones: int, ruta: Path) -> None:
    fig, ejes = plt.subplots(1, len(tablas), figsize=(11, 4.4), sharey=True)
    for ax, (rho, t) in zip(np.atleast_1d(ejes), tablas.items()):
        t = t.iloc[1:]                                           # en h = 0 la respuesta es 1 por construcción
        ax.axhline(NIVEL * 100, color=estilo.TINTA_2, lw=1.2, ls=(0, (4, 3)))
        ax.plot(t.index, t["cobertura_hac"] * 100, color=estilo.NARANJA, label="Newey-West")
        ax.plot(t.index, t["cobertura_aumentada"] * 100, color=estilo.AZUL, label="Aumentada con un rezago")
        if "cobertura_bootstrap" in t:
            ax.plot(t.index, t["cobertura_bootstrap"] * 100, color=estilo.AGUA, label="Aumentada con bootstrap")
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
    ap.add_argument("--B", type=int, default=499, help="muestras simuladas por repetición en el bootstrap (0 lo omite)")
    ap.add_argument("--procesos", type=int, default=None, help="núcleos a usar (por defecto, los disponibles)")
    args = ap.parse_args()
    estilo.aplicar()
    RESULTADOS.mkdir(exist_ok=True)
    tablas = {rho: experimento(rho, args.T, repeticiones=args.repeticiones, B=args.B, procesos=args.procesos)
              for rho in (0.5, 0.95)}
    pd.set_option("display.width", 200)
    for rho, t in tablas.items():
        t.round(4).to_csv(RESULTADOS / f"montecarlo_rho_{rho}.csv")
        print(f"\nrho = {rho}, T = {args.T}, {args.repeticiones:,} repeticiones, B = {args.B}")
        print(t.loc[[1, 6, 12, 24, 36]].round(3).to_string())
    figura_cobertura(tablas, args.T, args.repeticiones, RESULTADOS / "cobertura.png")
    figura_sesgo(tablas[0.95], 0.95, args.T, args.repeticiones, RESULTADOS / "sesgo.png")


if __name__ == "__main__":
    main()
