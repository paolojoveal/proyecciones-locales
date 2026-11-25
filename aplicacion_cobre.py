"""Aplicación: ¿qué pasa en el Perú después de un alza del precio del cobre?

Proyecciones locales mensuales de cinco variables peruanas sobre la variación del precio del
cobre, de 0 a 24 meses, con inferencia aumentada con rezagos y bandas de 90%.

    impulso      variación % mensual del precio del cobre (LME)
    respuestas   tipo de cambio, PBI desestacionalizado e IPC (cambio % acumulado del nivel);
                 riesgo país EMBIG (puntos básicos) y tasa de referencia (puntos porcentuales)
    controles    seis rezagos del impulso y de la respuesta; en la especificación principal,
                 además, el VIX, el dólar global y la tasa de la Fed, contemporáneos y rezagados

Las respuestas se expresan para un alza de 10% en el precio del cobre.

    python aplicacion_cobre.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import datos
from lp import estilo
from lp.estimador import proyeccion_local

RESULTADOS = Path(__file__).resolve().parent / "resultados"
HORIZONTES, REZAGOS, ALZA = 24, 6, 10.0
CIERRES = ("2020-03-01", "2020-12-01")          # meses de cierre por la pandemia

# respuesta -> (columna, transformación, unidad, usa la exclusión de 2020)
RESPUESTAS = {
    "Tipo de cambio (S/ por US$)": ("tipo_cambio", "log", "%", False),
    "Riesgo país, EMBIG": ("embig", "nivel", "pb", False),
    "PBI desestacionalizado": ("pbi_sa", "log", "%", True),
    "Índice de precios al consumidor": ("ipc", "log", "%", False),
    "Tasa de referencia": ("tasa_referencia", "nivel", "pp", False),
}


def preparar(d: pd.DataFrame) -> pd.DataFrame:
    m = pd.DataFrame(index=d.index)
    m["d_cobre"] = 100 * np.log(d["cobre"]).diff()
    for col, tipo, _, _ in RESPUESTAS.values():
        m[col] = 100 * np.log(d[col]) if tipo == "log" else d[col]
    m["d_vix"] = 100 * np.log(d["vix"]).diff()
    m["d_dolar"] = 100 * np.log(d["dolar"]).diff()
    m["d_fed"] = d["fed_funds"].diff()
    return m.dropna()


def estimar(m: pd.DataFrame, globales: bool) -> dict[str, pd.DataFrame]:
    controles = m[["d_vix", "d_dolar", "d_fed"]].to_numpy() if globales else None
    cierre = np.asarray((m.index >= CIERRES[0]) & (m.index <= CIERRES[1]))
    out = {}
    for nombre, (col, _, _, sin_2020) in RESPUESTAS.items():
        r = proyeccion_local(m[col].to_numpy(), m["d_cobre"].to_numpy(), controles, HORIZONTES, REZAGOS,
                             metodo="aumentada", acumulada=True, excluir=cierre if sin_2020 else None)
        inferior, superior = r.banda(0.90)
        out[nombre] = pd.DataFrame({"respuesta": r.irf, "inferior": inferior, "superior": superior, "n": r.n},
                                   index=pd.RangeIndex(HORIZONTES + 1, name="mes")).mul([ALZA, ALZA, ALZA, 1])
    return out


def figura(principal: dict, alternativa: dict, pie: str) -> None:
    fig, ejes = plt.subplots(2, 3, figsize=(12, 6.6), sharex=True)
    ejes = ejes.ravel()
    for ax, (nombre, r) in zip(ejes, principal.items()):
        unidad = RESPUESTAS[nombre][2]
        ax.axhline(0, color=estilo.EJE, lw=1)
        ax.fill_between(r.index, r["inferior"], r["superior"], color=estilo.AZUL, alpha=0.14, lw=0)
        ax.plot(r.index, alternativa[nombre]["respuesta"], color=estilo.CONTEXTO, lw=1.6)
        ax.plot(r.index, r["respuesta"], color=estilo.AZUL)
        ax.set_title(f"{nombre} ({unidad})", fontsize=10.5, loc="left")
    ax = ejes[-1]                                         # el sexto panel lleva la leyenda
    ax.axis("off")
    ax.plot([], [], color=estilo.AZUL, label="Con controles globales (VIX, dólar, Fed)")
    ax.fill_between([], [], [], color=estilo.AZUL, alpha=0.14, lw=0, label="Banda de 90%")
    ax.plot([], [], color=estilo.CONTEXTO, lw=1.6, label="Sin controles globales")
    ax.legend(loc="center left", fontsize=9.5)
    for ax in ejes[:5]:
        ax.tick_params(labelbottom=True)
    for ax in ejes[2:5]:
        ax.set_xlabel("Meses después del alza")
    fig.suptitle(f"Respuesta de la economía peruana a un alza de {ALZA:.0f}% en el precio del cobre", x=0.012, y=0.985,
                 ha="left", fontsize=12.5, fontweight="bold")
    fig.text(0.012, 0.94, "Proyecciones locales mensuales aumentadas con rezagos. Cambio acumulado frente al mes previo al alza",
             fontsize=9.5, color=estilo.TINTA_2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.text(0.012, 0.012, pie, fontsize=8, color=estilo.TENUE)
    fig.savefig(RESULTADOS / "respuesta_al_cobre.png")
    plt.close(fig)


def main() -> None:
    estilo.aplicar()
    m = preparar(datos.cargar())
    principal, alternativa = estimar(m, globales=True), estimar(m, globales=False)
    print(f"Muestra: {m.index[0]:%Y-%m} a {m.index[-1]:%Y-%m} ({len(m)} meses). Respuestas a un alza de {ALZA:.0f}% del cobre.\n")
    filas = []
    for nombre, r in principal.items():
        for h in (0, 3, 6, 12, 24):
            f = r.loc[h]
            filas.append({"respuesta": f"{nombre} ({RESPUESTAS[nombre][2]})", "mes": h, "efecto": f["respuesta"],
                          "inferior": f["inferior"], "superior": f["superior"],
                          "sin controles globales": alternativa[nombre].at[h, "respuesta"], "n": int(f["n"])})
    tabla = pd.DataFrame(filas).set_index(["respuesta", "mes"])
    pd.set_option("display.width", 200)
    print(tabla.round(2).to_string())
    RESULTADOS.mkdir(exist_ok=True)
    figura(principal, alternativa, f"Fuente: BCRP y FRED. Datos de {m.index[0]:%m/%Y} a {m.index[-1]:%m/%Y}. Elaboración propia.")
    tabla.round(4).to_csv(RESULTADOS / "respuesta_al_cobre.csv")
    pd.concat(principal, axis=1).round(4).to_csv(RESULTADOS / "respuesta_al_cobre_completa.csv")


if __name__ == "__main__":
    main()
