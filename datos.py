"""Descarga de datos mensuales: series del BCRP y de FRED, con caché local en data/.

Cada serie del BCRP se pide por separado y se comprueba que el nombre oficial que devuelve la
API termine como se espera. Así un código equivocado o recodificado falla con un mensaje claro en
lugar de entregar otra variable.
"""
from __future__ import annotations

import io
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).resolve().parent / "data"
BCRP_URL = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api/{codigo}/json/{inicio}/{fin}/esp"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={serie}"

# nombre -> (código, final del nombre oficial). Verificados contra la API en noviembre de 2026.
BCRP = {
    "cobre": ("PN01652XM", "Cobre - LME (¢US$ por libras)"),
    "tipo_cambio": ("PN01207PM", "Interbancario - Promedio"),
    "pbi_sa": ("PN01773AM", "(índice 2007 = 100) - PBI Desestacionalizado - mensual"),
    "ipc": ("PN38705PM", "(índice Dic.2021 = 100) - Índice de Precios al Consumidor (IPC)"),
    "embig": ("PN01129XM", "(EMBIG) - Perú"),
    "tasa_referencia": ("PD04722MM", "Tasa de Referencia de la Política Monetaria"),
}
FRED = {"vix": "VIXCLS", "dolar": "DTWEXBGS", "fed_funds": "FEDFUNDS"}
MESES = {"ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6, "jul": 7, "ago": 8, "set": 9, "sep": 9,
         "oct": 10, "nov": 11, "dic": 12}


def _pedir(url: str, intentos: int = 5) -> requests.Response:
    ultimo: Exception | None = None
    for i in range(intentos):
        try:
            r = requests.get(url, timeout=40)
            r.raise_for_status()
            return r
        except requests.RequestException as e:      # ambos servidores cortan conexiones con frecuencia
            ultimo = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"Sin respuesta tras {intentos} intentos: {url}") from ultimo


def _vigente(archivo: Path, horas: float) -> bool:
    return archivo.exists() and time.time() - archivo.stat().st_mtime < horas * 3600


def serie_bcrp(nombre: str, inicio: str = "2003-1") -> pd.Series:
    codigo, final = BCRP[nombre]
    js = _pedir(BCRP_URL.format(codigo=codigo, inicio=inicio, fin=f"{date.today().year}-{date.today().month}")).json()
    oficial = js["config"]["series"][0]["name"].strip()
    if not oficial.lower().endswith(final.lower()):
        raise RuntimeError(f"'{nombre}' ({codigo}) devolvió «{oficial}»; se esperaba una serie que termine en «{final}».")
    fechas = [pd.Timestamp(year=int(p["name"].split(".")[1]), month=MESES[p["name"][:3].lower()], day=1) for p in js["periods"]]
    valores = [pd.to_numeric(p["values"][0], errors="coerce") for p in js["periods"]]
    return pd.Series(valores, index=pd.DatetimeIndex(fechas, name="fecha"), name=nombre, dtype=float)


def serie_fred(nombre: str) -> pd.Series:
    """Promedio mensual de una serie de FRED (las diarias se promedian)."""
    texto = _pedir(FRED_URL.format(serie=FRED[nombre])).text
    df = pd.read_csv(io.StringIO(texto), parse_dates=[0], index_col=0, na_values=".")
    return df.iloc[:, 0].resample("MS").mean().rename(nombre).rename_axis("fecha")


def cargar(max_horas: float = 24) -> pd.DataFrame:
    """Todas las series en un DataFrame mensual."""
    archivo = DATA / "series_mensuales.csv"
    if _vigente(archivo, max_horas):
        return pd.read_csv(archivo, index_col=0, parse_dates=True)
    series = [serie_bcrp(n) for n in BCRP] + [serie_fred(n) for n in FRED]
    df = pd.concat(series, axis=1).sort_index()
    DATA.mkdir(exist_ok=True)
    df.to_csv(archivo)
    return df
