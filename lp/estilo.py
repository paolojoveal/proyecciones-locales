"""Estilo gráfico de las figuras (matplotlib).

    - El color cumple una función: identidad, magnitud o estado. Nunca es decoración.
    - Barras delgadas, grilla tenue y continua, un solo eje por gráfico.
    - El texto va en tinta neutra; el color lo llevan las marcas.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import Formatter, FuncFormatter

# ---------- superficie y tintas ----------
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"        # títulos y valores
TINTA_2 = "#52514e"      # ejes, leyendas y etiquetas
TENUE = "#898781"        # notas al pie
GRILLA = "#e1e0d9"
EJE = "#c3c2b7"
CONTEXTO = "#b5b3ab"     # series que solo dan contexto (referencias, activos individuales)

# ---------- color categórico: orden fijo, no se recicla ----------
AZUL, NARANJA, AGUA, AMARILLO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, VERDE, VIOLETA, ROJO = "#e87ba4", "#008300", "#6250d6", "#e34948"
SERIE = [AZUL, NARANJA, AGUA, AMARILLO, MAGENTA, VERDE, VIOLETA, ROJO]

# ---------- secuencial (magnitud) y divergente (signo, con gris neutro al centro) ----------
AZUL_SECUENCIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
MAPA_SECUENCIAL = LinearSegmentedColormap.from_list("azul", AZUL_SECUENCIAL)
MAPA_DIVERGENTE = LinearSegmentedColormap.from_list(
    "azul_rojo", ["#184f95", "#6da7ec", "#f0efec", "#ef8f8e", "#b52d2d"])

# ---------- estado: reservado para bueno / alerta / crítico ----------
ESTADO = {"bien": "#0ca30c", "alerta": "#fab219", "serio": "#ec835a", "critico": "#d03b3b"}

# ---------- tamaños de figura (pulgadas) ----------
ANCHO = (9.0, 4.6)       # una serie de tiempo o un gráfico principal
DOBLE = (11.0, 4.4)      # dos paneles lado a lado
CUADRADO = (6.8, 5.2)


def aplicar() -> None:
    """Fija los parámetros de matplotlib. Llamar una vez, antes de crear figuras."""
    plt.switch_backend("Agg")        # las figuras solo se guardan en archivo: no hace falta una ventana
    mpl.rcParams.update({
        "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE, "savefig.dpi": 170,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "text.color": TINTA, "axes.labelcolor": TINTA_2, "axes.titlecolor": TINTA,
        "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9.5,
        "axes.edgecolor": EJE, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
        "grid.color": GRILLA, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": EJE, "ytick.color": EJE,
        "xtick.labelcolor": TINTA_2, "ytick.labelcolor": TINTA_2,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "xtick.major.size": 3, "ytick.major.size": 0,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "lines.markersize": 7,
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": TINTA_2,
        "axes.prop_cycle": cycler(color=SERIE),
    })


def titulo(ax, texto: str, subtitulo: str | None = None) -> None:
    """Título alineado a la izquierda y, debajo, un subtítulo que dice cómo leer el gráfico."""
    ax.set_title(texto, loc="left", pad=24 if subtitulo else 10)
    if subtitulo:
        ax.text(0, 1.025, subtitulo, transform=ax.transAxes, fontsize=9.5,
                color=TINTA_2, va="bottom", ha="left")


def punto_final(ax, x, y, color: str) -> None:
    """Marca el último dato de una línea, con un anillo del color de la superficie."""
    ax.plot([x], [y], "o", ms=7, color=color, mec=SUPERFICIE, mew=2, zorder=5, clip_on=False)


def etiqueta(ax, x, y, texto: str, dx: float = 7, dy: float = 0, **kw) -> None:
    """Etiqueta directa junto a una marca. El texto va en tinta neutra, nunca en el color de la serie."""
    opciones = {"fontsize": 9, "color": TINTA_2, "va": "center", "ha": "left"}
    opciones.update(kw)
    ax.annotate(texto, (x, y), xytext=(dx, dy), textcoords="offset points",
                annotation_clip=False, **opciones)


class _Porcentaje(Formatter):
    """Etiquetas de porcentaje con los decimales que pide el paso entre marcas: nunca muestra
    '2%' donde la marca está en 2.5%."""

    def __init__(self, factor: float, decimales: int | None):
        self.factor, self.decimales = factor, decimales

    def __call__(self, x, pos=None):
        d = self.decimales
        if d is None:
            paso = abs(float(self.locs[1] - self.locs[0])) * self.factor if len(self.locs) > 1 else 1.0
            d = 0 if abs(paso - round(paso)) < 1e-6 and paso >= 1 else (1 if abs(paso * 10 - round(paso * 10)) < 1e-6 else 2)
        return f"{x * self.factor:.{d}f}%".replace("-", "\u2212")


def formato_pct(ax, eje: str = "y", decimales: int | None = None, factor: float = 1.0) -> None:
    """Formatea un eje como porcentaje. `factor=100` si los datos están en proporción (0.05 -> 5%)."""
    (ax.yaxis if eje == "y" else ax.xaxis).set_major_formatter(_Porcentaje(factor, decimales))


def formato_miles(ax, eje: str = "y", decimales: int = 0) -> None:
    fmt = FuncFormatter(lambda v, _: f"{v:,.{decimales}f}")
    (ax.yaxis if eje == "y" else ax.xaxis).set_major_formatter(fmt)


def guardar(fig, ruta: Path | str, pie: str | None = None) -> Path:
    """Ajusta márgenes, agrega la nota al pie (fuente o advertencia) y guarda la figura."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.045 if pie else 0, 1, 1))
    if pie:
        fig.text(0.012, 0.012, pie, fontsize=8, color=TENUE, ha="left", va="bottom")
    fig.savefig(ruta)
    plt.close(fig)
    return ruta
