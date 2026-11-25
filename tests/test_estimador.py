"""El estimador de proyecciones locales frente a casos de respuesta conocida."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lp.estimador import covarianza_robusta, proyeccion_local, rezagar  # noqa: E402
from lp.montecarlo import experimento, simular_ar1  # noqa: E402


def test_rezagar():
    s = np.arange(5.0)
    np.testing.assert_array_equal(rezagar(s, 1)[1:], [0, 1, 2, 3])
    np.testing.assert_array_equal(rezagar(s, -2)[:3], [2, 3, 4])
    assert np.isnan(rezagar(s, 1)[0]) and np.isnan(rezagar(s, -2)[-2:]).all()


def test_en_el_horizonte_cero_es_una_regresion_simple():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = 2.0 + 0.7 * x + rng.normal(size=500)
    res = proyeccion_local(y, x, horizontes=0, rezagos=0, metodo="hac")
    pendiente = np.cov(x, y)[0, 1] / np.var(x, ddof=1)
    assert res.irf[0] == pytest.approx(pendiente)
    assert res.n[0] == 500


def test_la_covarianza_coincide_con_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(1)
    Z = sm.add_constant(rng.normal(size=(300, 3)))
    u = np.convolve(rng.normal(size=304), np.ones(5) / 5, mode="valid")      # errores autocorrelacionados
    y = Z @ np.array([1.0, 0.5, -0.3, 0.2]) + u
    ajuste = sm.OLS(y, Z).fit()
    for rezagos in (0, 4, 12):
        tipo = {"cov_type": "HC1"} if rezagos == 0 else {"cov_type": "HAC", "cov_kwds": {"maxlags": rezagos, "use_correction": True}}
        referencia = sm.OLS(y, Z).fit(**tipo).cov_params()
        np.testing.assert_allclose(covarianza_robusta(Z, ajuste.resid, rezagos), referencia, rtol=1e-8)


@pytest.mark.parametrize("metodo", ["hac", "aumentada"])
def test_recupera_la_respuesta_de_un_ar1_en_muestra_grande(metodo):
    y = simular_ar1(0.8, 40_000, np.random.default_rng(2))
    res = proyeccion_local(y, y, horizontes=8, rezagos=0, metodo=metodo)
    np.testing.assert_allclose(res.irf, 0.8 ** np.arange(9), atol=0.03)


def test_respuesta_de_y_a_un_choque_en_x():
    """y_t = 0.6·y_{t-1} + 1.5·x_t + e_t con x ruido blanco: la respuesta en h es 1.5·0.6^h."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=30_000)
    y = np.zeros_like(x)
    for t in range(1, len(x)):
        y[t] = 0.6 * y[t - 1] + 1.5 * x[t] + rng.normal()
    res = proyeccion_local(y, x, horizontes=6, rezagos=2)
    np.testing.assert_allclose(res.irf, 1.5 * 0.6 ** np.arange(7), atol=0.04)


def test_la_version_acumulada_de_un_paseo_aleatorio_vale_uno():
    rng = np.random.default_rng(4)
    choque = rng.normal(size=20_000)
    nivel = np.cumsum(choque)
    res = proyeccion_local(nivel, choque, horizontes=5, rezagos=1, acumulada=True)
    np.testing.assert_allclose(res.irf, 1.0, atol=0.03)


def test_las_observaciones_excluidas_no_entran():
    rng = np.random.default_rng(5)
    y = simular_ar1(0.5, 400, rng)
    sucio = y.copy(); sucio[200] += 500                       # un dato absurdo
    marca = np.zeros(400, dtype=bool); marca[200] = True
    limpio = proyeccion_local(y, y, horizontes=4, rezagos=1)
    con_marca = proyeccion_local(sucio, sucio, horizontes=4, rezagos=1, excluir=marca)
    sin_marca = proyeccion_local(sucio, sucio, horizontes=4, rezagos=1)
    assert np.abs(con_marca.irf - limpio.irf).max() < 0.05
    assert np.abs(sin_marca.irf - limpio.irf)[1:].max() > 0.2
    assert (con_marca.n < limpio.n).all()


def test_la_cobertura_de_la_version_aumentada_es_cercana_a_la_nominal():
    t = experimento(0.5, T=240, horizontes=12, repeticiones=400, semilla=6)
    assert t.loc[[1, 6, 12], "cobertura_aumentada"].between(0.85, 0.95).all()
    assert t.loc[0, "verdadero"] == 1 and t.loc[3, "verdadero"] == pytest.approx(0.125)
