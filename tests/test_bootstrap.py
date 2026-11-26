"""El bootstrap frente a casos de respuesta conocida."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lp import bootstrap as bs  # noqa: E402
from lp.estimador import proyeccion_local  # noqa: E402
from lp.montecarlo import experimento, simular_ar1  # noqa: E402


def simular_var(A, sigma, T, rng):
    k = A.shape[0]
    innovaciones = rng.standard_normal((T, k)) @ np.linalg.cholesky(sigma).T
    Y = np.zeros((T, k))
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + innovaciones[t]
    return Y


def test_var_mco_recupera_los_coeficientes():
    A = np.array([[0.5, 0.1], [0.3, 0.4]])
    Y = 2.0 + simular_var(A, np.eye(2), 40_000, np.random.default_rng(0))
    c, estimada, U = bs.var_mco(Y, 1)
    np.testing.assert_allclose(estimada[0], A, atol=0.02)
    np.testing.assert_allclose(U.mean(axis=0), 0, atol=1e-10)
    np.testing.assert_allclose(c, (np.eye(2) - A) @ [2.0, 2.0], atol=0.05)


def test_la_correccion_de_sesgo_reproduce_la_formula_de_kendall():
    """En un AR(1) el sesgo de MCO es -(1 + 3·rho)/T: con rho = 0.9 y T = 100, 0.9 pasa a 0.937."""
    U = np.random.default_rng(1).normal(size=(300, 1))
    assert bs.corregir_sesgo(np.array([[[0.9]]]), U, 100)[0, 0, 0] == pytest.approx(0.937)
    assert bs.corregir_sesgo(np.array([[[0.5]]]), 3 * U, 240)[0, 0, 0] == pytest.approx(0.5 + 2.5 / 240)


def test_la_correccion_no_cruza_la_raiz_unitaria():
    U = np.random.default_rng(1).normal(size=(50, 1))
    corregido = bs.corregir_sesgo(np.array([[[0.99]]]), U, 50)       # sin salvaguarda daría 1.069
    assert 0.99 < corregido[0, 0, 0] < 1
    explosivo = np.array([[[1.02]]])
    assert bs.corregir_sesgo(explosivo, U, 50) is explosivo           # un VAR no estacionario no se toca


def test_respuesta_del_var_calculada_a_mano():
    assert bs.respuesta_var(np.array([[[0.8]]]), np.array([[2.0]]), 0, 0, 4) == pytest.approx(0.8 ** np.arange(5))
    A = np.array([[[0.5, 0.0], [0.3, 0.4]]])
    sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
    # impacto = cov / var = 0.5; después, las potencias de A aplicadas al vector (1, 0.5)
    esperado = [(np.linalg.matrix_power(A[0], h) @ [1.0, 0.5])[1] for h in range(4)]
    assert bs.respuesta_var(A, sigma, 1, 0, 3) == pytest.approx(esperado)
    assert esperado[:2] == pytest.approx([0.5, 0.5])


def test_la_proyeccion_local_estima_la_respuesta_del_var():
    """Con un control contemporáneo, la proyección local estima la respuesta a la innovación del
    impulso que el control no explica. En una muestra larga debe coincidir con `respuesta_var`."""
    A = np.array([[0.5, 0.1, 0.0], [0.2, 0.4, 0.1], [0.1, 0.3, 0.6]])
    sigma = np.array([[1.0, 0.4, 0.2], [0.4, 1.0, 0.5], [0.2, 0.5, 1.0]])
    Y = simular_var(A, sigma, 80_000, np.random.default_rng(2))
    lp = proyeccion_local(Y[:, 2], Y[:, 1], Y[:, [0]], horizontes=6, rezagos=1, metodo="aumentada")
    np.testing.assert_allclose(lp.irf, bs.respuesta_var(A[None], sigma, 2, 1, 6), atol=0.03)


@pytest.mark.parametrize("rezagos", [0, 2])
def test_la_version_vectorizada_coincide_con_la_de_referencia(rezagos):
    rng = np.random.default_rng(3)
    lote = np.stack([simular_ar1(0.9, 200, rng) for _ in range(3)])
    irf, ee = bs.lp_propia_lote(lote, 10, rezagos)
    for i, y in enumerate(lote):
        referencia = proyeccion_local(y, y, horizontes=10, rezagos=rezagos, metodo="aumentada")
        np.testing.assert_allclose(irf[i], referencia.irf, atol=1e-10)
        np.testing.assert_allclose(ee[i], referencia.ee, atol=1e-10)


def test_las_muestras_simuladas_siguen_al_var():
    """Con residuos nulos la simulación es determinista: cada serie decae como rho^t desde su inicio."""
    Y = np.arange(1.0, 11.0)[:, None]
    sim = bs.simular_lote(np.zeros(1), np.array([[[0.5]]]), np.zeros((6, 1)), Y, 4, np.random.default_rng(0))
    assert sim.shape == (4, 7, 1)
    np.testing.assert_allclose(sim[:, 1:, 0], sim[:, [0], 0] * 0.5 ** np.arange(1, 7))
    assert set(sim[:, 0, 0]) <= set(Y[:, 0])                          # el arranque sale de los datos


def test_el_intervalo_percentil_t_usa_los_cuantiles_al_reves():
    """Si t* está corrido hacia abajo (el estimador subestima), el intervalo se corre hacia arriba."""
    t_estrella = np.linspace(-3.0, 1.0, 401)[:, None]
    inferior, superior = bs.percentil_t(np.array([1.0]), np.array([0.1]), t_estrella, 0.90)
    assert inferior[0] == pytest.approx(1.0 - 0.1 * 0.8) and superior[0] == pytest.approx(1.0 + 0.1 * 2.8)


def test_el_bootstrap_general_es_reproducible_y_se_parece_al_rapido():
    y = simular_ar1(0.9, 240, np.random.default_rng(4))
    res, inferior, superior = bs.banda_bootstrap(y, y, horizontes=8, rezagos=0, B=199, semilla=1)
    _, inferior_2, superior_2 = bs.banda_bootstrap(y, y, horizontes=8, rezagos=0, B=199, semilla=1)
    np.testing.assert_array_equal(inferior, inferior_2)
    np.testing.assert_array_equal(superior, superior_2)
    assert (inferior[1:] < res.irf[1:]).all() and (res.irf[1:] < superior[1:]).all()
    _, rapido_inf, rapido_sup = bs.banda_bootstrap_propia(y, 8, 0, B=999, rng=np.random.default_rng(2))
    np.testing.assert_allclose(inferior[1:], rapido_inf[1:], atol=0.06)   # mismas reglas, otros sorteos
    np.testing.assert_allclose(superior[1:], rapido_sup[1:], atol=0.06)


def test_el_bootstrap_con_controles_y_periodos_excluidos():
    rng = np.random.default_rng(5)
    Y = simular_var(np.array([[0.5, 0.1, 0.0], [0.2, 0.4, 0.1], [0.1, 0.3, 0.6]]), np.eye(3), 260, rng)
    marca = np.zeros(260, dtype=bool)
    marca[100:104] = True
    Y[100:104, 2] += 50                                               # cuatro meses absurdos, marcados
    _, inferior, superior = bs.banda_bootstrap(Y[:, 2], Y[:, 1], Y[:, [0]], horizontes=6, rezagos=2,
                                               excluir=marca, B=99, semilla=0)
    assert np.isfinite(inferior).all() and np.isfinite(superior).all()
    assert (superior - inferior < 2.0).all()                          # los meses marcados no contaminan el VAR
    with pytest.raises(ValueError, match="al menos un rezago"):
        bs.banda_bootstrap(Y[:, 2], Y[:, 1], horizontes=2, rezagos=0, B=9)


def test_el_bootstrap_mejora_la_cobertura_con_series_persistentes():
    t = experimento(0.95, T=160, horizontes=16, repeticiones=160, semilla=7, B=199)
    lejos = t.loc[8:16]
    assert lejos["cobertura_bootstrap"].mean() > lejos["cobertura_aumentada"].mean() > lejos["cobertura_hac"].mean()
    assert lejos["cobertura_bootstrap"].mean() > 0.80
    assert (lejos["ancho_bootstrap"] > lejos["ancho_aumentada"]).all()


def test_el_resultado_no_depende_del_numero_de_procesos():
    uno = experimento(0.5, T=80, horizontes=4, repeticiones=24, semilla=3, B=49, procesos=1)
    dos = experimento(0.5, T=80, horizontes=4, repeticiones=24, semilla=3, B=49, procesos=2)
    np.testing.assert_allclose(uno.to_numpy(), dos.to_numpy())
