"""Contient les tests pour l'exercice 1 phase 1 du programme de data -> QML"""
import pytest
import numpy as np
from vecteurs import produit_scalaire, norme, angle_degres, travail


d = np.array([5, 0], dtype=np.float64)
d4 = np.array([5, 1], dtype=np.float64)
F1 = np.array([6, 8], dtype=np.float64)
F2 = np.array([0, -50], dtype=np.float64)
F3 = np.array([-6, 0], dtype=np.float64)
F4 = np.array([10, 2], dtype=np.float64)
u = np.array([2, -1, 3], dtype=np.float64)
v = np.array([4, 2, 1], dtype=np.float64)


@pytest.mark.parametrize("x, y, attendu", [
    (F1, d, 30.00),
    (F2, d, 00.00),
    (u, v, 9.00)
])
def test_produit_scalaire(x, y, attendu):
    assert produit_scalaire(x, y) == pytest.approx(attendu)


def test_echec_produit_scalaire():
    x = np.array([1, 2], dtype=np.float64)
    y = np.array([1, 2, 3], dtype=np.float64)
    with pytest.raises(ValueError):
        produit_scalaire(x, y)


def test_norme():
    assert norme(F1) == pytest.approx(10.00, abs=0.01)


@pytest.mark.parametrize("x, y, attendu", [
    (F1, d, 53.13),
    (F3, d, 180.00),
    (F4, d4, 0.00)
])
def test_angles_degres(x, y, attendu):
    assert angle_degres(x, y) == pytest.approx(attendu, abs=0.01)


def test_angles_nuls_degress():
    with pytest.raises(ValueError):
        angle_degres(np.array([0, 0], dtype=np.float64), d)


def test_travail():
    assert travail(F4, d4) == pytest.approx(52.00)
