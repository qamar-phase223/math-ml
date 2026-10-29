"""Tests pour l'exercice 2 phase 1 du programme de data -> QML"""
import pytest
import numpy as np
from matrices import produit_matrice_vecteur, produit_matriciel, rotation

p0 = np.array((0.0, 0.0), dtype=np.float64)
p1 = np.array((1.0, 0.0), dtype=np.float64)
p2 = np.array((1.0, 1.0), dtype=np.float64)
p3 = np.array((0.0, 1.0), dtype=np.float64)
P = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]],
             dtype=np.float64)
R = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)
E = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float64)
C = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
X = np.array([[1, 0, 2], [3, 1, 0], [0, 2, 1], [2, 2, 2]],
             dtype=np.float64)
w = np.array((2.0, 3.0, 5.0), dtype=np.float64)
RTR = produit_matriciel(R.T, R)
A = np.array([[1.5, -2.0], [0.5, 4.0], [-3.0, 1.0]])
B = np.array([[2.0, 0.0, -1.0, 3.5], [1.0, -2.5, 0.5, 1.0]])


@pytest.mark.parametrize("X, v, attendu", [
    (R, p2, np.array([-1.0, 1.0])),
    (E, p2, np.array([2.0, 1.0])),
    (C, p3, np.array([1.0, 1.0])),
    (X, w, np.array([12.0, 9.0, 11.0, 20.0]))
], ids=["Rp2", "Ep2", "Cp3", "Xw"]
)
def test_produit_matrice_vecteur(X, v, attendu):
    assert produit_matrice_vecteur(X, v) == pytest.approx(attendu, abs=1e-9)


@pytest.mark.parametrize("M, N, attendu", [
    (E, P, np.array([[0.0, 2.0, 2.0, 0.0], [0.0, 0.0, 1.0, 1.0]])),
    (E, C, np.array([[2.0, 2.0], [0.0, 1.0]])),
    (C, E, np.array([[2.0, 1.0], [0.0, 1.0]])),
    (RTR, R, np.array([[0.0, -1.0], [1.0, 0.0]]))
], ids=["EP", "EC", "CE", "R^TR"]
)
def test_produit_matriciel(M, N, attendu):
    assert produit_matriciel(M, N) == pytest.approx(attendu, abs=1e-9)


def test_concordance_numpy():
    assert produit_matriciel(A, B) == pytest.approx(A@B)


@pytest.mark.parametrize("angle, attendu", [
    (90.0, np.array([[0.0, -1.0], [1.0, 0.0]])),
    (0.0, np.eye(2))
], ids=["rotation(90)", "rotation(0.0)"]
)
def test_rotation(angle, attendu):
    assert rotation(angle) == pytest.approx(attendu)


def test_non_commutativite():
    assert not np.allclose(produit_matriciel(E, C), produit_matriciel(C, E))


def test_composition_rotation():
    assert produit_matriciel(rotation(30), rotation(60)) == pytest.approx(rotation(90))


def test_exception1():
    with pytest.raises(ValueError):
        produit_matrice_vecteur(X, np.array([1.0, 2.0]))


def test_exception2():
    with pytest.raises(ValueError):
        produit_matrice_vecteur(X, w.reshape(3, 1))


def test_exception3():
    with pytest.raises(ValueError):
        produit_matriciel(E, X)
