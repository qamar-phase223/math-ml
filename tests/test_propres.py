"""Contient les tests pour l'exercice 3 phase 1 du programme de data -> QML"""
import pytest
import numpy as np
from propres import est_symetrique, matrice_raideur, valeurs_propres
from propres import vecteur_propre, valeurs_singulieres, pulsations
from vecteurs import produit_scalaire, norme


k11, k12, k1L = 3.0, 1.0, 3.0
k21, k22, k2L = 1.0, 2.0, 1.0
k31, k32, k3L = 3.0, 4.0, 3.0
m1, m1L = 1.0, 4.0
K1 = matrice_raideur(k11, k21, k31)
K2 = matrice_raideur(k12, k22, k32)
A = np.array([[4.0, 0.0], [3.0, -5.0]], dtype="float64")
A_s = np.array([[9.0, -3.0], [-3.0, 1.0]], dtype="float64")
R = np.array([[0.0, -1.0], [1.0, 0.0]], dtype="float64")
D = np.array([[2.0, 0.0], [0.0, 5.0]], dtype="float64")


@pytest.mark.parametrize("x, y, z, attendu", [
    (k11, k21, k31, np.array([[4, -1], [-1, 4]], dtype="float64")),
    (k12, k22, k32, np.array([[3, -2], [-2, 6]], dtype="float64")),
], ids=["Système 1", "Système 2"]
)
def test_matrice_raideur(x, y, z, attendu):
    assert matrice_raideur(x, y, z) == pytest.approx(attendu, abs=1e-12)


@pytest.mark.parametrize("matrice, attendu", [
    (K1, True),
    (A, False)
], ids=["Symétrie K1", "Symétrie A"]
)
def test_symetrie(matrice, attendu):
    assert est_symetrique(matrice) is attendu


@pytest.mark.parametrize("matrice, attendu", [
    (K1, (3.0, 5.0)),
    (K2, (2.0, 7.0)),
    (A, (-5.0, 4.0)),
    (A_s, (0.0, 10.0))
], ids=["lambda K1", "lambda K2", "lambda A", "lambda A_s"])
def test_valeurs_propres(matrice, attendu):
    assert valeurs_propres(matrice) == pytest.approx(attendu, abs=1e-12)


@pytest.mark.parametrize("matrice, lam", [
    (K2, 2.0),
    (K2, 7.0),
    (A, 4.0),
    (A, -5.0)
], ids=["K2_lamIs2", "K2_lamIs7", "A_lamIs4", "A_lamIs-5"]
)
def test_vecteur_propre(matrice, lam):
    v = vecteur_propre(matrice, lam)
    assert norme(v) == pytest.approx(1.0, abs=1e-12)
    assert matrice @ v == pytest.approx(lam * v, abs=1e-12)


def test_ortho_vecteurs_propres_K2():
    v1 = vecteur_propre(K2, 2.0)
    v2 = vecteur_propre(K2, 7.0)
    assert produit_scalaire(v1, v2) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("matrice, sigmas", [
    (D, np.array((5.0, 2.0))),
    (R, np.array((1.0, 1.0))),
    (A_s, np.array((10.0, 0.0)))
], ids=["sigmasD", "sigmasR", "sigmasA_s"]
    )
def test_valeurs_singulieres(matrice, sigmas):
    assert valeurs_singulieres(matrice) == pytest.approx(sigmas, abs=1e-12)


@pytest.mark.parametrize("k1, k2, k3, m, attendu", [
    (k11, k21, k31, m1, np.sqrt((3, 5))),
    (k1L, k2L, k3L, m1L, np.sqrt((3, 5))/2)
], ids=["pulsations_S1", "pulsations_S1L"]
)
def test_pulsations(k1, k2, k3, m, attendu):
    assert pulsations(k1, k2, k3, m) == pytest.approx(attendu, abs=1e-12)


def test_concordance1():
    """valeurs_propres(K2) égale np.linalg.eigvalsh(K2)."""
    assert valeurs_propres(K2) == pytest.approx(np.linalg.eigvalsh(K2), abs=1e-12)


def test_concordance2():
    """valeurs_singulieres(A) égale np.linalg.svd(A)."""
    assert valeurs_singulieres(A) == pytest.approx(np.linalg.svd(A, compute_uv=False), abs=1e-12)


def test_exception1():
    with pytest.raises(ValueError):
        valeurs_propres(R)


def test_exception2():
    with pytest.raises(ValueError):
        valeurs_propres([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
