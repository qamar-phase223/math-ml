"""Module de fonctions de l'exercice 3 phase 1"""
import numpy as np
from matrices import produit_matriciel
from vecteurs import produit_scalaire


def est_symetrique(A):
    """Reçoit A (2, 2) et renvoie True (False)
    si A symétrique (non-symétrique)"""
    A = np.array(A, dtype="float64")
    if A.shape != (2, 2):
        raise ValueError
    return bool(np.allclose(A, A.T, rtol=0.0, atol=1e-12))


def matrice_raideur(k1, k2, k3):
    """Reçoit k1, k2, k3 (N/m, > 0) ; ValueError sinon.
    Renvoie la matrice de raideur K (2, 2), en N/m."""
    if not np.all(np.array([k1, k2, k3]) > 0):
        raise ValueError
    K = np.array([[k1+k2, -k2], [-k2, k2+k3]], dtype="float64")
    return K


def valeurs_propres(A):
    """reçoit A (2,2)
    renvoie ses valeurs propres, (2,) ordre croissant"""
    A = np.array(A, dtype="float64")
    if A.shape != (2, 2):
        raise ValueError
    a = A[0, 0]
    b = A[0, 1]
    c = A[1, 0]
    d = A[1, 1]
    delta = (a - d)**2 + 4*b*c
    if delta < 0:
        raise ValueError
    l1 = (a + d - np.sqrt(delta))/2
    l2 = (a + d + np.sqrt(delta))/2
    return np.array([l1, l2])


def vecteur_propre(A, lam):
    """reçoit A (2, 2) et lam (valeur propre, float64)
    renvoie le vecteur propre unitaire (2,)
    dont la première composante non nulle est positive."""
    A = np.array(A, dtype="float64")
    if A.shape != (2, 2):
        raise ValueError
    x, y = 0, 0
    a_lam = A[0, 0] - lam
    b = A[0, 1]
    c = A[1, 0]
    d_lam = A[1, 1] - lam
    if not np.isclose(b, 0.0):
        x = 1
        y = -a_lam/b
    else:
        x = 0
        y = 1
    if np.isclose(a_lam, 0.0) and np.isclose(b, 0.0):
        if not np.isclose(d_lam, 0.0):
            x = 1
            y = c/-d_lam
        else:
            x = 0
            y = 1
    vect = np.array((x, y))
    norme_vect = np.sqrt(produit_scalaire(vect, vect))
    vect_unit = vect/norme_vect + 0.0
    return vect_unit


def valeurs_singulieres(A):
    """reçoit A (2,2)
    renvoie ses valeurs singulières (2,), décroissantes, >= 0.0"""
    A = np.array(A, dtype="float64")
    if A.shape != (2, 2):
        raise ValueError
    AtA = produit_matriciel(A.T, A)
    lam1, lam2 = valeurs_propres(AtA)
    sigma1 = np.sqrt(max(lam2, 0.0))
    sigma2 = np.sqrt(max(lam1, 0.0))
    return np.array([sigma1, sigma2])


def pulsations(k1, k2, k3, m):
    """Reçoit k1, k2, k3 (N/m, > 0) et m (kg, > 0) ; ValueError sinon.
    Renvoie les deux pulsations propres (2,), en rad/s, croissantes."""
    if not m > 0:
        raise ValueError
    K = matrice_raideur(k1, k2, k3)
    lam1, lam2 = valeurs_propres(K)
    w1, w2 = np.sqrt(lam1/m), np.sqrt(lam2/m)
    return np.array([w1, w2])


