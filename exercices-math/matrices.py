"""Module contenant les fonctions utiles
à l'exercice 2 de la Phase1 du programme de data --> QML"""
import numpy as np
from vecteurs import produit_scalaire


def produit_matrice_vecteur(A, x):
    """Renvoie A x, de forme (m,), pour A de forme (m, n) et x de forme (n,).
    Lève ValueError si x n'est pas de forme (n,), y compris (n, 1)."""
    if A.shape[1] != x.shape[0]:
        raise ValueError(f"{A} et {x} n'ont pas une forme compatible")
    return np.array([produit_scalaire(ligne, x) for ligne in A])


def produit_matriciel(A, B):
    """Renvoie A B, de forme (m, p), pour A (m, n) et B (n, p).
    Lève ValueError si A n'a pas autant de colonnes que B a de lignes."""
    Ashape = A.shape
    Bshape = B.shape
    if Ashape[1] != Bshape[0]:
        raise ValueError(f"Formes incompatibles : A {Ashape} et B {Bshape}")
    colonnes = [produit_matrice_vecteur(A, B[:, j]) for j in range(Bshape[1])]
    return np.column_stack(colonnes)


def rotation(theta_deg):
    """Renvoie la matrice de rotation de forme (2, 2) d'angle theta_deg,
    en degrés, dans le sens antihoraire."""
    theta_rad = np.radians(theta_deg)
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([[c, -s],
                     [s, c]])
