"""Script contenant les fonctions utiles
à l'exercice 2 de la Phase1 du programme de data --> QML"""
import numpy as np
from vecteurs import produit_scalaire


def produit_matrice_vecteur(A, x):
    """Reçoit A : ndarray (m, n) et x : ndarray (n,)
    Renvoie Ax : ndarray (m,)"""
    if A.shape[1] != x.shape[0]:
        raise ValueError(f"{A} et {x} n'ont pas une forme compatible")
    return np.array([produit_scalaire(ligne, x) for ligne in A])


def produit_matriciel(A, B):
    """Calcule le produit matriciel AB"""
    Ashape = A.shape
    Bshape = B.shape
    if Ashape[1] != Bshape[0]:
        raise ValueError(f"Formes incompatibles : A {Ashape} et B {Bshape}")
    colonnes = [produit_matrice_vecteur(A, B[:, j]) for j in range(Bshape[1])]
    return np.column_stack(colonnes)


def rotation(theta_deg):
    """Renvoie la matrice de rotation d'angle theta_deg
    sens antihoraire; forme (2,2)"""
    theta_rad = np.radians(theta_deg)
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([[c, -s],
                     [s, c]])
