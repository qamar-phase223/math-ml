"""Module de fonctions de l'exercice 1 phase 1"""
import numpy as np


def produit_scalaire(u, v):
    """Calcule le produit scalaire entre u et v."""
    u = np.array(u)
    v = np.array(v)
    if u.shape != v.shape:
        raise ValueError(f"Les vecteurs {u} et {v} n'ont pas la même taille")
    return np.sum(u*v)


def norme(u):
    """Calcule la norme du vecteur u."""
    return np.sqrt(produit_scalaire(u, u))


def angle_degres(u, v):
    """Calcule l'angle entre u et v en degres."""
    if norme(u) == 0.0 or norme(v) == 0.0:
        raise ValueError(f"Un des vecteurs est nul : {u} ou {v}")
    return np.acos(produit_scalaire(u, v)/(norme(u)*norme(v)))*(180/np.pi)


def travail(F, d):
    """Calcule le travail de la force F sur le déplacement d."""
    return produit_scalaire(F, d)
