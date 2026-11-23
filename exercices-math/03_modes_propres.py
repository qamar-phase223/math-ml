"""Script de l'exercice 3 Phase 1"""
import numpy as np
from propres import est_symetrique, matrice_raideur, valeurs_propres
from propres import vecteur_propre, valeurs_singulieres, pulsations
from vecteurs import produit_scalaire


def main():
    """Calcule et affiche la sortie de l'exercice 1.3 (section 6)."""
    systeme1 = np.array([3.0, 1.0, 3.0, 1.0], dtype="float64")  # [k1,k2,k3,m]
    systeme2 = np.array([1.0, 2.0, 4.0, 1.0], dtype="float64")
    systeme1_lourd = np.array([3.0, 1.0, 3.0, 4.0], dtype="float64")
    A = np.array([[4.0, 0.0], [3.0, -5.0]], dtype="float64")
    A_s = np.array([[9.0, -3.0], [-3.0, 1.0]], dtype="float64")
    R = np.array([[0.0, -1.0], [1.0, 0.0]], dtype="float64")
    D = np.array([[2.0, 0.0], [0.0, 5.0]], dtype="float64")

    print("Exercice 1.3 - Valeurs propres, modes propres et SVD\n")

    print(f"A. Système 1 (k1 = {systeme1[0]:.4f}, k2 = {systeme1[1]:.4f}, "
          f"k3 = {systeme1[2]:.4f} N/m ; m = {systeme1[3]:.4f} kg)")
    K1 = matrice_raideur(systeme1[0], systeme1[1], systeme1[2])
    afficher_matrice("K", K1)
    print(f"K symétrique : {formater_bool(est_symetrique(K1))}")
    lambda_sys1 = valeurs_propres(K1)
    K1mode1 = vecteur_propre(K1, lambda_sys1[0])
    K1mode2 = vecteur_propre(K1, lambda_sys1[1])
    omega_sys1 = pulsations(systeme1[0], systeme1[1], systeme1[2], systeme1[3])
    print(f"lambda = {formater_couple(lambda_sys1)} N/m")
    print(f"omega = {formater_couple(omega_sys1)} rad/s")
    print(f"mode 1 = {formater_vecteur(K1mode1)}")
    print(f"mode 2 = {formater_vecteur(K1mode2)}")
    print(f"mode 1 . mode 2 = "
          f"{formater_nombre(produit_scalaire(K1mode1, K1mode2))}")

    print(f"\nB. Système 2 (k1 = {systeme2[0]:.4f}, k2 = {systeme2[1]:.4f}, "
          f"k3 = {systeme2[2]:.4f} N/m ; m = {systeme2[3]:.4f} kg)")
    K2 = matrice_raideur(systeme2[0], systeme2[1], systeme2[2])
    afficher_matrice("K", K2)
    print(f"K symétrique : {formater_bool(est_symetrique(K2))}")
    lambda_sys2 = valeurs_propres(K2)
    K2mode1 = vecteur_propre(K2, lambda_sys2[0])
    K2mode2 = vecteur_propre(K2, lambda_sys2[1])
    omega_sys2 = pulsations(systeme2[0], systeme2[1], systeme2[2], systeme2[3])
    print(f"lambda = {formater_couple(lambda_sys2)} N/m")
    print(f"omega = {formater_couple(omega_sys2)} rad/s")
    print(f"mode 1 = {formater_vecteur(K2mode1)}")
    print(f"mode 2 = {formater_vecteur(K2mode2)}")
    print(f"mode 1 . mode 2 = "
          f"{formater_nombre(produit_scalaire(K2mode1, K2mode2))}")

    print(f"\nC. Système 1 lourd (m = {systeme1_lourd[3]:.4f} kg)")
    k1_lourd = systeme1_lourd[0]
    k2_lourd = systeme1_lourd[1]
    k3_lourd = systeme1_lourd[2]
    m_lourd = systeme1_lourd[3]
    omega_sys1_lourd = pulsations(k1_lourd, k2_lourd, k3_lourd, m_lourd)
    print(f"omega = {formater_couple(omega_sys1_lourd)} rad/s")

    print("\nD. Valeurs propres et valeurs singulières")
    print(f"D : lambda = {formater_couple(valeurs_propres(D))} | "
          f"sigma = {formater_couple(valeurs_singulieres(D))}")
    try:
        valeurs_propres(R)
    except ValueError:
        print(f"R : lambda = aucune valeur réelle | "
              f"sigma = {formater_couple(valeurs_singulieres(R))}")
    else:
        print("R : anomalie car valeurs propres réelles inattendues")
    lamAs = valeurs_propres(A_s)
    sigmaAs = valeurs_singulieres(A_s)
    lamA = valeurs_propres(A)
    sigmaA = valeurs_singulieres(A)
    v1 = vecteur_propre(A, lamA[0])
    v2 = vecteur_propre(A, lamA[1])
    print(f"A_s : lambda = {formater_couple(lamAs)} | "
          f"sigma = {formater_couple(sigmaAs)}")
    print(f"A : lambda = {formater_couple(lamA)} | "
          f"sigma = {formater_couple(sigmaA)}")
    print(f"A : v1 . v2 = {formater_nombre(produit_scalaire(v1, v2))}")

    print("\nE. Contrôles sur A")
    a = A[0, 0]
    b = A[0, 1]
    c = A[1, 0]
    d = A[1, 1]
    lam = valeurs_propres(A)
    sigma = valeurs_singulieres(A)
    trace = a + d
    det = a*d - b*c
    trace_ok = np.isclose(trace, lam[0] + lam[1], rtol=0.0, atol=1e-12)
    det_ok = np.isclose(det, lam[0] * lam[1], rtol=0.0, atol=1e-12)
    sigma_ok = np.isclose(sigma[0] * sigma[1], abs(det), rtol=0.0, atol=1e-12)
    print(f"trace = somme des lambda : {formater_bool(trace_ok)}")
    print(f"det = produit des lambda : {formater_bool(det_ok)}")
    print(f"produit des sigma = |det| : {formater_bool(sigma_ok)}")


def formater_nombre(x):
    """Renvoie x à 4 décimales, sans jamais afficher -0.00."""
    return f"{round(x, 4) + 0.0:.4f}"


def formater_vecteur(v):
    """Renvoie un vecteur sous la forme (a, b, ...)."""
    return "(" + ", ".join(formater_nombre(x) for x in v) + ")"


def formater_couple(v):
    """Renvoie deux valeurs d'un même couple séparées par ;"""
    return " ; ".join(formater_nombre(x) for x in v)


def afficher_matrice(nom, M):
    """Affiche le nom suivi de la matrice, une ligne par ligne de M."""
    print(f"{nom} =")
    for ligne in M:
        print("  [" + ", ".join(formater_nombre(x) for x in ligne) + "]")


def formater_bool(b):
    """Renvoie 'vrai' ou 'faux' selon la valeur de b."""
    return "vrai" if b else "faux"


if __name__ == "__main__":
    main()
