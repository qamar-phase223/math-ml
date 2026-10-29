"""Script de l'exercice 2 phase 1
du programme de data --> QML."""
import numpy as np
from vecteurs import norme
from matrices import produit_matrice_vecteur, produit_matriciel, rotation


def main():
    """Fonction main() du script, orchestrant les calculs."""
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

    print("Exercice 1.2 - Matrices et multiplication matricielle\n")

    print("A. Matrice fois vecteur")
    Rp2 = produit_matrice_vecteur(R, p2)
    print(f"Rp2 = {formater_vecteur(Rp2)}")
    Ep2 = produit_matrice_vecteur(E, p2)
    print(f"Ep2 = {formater_vecteur(Ep2)}")
    Cp3 = produit_matrice_vecteur(C, p3)
    print(f"Cp3 = {formater_vecteur(Cp3)}")

    print('\nB. Image du carré par E')
    EP = produit_matriciel(E, P)
    afficher_matrice("EP", EP)

    print("\nC. Composition et ordre")
    EC = produit_matriciel(E, C)
    afficher_matrice("EC", EC)

    CE = produit_matriciel(C, E)
    afficher_matrice("CE", CE)

    print(f"EC = CE : {formater_bool(np.allclose(CE, EC))}")

    EC_p2 = produit_matrice_vecteur(EC, p2)
    print(f"(EC)p2 = {formater_vecteur(EC_p2)}")

    E_Cp2 = produit_matrice_vecteur(E, produit_matrice_vecteur(C, p2))
    print(f"E(Cp2) = {formater_vecteur(E_Cp2)}")

    CE_p2 = produit_matrice_vecteur(CE, p2)
    print(f"(CE)p2 = {CE_p2}")

    print("\nD. Transposées et rotations")
    RTR = produit_matriciel(R.T, R)
    afficher_matrice("R^T R", RTR)

    RR = produit_matriciel(R, R)
    afficher_matrice("RR", RR)

    print(f"rotation(90) = R: {formater_bool(np.allclose(R, rotation(90)))}")
    print(f"|p2| = {norme(p2):.2f} ; |Rp2| = {norme(Rp2):.2f}")

    print("\nE. Énergie consommée par jour")
    Xw = produit_matrice_vecteur(X, w)
    print(f"Formes : X {X.shape} ; w {w.shape} ; Xw {Xw.shape}")
    for i, val in enumerate(Xw, start=1):
        print(f"Jour {i} : {val} kWh")
    print(f"Total : {Xw.sum()} kWh")


def formater_nombre(x):
    """Renvoie x à 2 décimales, sans jamais afficher -0.00."""
    return f"{round(x, 2) + 0.0:.2f}"


def formater_vecteur(v):
    """Renvoie un vecteur sous la forme (a, b, ...)."""
    return "(" + ", ".join(formater_nombre(x) for x in v) + ")"


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
