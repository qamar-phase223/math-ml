"""Script de l'exercice 2 phase 1
du programme de data --> QML."""
import numpy as np
from vecteurs import norme
from matrices import produit_matrice_vecteur, produit_matriciel, rotation


def main():
    """Fonction main() du script, orchestrant les calculs."""
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

    print("Exercice 1.2 - Matrices et multiplication matricielle\n")

    print("A. Matrice fois vecteur")
    Rp2 = produit_matrice_vecteur(R, p2)
    print(f"Rp2 = (" +
          ", ".join(f"{round(x, 2) + 0.0:.2f}" for x in Rp2) + ")")
    Ep2 = produit_matrice_vecteur(E, p2)
    print(f"Ep2 = (" +
          ", ".join(f"{round(x, 2) + 0.0:.2f}" for x in Ep2) + ")")
    Cp3 = produit_matrice_vecteur(C, p3)
    print(f"Cp3 = (" +
          ", ".join(f"{round(x, 2) + 0.0:.2f}" for x in Cp3) + ")")

    print('\nB. Image du carré par E')
    EP = produit_matriciel(E, P)
    print("EP =")
    for ligne in EP:
        print(f"  [" +
              ", ".join(f"{round(x, 2) + 0.0:.2f}" for x in ligne) + "]")

    print("\nC. Composition et ordre")
    EC = produit_matriciel(E, C)
    print("EC =")
    for ligne in EC:
        print("  [" +
              ", ".join(f"{round(x, 2) + 0.0:.2f}" for x in ligne) + "]")
    CE = produit_matriciel(C, E)
    print("CE =")
    for ligne in CE:
        print("  [" +
              ", ".join(f"{round(x, 2) + 0.0:.2f}" for x in ligne) + "]")
    print(f"EC = CE : {str(np.allclose(EC, CE)).lower()}")

    print("\nD. Transposées et rotations")
    print("R^T R =")
    RTR = produit_matriciel(R.T, R)
    for ligne in RTR:
        print("  [" +
              ", ".join(f"{round(x, 2) + 0.0:.2f}" for x in ligne) + "]")
    print("RR = ")
    RR = produit_matriciel(R, R)
    for ligne in RR:
        print("  [" +
              ", ".join(f"{round(x, 2) + 0.0:.2f}" for x in ligne) + "]")
    print(f"rotation(90) = R: {str(np.allclose(R, rotation(90))).lower()}")
    print(f"|p2| = {norme(p2):.2f} ; |Rp2| = {norme(Rp2):.2f}")

    print("\nE. Énergie consomée par jour")
    Xw = produit_matrice_vecteur(X, w)
    lXw = len(Xw)
    print(f"Formes : X {X.shape} ; w {w.shape} ; Xw {Xw.shape}")
    for i, val in enumerate(Xw, start=1):
        print(f"Jour {i} : {val} kWh")
    print(f"Total = {Xw.sum()} kWh")


if __name__ == "__main__":
    main()
