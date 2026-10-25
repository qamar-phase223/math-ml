"""Script de l'exercice 1 de la Phase 1 du programme data QML."""
import numpy as np
from vecteurs import produit_scalaire, norme, angle_degres, travail


def main():
    """Fonction main du script, orchestrant les calculs."""
    d = np.array([5, 0], dtype=np.float64)
    d4 = np.array([5, 1], dtype=np.float64)
    F1 = np.array([6, 8], dtype=np.float64)
    F2 = np.array([0, -50], dtype=np.float64)
    F3 = np.array([-6, 0], dtype=np.float64)
    F4 = np.array([10, 2], dtype=np.float64)
    u = np.array([2, -1, 3], dtype=np.float64)
    v = np.array([4, 2, 1], dtype=np.float64)
    F = [F1, F2, F3]

    print(f"Chariot : d = {tuple(float(x) for x in d)} m")
    for ind, f in enumerate(F, start=1):
        print(f"F{ind} : W = {travail(f, d):.2f} J ; "
              f"|F{ind}| = {norme(f):.2f} N ; "
              f"angle = {angle_degres(f, d):.2f}°")
    print(f"Rampe : d4 = {tuple(float(x) for x in d4)} m")
    print(f"F4 : W = {travail(F4, d4):.2f} J ; "
          f"|F4| = {norme(F4):.2f} N ; "
          f"angle = {angle_degres(F4, d4):.2f}°")
    print(f"u · v = {produit_scalaire(u, v):.2f} ; "
          f"|u| = {norme(u):.2f} ; "
          f"|v| = {norme(v):.2f} ; "
          f"angle = {angle_degres(u, v):.2f}°")
    u_v = abs(produit_scalaire(u, v))
    nu = norme(u)
    nv = norme(v)
    if u_v**2 <= (nu**2)*(nv**2):
        print(f"Cauchy-Schwarz : |u · v| = {u_v:.2f}"
              f" <= |u| |v| = {nu*nv:.2f} : vérifié")
    else:
        print(f"Cauchy-Schwarz : |u · v| = {u_v:.2f}"
              f" <= |u| |v| = {nu*nv:.2f} : non vérifié")


if __name__ == "__main__":
    main()
