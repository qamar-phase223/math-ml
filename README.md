# math-ml

Ce dépôt contient les exercices de la phase 1 du parcours Data → QML : les fondements mathématiques du machine learning, implémentés en Python avec NumPy et vérifiés par des tests automatisés.

## Structure du dépôt

```
math-ml/
├── exercices-math/
│   ├── 01_produit_scalaire.py
│   └── vecteurs.py
├── tests/
│   └── test_vecteurs.py
├── pytest.ini
├── requirements.txt
└── README.md
```

## Installation

```
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
```

## Tests et vérification du style

À exécuter depuis la racine du dépôt :

```
python -m pytest -v
python -m pycodestyle exercices-math tests
```

Le fichier `pytest.ini` déclare le dossier `exercices-math` dans le `pythonpath` et le dossier `tests` comme emplacement des tests. Les fichiers de test importent ainsi directement les modules (`from vecteurs import ...`) sans manipulation du chemin.

## Exercices

- `exercices-math/01_produit_scalaire.py` et `exercices-math/vecteurs.py` : produit scalaire, norme, angle entre deux vecteurs et application au travail d'une force.

## Exercice 1.1 — Produit scalaire, norme et angle (`01_produit_scalaire.py` et `vecteurs.py`)

**Objectif** : calculer le produit scalaire, la norme et l'angle entre deux vecteurs, à la main puis avec NumPy, les appliquer au travail d'une force sur un déplacement, et transformer les résultats du calcul à la main en tests pytest.

### Contenu

| Fonction | Fichier | Rôle |
|---|---|---|
| `produit_scalaire` | `vecteurs.py` | Calcule le produit scalaire de deux vecteurs ; lève `ValueError` si leurs formes diffèrent. |
| `norme` | `vecteurs.py` | Calcule la norme euclidienne d'un vecteur à partir du produit scalaire. |
| `angle_degres` | `vecteurs.py` | Calcule l'angle entre deux vecteurs, en degrés ; lève `ValueError` si l'un des vecteurs est nul. Le cosinus est borné à [-1, 1] par `np.clip` avant l'appel à `arccos`. |
| `travail` | `vecteurs.py` | Calcule le travail W = F · d d'une force F sur un déplacement d. |
| `main` | `01_produit_scalaire.py` | Applique les fonctions à quatre forces sur deux déplacements (chariot sur sol horizontal et chariot sur une rampe), puis à deux vecteurs abstraits à 3 composantes, et vérifie l'inégalité de Cauchy-Schwarz. |

### Données

Repère, pour tous les vecteurs du plan : axe x horizontal, orienté vers la droite ; axe y vertical, orienté vers le haut.

- **A. Chariot sur sol horizontal** : déplacement d = (5.0, 0.0) m ; traction par une corde F1 = (6.0, 8.0) N ; poids F2 = (0.0, -50.0) N ; frottement F3 = (-6.0, 0.0) N.
- **B. Chariot tiré sur une rampe** : déplacement d4 = (5.0, 1.0) m, soit une montée de 1 m pour 5 m à l'horizontale ; traction parallèle à la pente F4 = (10.0, 2.0) N.
- **C. Vecteurs abstraits à 3 composantes, sans unité** : u = (2.0, -1.0, 3.0) et v = (4.0, 2.0, 1.0).

### Tests (`tests/test_vecteurs.py`)

Dix cas, écrits à partir du calcul à la main.

| Fonction testée | Cas couverts |
|---|---|
| `produit_scalaire` | Cas général, vecteurs perpendiculaires, vecteurs à 3 composantes, formes incompatibles (`ValueError` attendue). |
| `norme` | Cas général. |
| `angle_degres` | Angle aigu, vecteurs de sens opposés (180°), vecteurs de même direction et de même sens (0°), vecteur nul (`ValueError` attendue). |
| `travail` | Traction parallèle à la pente sur la rampe. |

Les comparaisons utilisent `pytest.approx`, avec une tolérance absolue (`abs=`) explicite lorsque la valeur attendue est nulle.

### Lancement

À exécuter depuis la racine du dépôt :

```
source .venv/Scripts/activate
python exercices-math/01_produit_scalaire.py
python -m pytest -v
```

### Résultats clés

| Vecteurs | Travail ou produit scalaire | Norme | Angle |
|---|---|---|---|
| F1 = (6.0, 8.0) N, d = (5.0, 0.0) m | 30.00 J | 10.00 N | 53.13° |
| F2 = (0.0, -50.0) N, d = (5.0, 0.0) m | 0.00 J | 50.00 N | 90.00° |
| F3 = (-6.0, 0.0) N, d = (5.0, 0.0) m | -30.00 J | 6.00 N | 180.00° |
| F4 = (10.0, 2.0) N, d4 = (5.0, 1.0) m | 52.00 J | 10.20 N | 0.00° |
| u = (2.0, -1.0, 3.0), v = (4.0, 2.0, 1.0) | 9.00 | 3.74 et 4.58 | 58.34° |

- **Tests** : `pytest -v` affiche 10 passed.
- **Cauchy-Schwarz** : \|u · v\| = 9.00 ≤ \|u\| \|v\| = 17.15, inégalité vérifiée.
- **Vecteurs de même direction et de même sens** : pour F4 et d4, le cosinus calculé vaut 1.0000000000000002 au lieu de 1 en raison des arrondis de la virgule flottante, ce qui fait renvoyer `nan` à `arccos`. Le bornage du cosinus par `np.clip` corrige ce cas limite.
- **Tolérances des tests** : autour de 0, la tolérance relative par défaut de `pytest.approx` devient nulle ; une tolérance absolue est donc fixée explicitement (10⁻⁹ pour un produit scalaire, 10⁻⁵ degré pour un angle, l'arc cosinus amplifiant les erreurs d'arrondi au voisinage de 0°).
