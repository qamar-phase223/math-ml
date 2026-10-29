# math-ml

Ce dépôt contient les exercices de la phase 1 du parcours Data → QML : les fondements mathématiques du machine learning, implémentés en Python avec NumPy et vérifiés par des tests automatisés.

## Structure du dépôt

```
math-ml/
├── exercices-math/
│   ├── 01_produit_scalaire.py
│   ├── 02_matrices.py
│   ├── matrices.py
│   └── vecteurs.py
├── tests/
│   ├── test_matrices.py
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

Résultat attendu : `26 passed` (10 cas pour l'exercice 1.1, 16 cas pour l'exercice 1.2).

## Exercices

- `exercices-math/01_produit_scalaire.py` et `exercices-math/vecteurs.py` : produit scalaire, norme, angle entre deux vecteurs et application au travail d'une force.
- `exercices-math/02_matrices.py` et `exercices-math/matrices.py` : produit matrice-vecteur, produit matriciel et matrice de rotation, appliqués aux transformations du plan et au calcul d'une énergie consommée.

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

## Exercice 1.2 — Matrices et multiplication matricielle (`02_matrices.py` et `matrices.py`)

**Objectif** : voir une matrice comme une transformation du plan, calculer le produit matrice-vecteur et le produit matriciel comme des séries de produits scalaires, à la main puis avec NumPy, constater que l'ordre des transformations compte et maîtriser les formes (*shapes*) des tableaux.

### Contenu

| Fonction | Fichier | Rôle |
|---|---|---|
| `produit_matrice_vecteur` | `matrices.py` | Calcule A x, de forme `(m,)`, pour A de forme `(m, n)` et x de forme `(n,)` : chaque composante est le produit scalaire d'une ligne de A avec x. Lève `ValueError` si x n'est pas de forme `(n,)`, y compris s'il est de forme `(n, 1)`. |
| `produit_matriciel` | `matrices.py` | Calcule A B, de forme `(m, p)`, pour A de forme `(m, n)` et B de forme `(n, p)` : la colonne j du résultat est A fois la colonne j de B. Lève `ValueError` si A n'a pas autant de colonnes que B a de lignes. |
| `rotation` | `matrices.py` | Renvoie la matrice de rotation de forme `(2, 2)` d'un angle donné en degrés, dans le sens antihoraire. |
| `main` | `02_matrices.py` | Applique les fonctions aux sommets du carré unité et aux trois transformations, compare les compositions, vérifie les propriétés de la rotation et calcule l'énergie consommée par jour. |
| `formater_nombre`, `formater_vecteur`, `afficher_matrice`, `formater_bool` | `02_matrices.py` | Mettent en forme les résultats : 2 décimales sans `-0.00`, vecteurs entre parenthèses, matrices ligne par ligne, booléens en `vrai` ou `faux`. |

Le produit matriciel est construit uniquement à partir de `produit_scalaire` (exercice 1.1) : `produit_scalaire` est appelé par `produit_matrice_vecteur`, lui-même appelé par `produit_matriciel`. L'opérateur `@` n'est utilisé que dans les tests, comme référence.

### Données

Repère : origine O ; axe x horizontal, orienté vers la droite ; axe y vertical, orienté vers le haut ; grandeurs sans unité.

- **Carré unité** : sommets p0 = (0.0, 0.0), p1 = (1.0, 0.0), p2 = (1.0, 1.0) et p3 = (0.0, 1.0), rangés en colonnes dans la matrice P de forme `(2, 4)`.
- **Transformations** :
  - rotation d'un quart de tour, sens antihoraire : R = [[0.0, -1.0], [1.0, 0.0]] ;
  - étirement de facteur 2 selon x : E = [[2.0, 0.0], [0.0, 1.0]] ;
  - cisaillement horizontal : C = [[1.0, 1.0], [0.0, 1.0]].
- **Énergie consommée** : X, de forme `(4, 3)`, donne en heures la durée d'utilisation de trois appareils (colonnes) pendant quatre jours (lignes) ; w = (2.0, 3.0, 5.0) donne leurs puissances en kW.

### Tests (`tests/test_matrices.py`)

Seize cas, écrits à partir du calcul à la main.

| Groupe | Cas couverts | Tolérance |
|---|---|---|
| Produit matrice-vecteur | R p2, E p2, C p3, X w | `abs=1e-9` |
| Produit matriciel | E P, E C, C E, R^T R comparé à l'identité | `abs=1e-9` |
| Concordance avec NumPy | A de forme `(3, 2)` fois B de forme `(2, 4)`, comparé à `A @ B` | `abs=1e-12` |
| Rotation | `rotation(90.0)` égale R ; `rotation(0.0)` égale l'identité | `abs=1e-12` |
| Propriétés | E C différent de C E ; `rotation(30.0)` composée avec `rotation(60.0)` égale `rotation(90.0)` | `abs=1e-12` |
| Exceptions | vecteur de forme `(2,)` au lieu de `(3,)` ; vecteur rangé en colonne, de forme `(3, 1)` ; produit E X non défini | `ValueError` attendue |

Les valeurs attendues des matrices sont données sous forme de tableaux NumPy : `pytest.approx` refuse les listes imbriquées.

### Lancement

À exécuter depuis la racine du dépôt :

```
source .venv/Scripts/activate
python exercices-math/02_matrices.py
python -m pytest -v tests/test_matrices.py
```

### Résultats clés

| Calcul | Résultat | Interprétation |
|---|---|---|
| R p2 | (-1.00, 1.00) | Le point (1.0, 1.0) tourne d'un quart de tour. |
| E p2 | (2.00, 1.00) | Seule la composante x est doublée. |
| C p3 | (1.00, 1.00) | Le point (0.0, 1.0) glisse vers la droite. |
| E P | [[0.00, 2.00, 2.00, 0.00], [0.00, 0.00, 1.00, 1.00]] | Le carré devient un rectangle de 2 × 1. |
| E C et C E | [[2.00, 2.00], [0.00, 1.00]] et [[2.00, 1.00], [0.00, 1.00]] | L'ordre des transformations compte : E C ≠ C E. |
| (E C) p2, E (C p2), (C E) p2 | (4.00, 1.00), (4.00, 1.00), (3.00, 1.00) | Associativité vérifiée ; la transformation de droite agit en premier. |
| R^T R | [[1.00, 0.00], [0.00, 1.00]] | La transposée de R annule la rotation. |
| R R | [[-1.00, 0.00], [0.00, -1.00]] | Deux quarts de tour forment un demi-tour. |
| \|p2\| et \|R p2\| | 1.41 et 1.41 | La rotation conserve les longueurs. |
| X w | 12.00, 9.00, 11.00 et 20.00 kWh | Énergie consommée par jour ; total : 52.00 kWh. |

- **Tests** : `pytest -v` affiche 26 passed (10 cas de l'exercice 1.1 et 16 cas de l'exercice 1.2).
- **Assemblage des colonnes** : `np.array` empile une liste de vecteurs en lignes, ce qui donnait la transposée de A B ; les colonnes sont assemblées par `np.column_stack`.
- **Formes `(n,)` et `(n, 1)`** : un vecteur rangé en colonne, de forme `(3, 1)`, est refusé par `produit_matrice_vecteur`, même s'il a le bon nombre d'éléments.
- **Affichage des zéros** : un résidu d'arrondi négatif, comme -1.2e-16, s'affiche `-0.00` avec le format `.2f`. L'expression `round(x, 2) + 0.0` le ramène à `0.00`.
- **Tolérance des rotations** : `rotation(180.0)` contient des résidus d'environ 1.2e-16 à la place des zéros, car `np.radians(180.0)` n'est pas exactement égal à π ; les tests de rotation utilisent donc une tolérance absolue de 10⁻¹².
