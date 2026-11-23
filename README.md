# math-ml

Ce dépôt contient les exercices de la phase 1 du parcours Data → QML : les fondements mathématiques du machine learning, implémentés en Python avec NumPy et vérifiés par des tests automatisés.

## Structure du dépôt

```
math-ml/
├── exercices-math/
│   ├── 01_produit_scalaire.py
│   ├── 02_matrices.py
│   ├── 03_modes_propres.py
│   ├── matrices.py
│   ├── propres.py
│   └── vecteurs.py
├── tests/
│   ├── test_matrices.py
│   ├── test_propres.py
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

Résultat attendu : `48 passed` (10 cas pour l'exercice 1.1, 16 cas pour l'exercice 1.2, 22 cas pour l'exercice 1.3).

## Exercices

- `exercices-math/01_produit_scalaire.py` et `exercices-math/vecteurs.py` : produit scalaire, norme, angle entre deux vecteurs et application au travail d'une force.
- `exercices-math/02_matrices.py` et `exercices-math/matrices.py` : produit matrice-vecteur, produit matriciel et matrice de rotation, appliqués aux transformations du plan et au calcul d'une énergie consommée.
- `exercices-math/03_modes_propres.py` et `exercices-math/propres.py` : valeurs et vecteurs propres, modes propres d'un système masses-ressorts et comparaison avec les valeurs singulières.

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
python -m pytest -v tests/test_vecteurs.py
```

### Résultats clés

| Vecteurs | Travail ou produit scalaire | Norme | Angle |
|---|---|---|---|
| F1 = (6.0, 8.0) N, d = (5.0, 0.0) m | 30.00 J | 10.00 N | 53.13° |
| F2 = (0.0, -50.0) N, d = (5.0, 0.0) m | 0.00 J | 50.00 N | 90.00° |
| F3 = (-6.0, 0.0) N, d = (5.0, 0.0) m | -30.00 J | 6.00 N | 180.00° |
| F4 = (10.0, 2.0) N, d4 = (5.0, 1.0) m | 52.00 J | 10.20 N | 0.00° |
| u = (2.0, -1.0, 3.0), v = (4.0, 2.0, 1.0) | 9.00 | 3.74 et 4.58 | 58.34° |

- **Tests** : `python -m pytest -v tests/test_vecteurs.py` affiche 10 passed.
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

- **Tests** : `python -m pytest -v tests/test_matrices.py` affiche 16 passed.
- **Assemblage des colonnes** : `np.array` empile une liste de vecteurs en lignes, ce qui donnait la transposée de A B ; les colonnes sont assemblées par `np.column_stack`.
- **Formes `(n,)` et `(n, 1)`** : un vecteur rangé en colonne, de forme `(3, 1)`, est refusé par `produit_matrice_vecteur`, même s'il a le bon nombre d'éléments.
- **Affichage des zéros** : un résidu d'arrondi négatif, comme -1.2e-16, s'affiche `-0.00` avec le format `.2f`. L'expression `round(x, 2) + 0.0` le ramène à `0.00`.
- **Tolérance des rotations** : `rotation(180.0)` contient des résidus d'environ 1.2e-16 à la place des zéros, car `np.radians(180.0)` n'est pas exactement égal à π ; les tests de rotation utilisent donc une tolérance absolue de 10⁻¹².

## Exercice 1.3 — Valeurs propres, modes propres et valeurs singulières (`03_modes_propres.py` et `propres.py`)

**Objectif** : calculer les valeurs et vecteurs propres de matrices 2 × 2, à la main puis avec NumPy, en tirer les pulsations propres d'un système masses-ressorts, et comparer valeurs propres et valeurs singulières sur des matrices symétriques et non symétriques.

### Contenu

| Fonction | Fichier | Rôle |
|---|---|---|
| `matrice_raideur` | `propres.py` | Renvoie la matrice de raideur K, de forme `(2, 2)`, en N/m, de deux masses reliées par trois ressorts k1, k2 et k3 entre deux murs ; lève `ValueError` si une raideur n'est pas strictement positive. |
| `est_symetrique` | `propres.py` | Indique si une matrice de forme `(2, 2)` est égale à sa transposée, avec une tolérance absolue de 10⁻¹². |
| `valeurs_propres` | `propres.py` | Renvoie les deux valeurs propres, par ordre croissant, en résolvant le polynôme caractéristique ; lève `ValueError` si elles ne sont pas réelles (discriminant négatif) ou si la matrice n'est pas de forme `(2, 2)`. |
| `vecteur_propre` | `propres.py` | Renvoie le vecteur propre unitaire associé à une valeur propre, dont la première composante non nulle est positive ; la norme est calculée avec `produit_scalaire`. |
| `valeurs_singulieres` | `propres.py` | Renvoie les deux valeurs singulières, par ordre décroissant : racines carrées des valeurs propres de AᵀA, matrice calculée avec `produit_matriciel`. |
| `pulsations` | `propres.py` | Renvoie les pulsations propres ω = √(λ / m), en rad/s, par ordre croissant ; lève `ValueError` si la masse n'est pas strictement positive. |
| `main` | `03_modes_propres.py` | Calcule les modes propres de deux systèmes masses-ressorts et d'un système alourdi, compare valeurs propres et valeurs singulières sur quatre matrices, puis vérifie la trace, le déterminant et le produit des valeurs singulières de A. |
| `formater_nombre`, `formater_vecteur`, `formater_couple`, `afficher_matrice`, `formater_bool` | `03_modes_propres.py` | Mettent en forme les résultats : 4 décimales sans `-0.0000`, vecteurs entre parenthèses, couples séparés par ` ; `, matrices ligne par ligne, booléens en `vrai` ou `faux`. |

Le module réutilise `produit_scalaire` (exercice 1.1) et `produit_matriciel` (exercice 1.2). Ni l'opérateur `@` ni `np.linalg` ne sont utilisés dans le module et le script : `np.linalg` sert uniquement de référence dans deux tests de concordance.

### Données

**Système masses-ressorts.** Deux masses égales m glissent sans frottement sur un support horizontal ; le ressort k1 relie le mur de gauche à la première masse, k2 relie les deux masses, k3 relie la seconde masse au mur de droite. x1 et x2 sont les déplacements des masses à partir de l'équilibre, en m, positifs vers la droite. Le bilan des forces donne m x'' = −K x, avec K = [[k1 + k2, -k2], [-k2, k2 + k3]].

| Système | k1 (N/m) | k2 (N/m) | k3 (N/m) | m (kg) |
|---|---|---|---|---|
| Système 1 | 3.0 | 1.0 | 3.0 | 1.0 |
| Système 2 | 1.0 | 2.0 | 4.0 | 1.0 |
| Système 1 lourd | 3.0 | 1.0 | 3.0 | 4.0 |

**Matrices de comparaison**, sans unité, de forme `(2, 2)` :

- A = [[4.0, 0.0], [3.0, -5.0]], non symétrique ;
- A_s = [[9.0, -3.0], [-3.0, 1.0]], symétrique ;
- R = [[0.0, -1.0], [1.0, 0.0]], rotation d'un quart de tour ;
- D = [[2.0, 0.0], [0.0, 5.0]], diagonale.

### Tests (`tests/test_propres.py`)

Vingt-deux cas, écrits à partir du calcul à la main.

| Groupe | Cas couverts | Tolérance |
|---|---|---|
| Matrice de raideur | Systèmes 1 et 2 | `abs=1e-12` |
| Symétrie | K du système 1 (symétrique), A (non symétrique) | — |
| Valeurs propres | K des systèmes 1 et 2, A, A_s | `abs=1e-12` |
| Vecteurs propres | K du système 2 (λ = 2.0 et 7.0), A (λ = 4.0 et -5.0) : norme égale à 1 et A v = λ v | `abs=1e-12` |
| Orthogonalité | Produit scalaire des deux modes propres du système 2 | `abs=1e-12` |
| Valeurs singulières | D, R, A_s | `abs=1e-12` |
| Pulsations | Système 1, système 1 lourd | `abs=1e-12` |
| Concordance avec NumPy | `valeurs_propres` comparée à `np.linalg.eigvalsh` ; `valeurs_singulieres` comparée à `np.linalg.svd` | `abs=1e-12` |
| Exceptions | R, sans valeur propre réelle ; matrice de forme `(3, 3)` | `ValueError` attendue |

Les matrices d'entrée des tests sont écrites en dur, et non construites avec `matrice_raideur` : une erreur dans cette fonction ne fait échouer que son propre test.

### Lancement

À exécuter depuis la racine du dépôt :

```
source .venv/Scripts/activate
python exercices-math/03_modes_propres.py
python -m pytest -v tests/test_propres.py
```

### Résultats clés

| Système | Valeurs propres (N/m) | Pulsations (rad/s) | Modes propres |
|---|---|---|---|
| Système 1 | 3.0000 ; 5.0000 | 1.7321 ; 2.2361 | (0.7071, 0.7071) : masses en phase ; (0.7071, -0.7071) : masses en opposition |
| Système 2 | 2.0000 ; 7.0000 | 1.4142 ; 2.6458 | (0.8944, 0.4472) ; (0.4472, -0.8944) |
| Système 1 lourd | 3.0000 ; 5.0000 | 0.8660 ; 1.1180 | Mêmes modes que le système 1 |

| Matrice | Valeurs propres | Valeurs singulières | Interprétation |
|---|---|---|---|
| D | 2.0000 ; 5.0000 | 5.0000 ; 2.0000 | Matrice diagonale : les valeurs propres se lisent sur la diagonale. |
| R | Aucune valeur réelle | 1.0000 ; 1.0000 | La rotation ne laisse aucune direction sur sa droite et conserve les longueurs. |
| A_s | 0.0000 ; 10.0000 | 10.0000 ; 0.0000 | Matrice symétrique de rang 1 : tout le plan est aplati sur une droite. |
| A | -5.0000 ; 4.0000 | 6.3246 ; 3.1623 | Matrice non symétrique : vecteurs propres non orthogonaux (produit scalaire 0.3162) et valeurs singulières différentes des \|λ\|. |

- **Tests** : `python -m pytest -v tests/test_propres.py` affiche 22 passed ; `python -m pytest -v` affiche 48 passed pour l'ensemble du dépôt.
- **Symétrie de K** : le ressort k2 exerce sur les deux masses des forces égales et opposées (troisième loi de Newton), d'où K12 = K21 = -k2. Les valeurs propres sont donc réelles et les deux modes propres orthogonaux (produit scalaire 0.0000 pour les systèmes 1 et 2).
- **Effet de la masse** : quadrupler la masse laisse K et ses modes propres inchangés et divise les pulsations par √4 = 2.
- **Contrôles sur A** : trace(A) = -1.0 = somme des valeurs propres ; det(A) = -20.0 = produit des valeurs propres ; produit des valeurs singulières = 20.0 = \|det(A)\|.
- **Discriminant** : il est calculé sous la forme (a - d)² + 4bc, égale à trace² - 4 det.
- **Vecteurs propres** : ils sont testés par leurs propriétés (norme 1 et A v = λ v), et non par leurs composantes, car un vecteur propre n'est défini qu'à un facteur près.
- **Arrondis** : une valeur propre de AᵀA légèrement négative, due aux arrondis, est ramenée à 0 avant la racine carrée.
