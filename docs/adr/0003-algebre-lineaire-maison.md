# ADR-0003 : Algèbre linéaire maison, en double, rangée ligne par ligne

- **Statut** : Proposé
- **Date** : 2026-10-21
- **Décideurs** : Ridwane777, Meric-BG (en attente de validation)
- **Issue** : #1

## Contexte

Un réseau de neurones passe presque tout son temps dans des produits de matrices : entrées × poids
pour chaque couche, en avant (forward) comme en arrière (backward). Un batch de 32 positions encodées
sur 768 entrées, multiplié par une couche de 128 neurones, représente déjà environ 3 millions de
multiplications. Le sujet interdit les bibliothèques de réseaux de neurones, mais ne dit rien des
bibliothèques d'algèbre linéaire. Il faut aussi pouvoir vérifier chaque gradient par différences
finies avec une tolérance de 1e-6 (issue #3).

## Options envisagées

| Option | Pour | Contre |
| --- | --- | --- |
| **Classe `Matrix` maison** | Aucune ambiguïté vis-à-vis du sujet ; tout le calcul est écrit et défendable par l'équipe ; aucune dépendance sur l'image de l'autograder | Plus lente qu'une bibliothèque optimisée tant que #33 (blocs, SIMD) n'est pas fait ; risque de bugs |
| Eigen (bibliothèque C++ d'algèbre linéaire, en-têtes seuls) | Très rapide, vectorisée, éprouvée ; API expressive | Peut être jugée contraire à l'esprit du sujet ; dépendance à installer ou à embarquer ; cache une partie du travail évalué |
| BLAS / OpenBLAS (bibliothèque standard de calcul matriciel) | Produit matriciel le plus rapide possible, multicœur | Dépendance système à vérifier sur « the dump » ; API C de bas niveau ; même risque d'interprétation qu'Eigen |

Pour la précision des nombres :

| Option | Pour | Contre |
| --- | --- | --- |
| **`double`** (≈ 16 chiffres) | Vérification des gradients à 1e-6 possible ; moins d'erreurs d'arrondi cumulées | Deux fois plus de mémoire ; calcul vectorisé environ deux fois plus lent |
| `float` (≈ 7 chiffres) | Plus rapide, plus léger | Les arrondis dépassent la tolérance de 1e-6 : le test des gradients échouerait sur un code juste |

## Décision

Une classe **`my_torch::Matrix` maison**, en **`double`** (`using Scalar = double`), stockée dans **un
seul tableau contigu, ligne par ligne** (row-major) : la case (r, c) est à l'indice `r × colonnes + c`.
Un batch est une matrice avec **une ligne par exemple**. Le produit matriciel parcourt les boucles dans
l'ordre i-k-j pour lire la mémoire dans l'ordre.

Pourquoi un seul tableau plutôt qu'un `vector<vector<double>>` : le processeur charge la mémoire par
blocs de 64 octets (lignes de cache). Des cases voisines dans la matrice sont voisines en mémoire, donc
déjà chargées ; un tableau de lignes disperse les lignes en mémoire et coûte une allocation par ligne.

## Conséquences

- ✅ Tout le calcul est à nous et explicable en soutenance ; aucune dépendance d'exécution.
- ✅ La vérification des gradients (#3) est possible avec une tolérance stricte.
- ✅ Passer en `float` plus tard ne demande de changer qu'une ligne (`Scalar`).
- ❌ Produit matriciel naïf lent au départ : l'optimisation (blocs, SIMD, threads) devient un travail à
  part entière (#33, #25), à mesurer contre la version naïve.
- ❌ Deux fois plus de mémoire et de temps qu'en `float`.
- ❌ Bugs numériques possibles dans notre code : couverts par les tests unitaires et la vérification
  des gradients.
- Suivi : demander à l'encadrement si Eigen ou BLAS seraient tolérés. Si oui, et si #33 ne suffit pas,
  un nouvel ADR pourra comparer notre `Matrix` optimisée à Eigen sur un benchmark.
