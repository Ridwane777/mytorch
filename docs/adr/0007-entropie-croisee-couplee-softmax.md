# ADR-0007 : Entropie croisée couplée à Softmax, gradient sauté dans la rétropropagation

- **Statut** : Proposé
- **Date** : 2026-10-24
- **Décideurs** : Meric-BG (en attente de Ridwane777, qui écrit la rétropropagation dans #4)
- **Issue** : #3

## Contexte

Le réseau de l'analyseur finit par une couche `Softmax`, qui transforme les scores `z` en
probabilités `p` (positives, de somme 1), puis le coût compare `p` à la cible `t` (un vecteur
one-hot : 1 pour la bonne classe, 0 ailleurs). Pour l'entropie croisée, `L = -Σ t_i·log p_i`.

La rétropropagation applique la règle de dérivation en chaîne : le gradient du coût par rapport à `p`
est d'abord calculé, puis chaque couche le transforme en remontant. Pour l'entropie croisée,
`dL/dp_i = -t_i / p_i`. Le backward de `Softmax` le multiplie ensuite par `p_i` environ. Quand le
réseau se trompe gravement, la bonne classe a `p_i ≈ 0` : la division par `p_i` donne un nombre
énorme, que le Softmax multiplie par un nombre minuscule, et **l'information du gradient est perdue
par arrondi**. C'est précisément le cas où l'on a le plus besoin d'un gradient fort.

Mesure (programme jetable appelant `Softmax::backward` sur deux classes, bonne classe écrasée de
`écart` points de score, `-t/p` borné à ε = 1e-15) : à écart 10 et 30 le chemin naïf est encore
exact (−0,99996 et −1,0), mais à écart 40 (`p = 4e-18 < ε`) il donne **−0,004 au lieu de −1** :
le gradient disparaît d'un coup, sans erreur ni NaN. Le chemin couplé donne −1 dans les trois cas.

Mathématiquement, les deux étapes se simplifient : `dL/dz_i = p_i - t_i`. Cette forme ne divise
jamais par `p_i` et se calcule en une soustraction. L'ADR-0004 annonçait déjà ce couplage sans en
fixer le mécanisme.

## Options envisagées

| Option | Pour | Contre |
| --- | --- | --- |
| **`CrossEntropy` prend la sortie du Softmax, renvoie `dL/dz` et se déclare « fusionnée » (`fusesSoftmax()`) ; la rétropropagation saute alors le backward du Softmax final** | Gradient stable ; le réseau garde sa couche `Softmax` visible, donc `predict` renvoie des probabilités, comme dans l'exemple de l'ADR-0004 ; aucun changement de la liste de couches | Un cas particulier dans la rétropropagation ; `gradient()` ne renvoie plus `dL/dprédiction` pour ce coût, ce qu'il faut documenter |
| `CrossEntropy` prend les scores `z` (logits) et applique le softmax en interne ; le réseau n'a pas de couche Softmax | Aucun cas particulier : c'est le modèle de `nn.CrossEntropyLoss` de PyTorch ; calcul stable par soustraction du maximum | `predict` ne renvoie plus de probabilités : il faut un softmax à part à la prédiction ; l'exemple de l'ADR-0004 et les fichiers de réseau changent |
| Garder `dL/dp = -t/p` et borner `p` à ε (1e-15) | Aucune modification de l'API | Le gradient reste faux quand `p < ε` (la bonne classe est justement celle qui est écrasée) ; seulement un pansement |

## Décision

On retient la première option. `Loss` gagne une méthode virtuelle `fusesSoftmax()` (faux par défaut).
`CrossEntropy` la renvoie à vrai : sa `value()` lit les probabilités (bornées à 1e-15 avant le
logarithme), et son `gradient()` renvoie directement le gradient par rapport à l'**entrée** du
Softmax final, `(p·W - w⊙q) / taille_du_batch` (`W = Σ w_c·q_c`), soit `(p - t) / batch` sans
pondération ni lissage. `Network::fit` (#4) doit, quand `loss.fusesSoftmax()` est vrai, sauter le
`backward` de la dernière couche si c'est un `Softmax`. Le test de gradient du réseau complet
(`tests/my_torch/gradient_check.hpp`) fait déjà ce saut et prouve par différences finies que le
résultat est juste ; un test vérifie aussi qu'il détecte l'oubli du saut.

Deux options de la même classe, sans effet sur l'API : le **lissage des étiquettes** `s` remplace la
cible par `q = (1 - s)·t + s/classes` (le réseau n'est plus poussé vers des probabilités de 0 ou 1,
donc des scores infinis), et des **poids par classe** `w_c` multiplient le coût des exemples de la
classe `c`, pour le déséquilibre des classes (#16). La moyenne reste divisée par la taille du batch,
pas par la somme des poids.

## Conséquences

- ✅ Gradient exact et stable même quand la bonne classe a une probabilité presque nulle.
- ✅ `predict` renvoie toujours des probabilités ; l'API de l'ADR-0004 ne change pas.
- ✅ Lissage et pondération des classes sont couverts par le même test de gradient.
- ❌ Cas particulier à connaître dans la rétropropagation : si #4 l'oublie, le gradient est faux sans
  erreur visible (seul le test de gradient du réseau complet le détecte).
- ❌ `gradient()` n'a pas le même sens pour tous les coûts (`dL/dp` pour MSE, `dL/dz` pour
  `CrossEntropy`) ; documenté dans `Loss.hpp`.
- ❌ Un réseau dont la dernière couche n'est pas un `Softmax` ne peut pas utiliser `CrossEntropy` :
  le gradient serait celui d'un autre calcul. Un coût pour les sorties Sigmoid existe
  (`BinaryCrossEntropy`).
- ❌ La *focal loss* n'est pas faite : son gradient par rapport aux scores n'a pas cette forme
  simple et demanderait son propre ADR.
- Suivi : #4 appelle `fusesSoftmax()` dans `Network::fit` et ajoute un test sur un réseau qui apprend
  XOR avec Softmax + `CrossEntropy`. À revoir si une couche suit le Softmax (non prévu).
