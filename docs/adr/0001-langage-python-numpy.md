# ADR-0001 : Python 3 avec NumPy comme langage du projet

- **Statut** : Abandonné, remplacé par [ADR-0002](0002-langage-cpp.md)
- **Date** : 2026-10-21
- **Décideurs** : Ridwane777

## Contexte

Le sujet impose un langage qui fonctionne sur « the dump » (l'image de l'autograder). Les bibliothèques de réseaux de neurones
(PyTorch, TensorFlow…) sont interdites : la propagation avant, la rétropropagation, les optimiseurs et la sauvegarde doivent être
écrits par l'équipe. Le livrable comprend aussi des scripts d'entraînement reproductibles et des benchmarks avec graphiques.

## Options envisagées

1. **Python 3 + NumPy** : calcul matriciel vectorisé, écosystème pour les graphiques (matplotlib), itérations rapides.
2. **Python 3 pur** (sans NumPy) : aucune dépendance, mais les produits matriciels en boucles Python sont très lents.
3. **C / C++** : performances maximales et contrôle de la mémoire, mais algèbre linéaire, sérialisation et graphiques à écrire
   ou à intégrer, cycle de développement plus long.
4. **Rust** : performances et sûreté mémoire, mais courbe d'apprentissage et outillage moins familier pour l'équipe.

## Décision

**Python 3 avec NumPy.** NumPy sert uniquement d'outil d'algèbre linéaire (tableaux, produits matriciels) ; toute la logique
de réseau de neurones (couches, activations, coût, rétropropagation, optimiseurs, sauvegarde) est écrite à la main.
matplotlib est utilisé seulement pour les graphiques de benchmark, hors de l'exécutable évalué.

## Conséquences

✅ **Gagné**
- Vitesse de développement : plus de temps pour l'expérimentation (hyperparamètres, régularisation, benchmarks).
- Calcul vectorisé en C via NumPy : entraînement en mini-batch réaliste sans écrire de BLAS.
- Graphiques et scripts d'analyse dans le même langage que le module.
- Pas de compilation : le Makefile se limite à rendre les exécutables lançables.

❌ **Perdu**
- Performances inférieures à du C/C++ optimisé sur les parties non vectorisées (parsing FEN, boucles d'entraînement).
- Dépendance externe (NumPy) à vérifier sur l'image de l'autograder (versions locales : Python 3.12.3, NumPy 1.26.4).
- Risque d'interprétation : il faudra pouvoir montrer en soutenance que NumPy n'est pas une bibliothèque de réseaux de neurones
  et qu'aucune différentiation automatique n'est utilisée.
- Parallélisme (bonus) limité par le GIL ; il passera par la vectorisation ou le multiprocessing.
