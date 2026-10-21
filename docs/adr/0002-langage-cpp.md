# ADR-0002 : C++20 comme langage du projet

- **Statut** : Accepté
- **Date** : 2026-10-21
- **Décideurs** : Ridwane777, Meric-BG
- **Remplace** : [ADR-0001](0001-langage-python-numpy.md) (Python + NumPy, resté au stade de proposition)

## Contexte

Le sujet accepte tout langage qui fonctionne sur « the dump » (l'image de l'autograder) et interdit les
bibliothèques de réseaux de neurones. Tout le calcul (produits matriciels, rétropropagation,
optimiseurs) doit donc être écrit par l'équipe. L'entraînement porte sur 723 019 positions : la vitesse
de calcul pèse directement sur le nombre d'expériences possibles (recherche d'hyperparamètres,
comparaisons d'encodages). Le sujet valorise aussi le parallélisme en bonus.

## Options envisagées

| Option | Pour | Contre |
| --- | --- | --- |
| Python + NumPy (ADR-0001) | Développement rapide, graphiques dans le même langage | Dépendance NumPy à faire valider ; NumPy fait une partie du calcul à notre place ; boucles non vectorisées lentes ; parallélisme limité par le GIL |
| Python pur | Aucune dépendance | Produits matriciels en boucles Python : entraînement sur 723 000 positions trop lent |
| **C++20** | Performance native, contrôle de la mémoire et du calcul, threads standard (`std::thread`), compilateur présent sur l'image the school, chaque opération est écrite et défendable par l'équipe | Algèbre linéaire, sérialisation et parsing à écrire ; erreurs mémoire possibles ; cycle de développement plus long |
| Rust | Performance et sûreté mémoire | Outillage et langage moins maîtrisés par le binôme |

## Décision

**C++20**, compilé par un Makefile (exigé par le sujet), sans dépendance d'exécution autre que la
bibliothèque standard. Le module `my_torch` est une bibliothèque statique (`libmy_torch.a`) liée à
`my_torch_analyzer` et `my_torch_generator`. Tests unitaires avec Criterion (outil standard à the school).

Les scripts hors exécutables (graphiques de benchmark, analyse du dataset) peuvent rester en Python :
ils ne sont pas évalués par l'autograder et ne font pas partie du réseau.

## Conséquences

- ✅ Entraînement rapide, donc plus d'expériences et de benchmarks dans le même temps.
- ✅ Aucune ambiguïté sur l'interdiction des bibliothèques : tout le calcul est à nous.
- ✅ Parallélisme natif possible pour le bonus (`std::thread`).
- ❌ Classe de matrice et opérations à écrire et optimiser nous-mêmes (cache, vectorisation), avec le
  risque de bugs numériques : d'où les tests de gradient par différences finies (issue #3).
- ❌ Erreurs mémoire possibles : `make debug` compile avec AddressSanitizer et UBSan.
- ❌ Plus de code à écrire qu'en Python pour le même résultat.
- Suivi : vérifier la version de g++ et de Criterion sur l'image de l'autograder (en local : g++ 13.3,
  Criterion 2.4.1). Le choix d'une bibliothèque d'algèbre linéaire (Eigen…) ou d'un code maison fera
  l'objet d'un ADR séparé (issue #1).
