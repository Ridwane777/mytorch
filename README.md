# my_torch — réseau de neurones fait main et analyseur d'échiquier

Projet the school , en C++20 ([ADR-0002](docs/adr/0002-langage-cpp.md)). Deux livrables :

- **`my_torch`** (`include/my_torch/`, `src/my_torch/`) : une bibliothèque de réseau de neurones
  générique, écrite sans bibliothèque de deep learning (PyTorch, TensorFlow… sont interdits). Elle
  permet de créer, entraîner, sauvegarder et recharger un réseau.
- **`my_torch_analyzer`** : un exécutable qui lit des positions d'échecs en notation FEN et prédit
  l'état de la partie (`Checkmate`, `Check`, `Nothing`, avec la couleur), par apprentissage supervisé.

> État : squelette du dépôt. Les choix techniques sont dans [docs/adr/](docs/adr/README.md).

## Compilation et utilisation

Prérequis : g++ (C++20) et make ; Criterion pour les tests.

```bash
make                    # my_torch_analyzer et my_torch_generator
./my_torch_analyzer --predict my_torch_network_<nom>.nn chessboards.txt
./my_torch_analyzer --train [--save NOUVEAU.nn] RESEAU.nn chessboards.txt
./my_torch_generator config.conf 3
```

En cas d'erreur, le message part sur la sortie d'erreur et le programme quitte avec le code 84.

| Règle | Effet |
| --- | --- |
| `make` | Compile en `-O2` |
| `make debug` | Recompile avec AddressSanitizer et UBSan |
| `make tests_run` | Compile et lance les tests Criterion (avec couverture) |
| `make clean` / `fclean` / `re` | Nettoyage et recompilation |

## Organisation du dépôt

| Chemin | Contenu |
| --- | --- |
| `include/my_torch/` | En-têtes publics de la bibliothèque : seul contrat entre le module et l'analyseur |
| `src/my_torch/` | Implémentation de la bibliothèque, compilée en `build/libmy_torch.a` |
| `src/analyzer/` | Partie échecs : lecture FEN, encodage, interface en ligne de commande |
| `src/generator/` | `my_torch_generator` : crée des réseaux vierges à partir d'une configuration |
| `tests/` | Tests Criterion, mêmes sous-dossiers que `src/` |
| `my_torch_network_*.nn` | Réseau pré-entraîné évalué par l'autograder |
| `configs/` | Configurations de réseaux (hyperparamètres) |
| `scripts/` | Entraînement, benchmarks et graphiques, pour reproduire nos résultats |
| `bonus/` | Bonus (exigé par le sujet) |
| `docs/` | Dossier projet : décisions, performance, légal, business, rétrospective. Carte : [docs/README.md](docs/README.md) |

Les jeux de données (`dataset/`) ne sont pas poussés : ils sont trop lourds. Ils sont montrés en
soutenance et décrits dans `docs/performance/`.

Avant le premier commit : [CONTRIBUTING.md](CONTRIBUTING.md).
