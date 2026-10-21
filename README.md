# my_torch — réseau de neurones fait main et analyseur d'échiquier

Projet the school . Deux livrables :

- **`my_torch`** (`src/my_torch/`) : un module de réseau de neurones générique, écrit sans
  bibliothèque de deep learning (PyTorch, TensorFlow… sont interdits). Il permet de créer,
  entraîner, sauvegarder et recharger un réseau.
- **`my_torch_analyzer`** : un exécutable qui lit des positions d'échecs en notation FEN et
  prédit l'état de la partie (`Checkmate`, `Check`, `Nothing`), par apprentissage supervisé.

> État : squelette du dépôt. Les choix techniques sont en cours dans [docs/adr/](docs/adr/README.md).

## Utilisation

```bash
make
./my_torch_analyzer --predict my_torch_network_<nom>.nn chessboards.txt
./my_torch_analyzer --train [--save NOUVEAU.nn] RESEAU.nn chessboards.txt
```

En cas d'erreur, le message part sur la sortie d'erreur et le programme quitte avec le code 84.

## Organisation du dépôt

| Chemin | Contenu |
| --- | --- |
| `my_torch_analyzer` | Point d'entrée exigé par le sujet |
| `my_torch_network_*.nn` | Réseau pré-entraîné évalué par l'autograder |
| `src/my_torch/` | Module générique de réseau de neurones |
| `src/analyzer/` | Partie échecs : lecture FEN, encodage, interface en ligne de commande |
| `configs/` | Configurations de réseaux (hyperparamètres) |
| `scripts/` | Entraînement, recherche d'hyperparamètres, benchmarks et graphiques, pour reproduire nos résultats |
| `tests/` | Tests, mêmes sous-dossiers que `src/` |
| `bonus/` | Bonus (exigé par le sujet) |
| `docs/` | Dossier projet : décisions, performance, légal, business, rétrospective. Carte : [docs/README.md](docs/README.md) |

Les jeux de données (`dataset/`) ne sont pas poussés : ils sont trop lourds. Ils sont montrés en
soutenance et décrits dans `docs/performance/`.

## Développement

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
make tests_run
ruff check src tests scripts my_torch_analyzer
```

Avant le premier commit : [CONTRIBUTING.md](CONTRIBUTING.md).
