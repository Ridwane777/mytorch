# Comment on travaille sur ce dépôt

## La règle principale

**Personne ne pousse directement sur `main`.** On y entre par une Pull Request relue par l'autre
membre du binôme. C'est depuis `main` que l'autograder évalue et qu'on fait la soutenance.

## Le cycle

```bash
git checkout main && git pull
git checkout -b <type>/<sujet-court>      # ex. feat/fen-parser, docs/adr-encodage
# ... commits ...
git pull --rebase origin main
git push -u origin <type>/<sujet-court>
```

Types de branche et de commit : `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `perf`.
Messages : `type(portée): description` (ex. `feat(my_torch): couche dense et rétropropagation`).

## Relecture

Une approbation suffit, puis l'auteur fusionne. Branches courtes : une PR = une idée.

## Avant de demander une relecture

```bash
make re
make tests_run
ruff check src tests scripts my_torch_analyzer
```

La CI (`.github/workflows/ci.yml`) lance les mêmes vérifications. Une PR rouge ne se fusionne pas.

## Décisions et documentation

Un choix structurant (encodage du plateau, architecture du réseau, fonction de coût, optimiseur,
format de sauvegarde…) s'écrit dans `docs/adr/` dans la même PR. Un résultat d'entraînement cité
dans la doc doit pouvoir être reproduit avec un script de `scripts/` et une config de `configs/`.
Voir `AGENTS.md`.
