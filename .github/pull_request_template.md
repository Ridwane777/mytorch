## Ce que ça change

<!-- Une ou deux phrases : ce qu'un relecteur doit comprendre avant de lire le code. -->

## Décision concernée

<!-- L'ADR (docs/adr/) que cette PR applique ou remet en cause, ou « aucune ». -->

## Comment le vérifier

<!-- La commande à lancer, ou ce qu'il faut regarder. -->

## Avant de demander une relecture

- [ ] `python -m pytest -q` passe
- [ ] `ruff check src tests scripts my_torch_analyzer` ne signale rien
- [ ] `make re` fonctionne
- [ ] la branche est à jour avec `main` (`git pull --rebase origin main`)
- [ ] aucune bibliothèque de réseaux de neurones (PyTorch, TensorFlow…) n'est importée
- [ ] si la PR fixe un choix structurant, l'ADR est écrit et ajouté à l'index
- [ ] si la PR change un résultat mesuré, `docs/performance/` est à jour avec son script
