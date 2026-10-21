Closes #

## Ce que ça change

<!-- Une ou deux phrases : ce qu'un relecteur doit comprendre avant de lire le code. -->

## Décision concernée

<!-- L'ADR (docs/adr/) que cette PR applique ou remet en cause, ou « aucune ». -->

## Comment le vérifier

<!-- La commande à lancer, ou ce qu'il faut regarder. -->

## Avant de demander une relecture

- [ ] `make re` compile sans avertissement
- [ ] `make tests_run` passe
- [ ] `make debug` puis le cas testé : aucune erreur des sanitizers
- [ ] `clang-format --dry-run --Werror` ne signale rien
- [ ] la branche est à jour avec `main` (`git pull --rebase origin main`)
- [ ] aucune bibliothèque de réseaux de neurones (PyTorch, TensorFlow…) n'est importée
- [ ] si la PR fixe un choix structurant, l'ADR est écrit et ajouté à l'index
- [ ] si la PR change un résultat mesuré, `docs/performance/` est à jour avec son script
