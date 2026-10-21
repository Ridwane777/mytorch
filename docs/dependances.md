# Dépendances

Pour chaque dépendance : rôle, version testée, licence, et ce qui se passe si elle disparaît.

| Dépendance | Rôle | Version | Licence | Si elle disparaît |
| --- | --- | --- | --- | --- |
| Python | Langage (ADR-0001, proposé) | 3.12.3 en local, 3.10 en CI | PSF | — |
| pytest | Tests (dev) | ≥ 8.0 | MIT | Remplaçable par unittest |
| ruff | Lint (dev) | ≥ 0.6 | MIT | Remplaçable par flake8 |

À vérifier : les versions disponibles sur l'image de l'autograder (« the dump »).
