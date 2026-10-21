# Dépendances

Pour chaque dépendance : rôle, version testée, licence, et ce qui se passe si elle disparaît.
Les exécutables n'ont aucune dépendance d'exécution autre que la bibliothèque standard C++.

| Dépendance | Rôle | Version testée | Licence | Si elle disparaît |
| --- | --- | --- | --- | --- |
| g++ (C++20) | Compilation (ADR-0002) | 13.3.0 | GPL v3 avec exception d'exécution | clang++ 18 (testé en local) |
| GNU make | Compilation (exigé par le sujet) | — | GPL v3 | — |
| Criterion | Tests unitaires (dev) | 2.4.1 | MIT | GoogleTest, Catch2 |
| clang-format | Format du code (dev) | — | Apache 2.0 avec exception LLVM | — |

À vérifier : les versions disponibles sur l'image de l'autograder (« the dump »).
