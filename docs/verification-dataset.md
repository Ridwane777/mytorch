# Vérification des labels du dataset par l'oracle

## Objectif

Vérifier que les labels du dataset sont exacts avant d'entraîner un réseau dessus. Un label faux
apprend une erreur au réseau. Outil : l'oracle par les règles (#37, ADR-0006), indépendant du
réseau.

## Méthode

1. Chaque ligne est lue par le parseur de l'analyseur (#13, ADR-0005).
2. L'oracle classe la position : il regarde si le roi au trait est attaqué et s'il reste un coup
   légal.
3. Le verdict est comparé au label de la ligne. Les pats et les positions illégales sont comptés à
   part, car ils n'ont pas de label dans le sujet.

Reproduire :

```
make -C bonus
./bonus/my_torch_oracle dataset/*/*.txt
```

## Machine

Intel Core i7-1355U (13e génération, 12 threads, un seul utilisé), Linux, g++ 13.3, `-O2`.

## Résultats (mesurés le 2026-10-23)

| Mesure | Valeur |
| --- | --- |
| Positions lues | 723 019 |
| Accord avec le label | 723 019 (100,00 %) |
| Désaccords | 0 |
| Pats | 0 |
| Positions illégales | 0 |
| Temps total | 5,2 s |
| Mémoire maximale | 25 Mo |

## Conclusion

Les 723 019 labels sont exacts selon les règles du jeu. Le dataset ne contient aucun pat, ce qui
confirme qu'une classe « pat » (#24) ne pourra pas être apprise sur ces données seules. Il faudra
générer des positions et les étiqueter avec l'oracle.
