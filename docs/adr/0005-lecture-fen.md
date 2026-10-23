# ADR-0005 : Lecture des FEN, validation du format et représentation du plateau

- **Statut** : Proposé
- **Date** : 2026-10-23
- **Décideurs** : Ridwane777 (en attente de Meric-BG)
- **Issue** : #13 (PR #48)

## Contexte

`my_torch_analyzer` lit des fichiers de positions en notation FEN, suivies d'un label (`--train`) ou non
(`--predict`). Le sujet exige qu'une entrée invalide produise une erreur et le code de sortie 84. Le
dataset compte 723 019 lignes ; il est relu à chaque entraînement. Guide de la notation :
[`docs/lire-une-fen.md`](../lire-une-fen.md).

Deux choix sont à faire : **jusqu'où valider** une ligne, et **comment représenter** le plateau en
mémoire avant l'encodage en entrées du réseau (#14).

Une position est **mal formée** si le texte ne respecte pas la notation (une rangée de 9 cases, une
lettre inconnue). Elle est **illégale** si le texte est correct mais que la situation ne peut pas
arriver dans une vraie partie : un pion sur la 1re ou la 8e rangée, deux rois sur des cases voisines,
neuf pions blancs, ou le camp qui n'a pas le trait déjà en échec.

## Options envisagées

**Jusqu'où valider ?**

| Option | Pour | Contre |
| --- | --- | --- |
| Format seul | Le plus simple ; règle claire | Accepte un plateau sans roi, sur lequel « échec » et « mat » n'ont pas de sens |
| **Format + exactement un roi par couleur** | Garantit une ligne exploitable pour la classification ; contrôle en une ligne | Laisse passer les autres positions illégales |
| Format + légalité complète | Rejette tout ce qui ne peut pas arriver en partie | Demande la génération des coups (le travail de l'oracle, #37) ; plus lent ; une vraie partie n'est jamais nécessaire pour classer une position |

**Comment représenter le plateau ?**

| Option | Pour | Contre |
| --- | --- | --- |
| **64 `char` dans l'ordre de la FEN** (case 0 = a8, 63 = h1 ; `'.'` = vide) | Copie directe du texte, lisible dans un débogueur, testable case par case ; remplissage sans calcul | Retrouver les pièces d'un type demande de parcourir les 64 cases |
| *Bitboards* : 12 entiers de 64 bits, un par type de pièce et couleur, un bit par case | Représentation des moteurs d'échecs ; attaques calculées en quelques opérations binaires ; très rapide pour l'oracle | Plus difficile à lire, à déboguer et à défendre ; inutile pour la seule lecture |
| Grille 8 × 8 (`char[8][8]`) | Accès naturel par (rangée, colonne) | Même contenu que les 64 cases avec deux indices à manipuler ; aucun gain |

## Décision

Le parseur vérifie **le format** des 6 champs et du label, plus **exactement un roi de chaque couleur**.
Il ne vérifie pas la légalité échiquéenne, confiée à l'oracle (#37). Le plateau est un tableau de 64
`char` dans l'ordre de la FEN, contenu dans une `Position` distincte de l'encodage.

## Conséquences

- ✅ Toute ligne acceptée est classable ; toute ligne mal formée lève `my_torch::Error` avec
  `fichier:ligne: raison`, d'où le code 84.
- ✅ Rapide : les 723 019 lignes du dataset sont lues et validées en 0,55 s pour 25 Mo de mémoire
  (mesure du 2026-10-23, build `-O2`, programme de contrôle hors dépôt) ; aucune ligne refusée.
- ✅ Lecture et encodage séparés : on peut changer d'encodage (#14, CNN #29) sans toucher au parseur
  et à ses tests.
- ❌ Des positions illégales mais bien formées sont acceptées ; elles seront signalées par l'oracle
  (#37), pas par le parseur.
- ❌ L'oracle devra parcourir les 64 cases pour trouver les pièces, plus lentement qu'avec des
  *bitboards*.
- Suivi : si l'oracle (#37) ou l'augmentation de données (#38) s'avèrent trop lents avec cette
  représentation, un nouvel ADR introduira des *bitboards* en interne, sans changer l'interface de
  `Position`.
