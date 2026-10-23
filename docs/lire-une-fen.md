# Lire une ligne du dataset (notation FEN)

Guide pour comprendre les données sans connaître les échecs. Il sert de référence au parseur FEN (#13)
et à l'encodage des positions.

## 1. Le dataset

Le dataset est fourni par the school et n'est pas poussé (`dataset/` est dans `.gitignore`). Il contient trois dossiers
(`check/`, `checkmate/`, `nothing/`), chacun avec `10_pieces.txt`, `20_pieces.txt` et `many_pieces.txt`.
Une ligne = une position + la réponse attendue. Répartition mesurée sur les 723 019 lignes :

| Label | Lignes |
| --- | --- |
| `Nothing` | 300 000 |
| `Check White` | 135 587 |
| `Check Black` | 113 713 |
| `Checkmate White` | 94 295 |
| `Checkmate Black` | 79 424 |

## 2. Les règles d'échecs utiles

Deux camps, **Blancs** et **Noirs**, jouent **chacun leur tour**, Blancs d'abord. Chaque camp a 16 pièces.

| Pièce | Lettre | Déplacement |
| --- | --- | --- |
| Roi (*King*) | K | 1 case dans toutes les directions |
| Dame (*Queen*) | Q | En ligne droite ou en diagonale, autant de cases que voulu |
| Tour (*Rook*) | R | En ligne droite (horizontale ou verticale), autant de cases que voulu |
| Fou (*Bishop*) | B | En diagonale, autant de cases que voulu |
| Cavalier (*kNight*) | N | En « L » : 2 cases puis 1 sur le côté ; seule pièce qui saute par-dessus les autres |
| Pion (*Pawn*) | P | Avance d'1 case (2 au premier mouvement), **prend en diagonale** d'1 case vers l'avant |

Les pièces, sauf le cavalier, ne traversent pas les autres : une pièce sur le chemin bloque. Prendre une pièce, c'est
aller sur sa case.

Ce que le réseau doit reconnaître :

- **Échec** : un roi est attaqué par une pièce adverse. Le joueur doit en sortir au coup suivant.
- **Échec et mat** : le roi est attaqué et **aucun coup** ne permet d'en sortir. La partie est finie.
- **Nothing** : aucun roi n'est attaqué.

« Check White » se lit **« le Blanc fait échec »** (c'est le roi noir qui est attaqué) ; « Checkmate Black » : le Noir
fait mat.

## 3. Les 6 champs d'une ligne

Exemple réel (`dataset/check/10_pieces.txt`) :

```
8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59 Check White
└──────────┬─────────────┘ │ │ │ │ └┬┘ └────┬────┘
        champ 1            2 3 4 5  6    label (pas FEN)
```

| # | Champ | Exemple | Signification |
| --- | --- | --- | --- |
| 1 | Placement des pièces | `8/8/R2k4/…` | Les 8 rangées, de la 8e à la 1re, séparées par `/` |
| 2 | Trait | `b` | Qui doit jouer : `w` Blancs, `b` Noirs |
| 3 | Roques | `-` | Roques encore possibles (`KQkq`), `-` si aucun |
| 4 | En passant | `-` | Case où une prise en passant est possible, sinon `-` |
| 5 | Demi-coups | `7` | Demi-coups depuis la dernière prise ou le dernier coup de pion |
| 6 | Numéro du coup | `59` | Numéro du coup complet en cours, commence à 1 |
| + | Label | `Check White` | Réponse attendue, ajoutée par le dataset ; peut contenir un espace |

## 4. Traduire le champ 1, pas à pas

### Étape 1 : couper aux `/`

On obtient 8 morceaux = 8 rangées, **de haut en bas** (rangée 8, côté noir, jusqu'à la rangée 1, côté blanc).

| Morceau | `8` | `8` | `R2k4` | `4r1p1` | `8` | `5K2` | `5P2` | `8` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Rangée | 8 | 7 | 6 | 5 | 4 | 3 | 2 | 1 |

### Étape 2 : lire chaque morceau de gauche à droite (colonnes a → h)

- un **chiffre** = autant de cases vides ;
- une **lettre** = une pièce sur la case suivante. **Majuscule = blanc, minuscule = noir.**

```
R2k4  : R → tour blanche en a6 ; 2 → b6, c6 vides ; k → roi noir en d6 ; 4 → e6 à h6 vides
4r1p1 : 4 → a5 à d5 vides ; r → tour noire en e5 ; 1 → f5 vide ; p → pion noir en g5 ; 1 → h5 vide
5K2   : 5 → a3 à e3 vides ; K → roi blanc en f3 ; 2 → g3, h3 vides
5P2   : 5 → vides ; P → pion blanc en f2 ; 2 → vides
```

Contrôle : dans chaque morceau, chiffres + lettres = 8 (`4r1p1` → 4 + 1 + 1 + 1 + 1 = 8).

### Étape 3 : l'échiquier

```
8 | .  .  .  .  .  .  .  . |
7 | .  .  .  .  .  .  .  . |
6 | R  .  .  k  .  .  .  . |
5 | .  .  .  .  r  .  p  . |
4 | .  .  .  .  .  .  .  . |
3 | .  .  .  .  .  K  .  . |
2 | .  .  .  .  .  P  .  . |
1 | .  .  .  .  .  .  .  . |
    a  b  c  d  e  f  g  h
```

Blancs : roi f3, tour a6, pion f2. Noirs : roi d6, tour e5, pion g5.

### Étape 4 : vérifier le label

La tour blanche en a6 regarde vers la droite : b6 et c6 sont vides, puis d6 = roi noir. Le roi noir est attaqué, le
Blanc fait échec : **Check White**. Cohérent avec le trait `b` : c'est aux Noirs de jouer pour sortir de l'échec.

## 5. Les champs 2 à 6

### Trait (champ 2)

`w` (*white*) ou `b` (*black*) : le camp qui doit jouer. C'est en général le camp qui subit l'échec.

### Roque (champ 3)

Coup spécial qui déplace **le roi et une tour en même temps** pour abriter le roi dans un coin : le roi avance de 2
cases vers la tour, la tour passe de l'autre côté du roi. Petit roque côté roi (colonne h), grand roque côté dame
(colonne a). Autorisé seulement si ni le roi ni cette tour n'ont bougé, si rien ne les sépare et **si le roi n'est pas en
échec**.

| Lettre | Roque encore possible |
| --- | --- |
| `K` | Blancs, petit roque |
| `Q` | Blancs, grand roque |
| `k` | Noirs, petit roque |
| `q` | Noirs, grand roque |
| `-` | Aucun |

### Prise en passant (champ 4)

Règle spéciale des pions : un pion qui avance de **2 cases** d'un coup peut être pris par un pion adverse placé à côté,
**comme s'il n'avait avancé que d'1 case**. Ce droit ne dure qu'un coup. Le champ donne la case sautée (`e3`, `d6`…),
toujours sur la 3e ou la 6e rangée, ou `-`.

### Demi-coups (champ 5) et numéro du coup (champ 6)

- Un **demi-coup** est le coup d'**un seul** joueur, quel qu'il soit. Un **coup complet** est un coup blanc suivi de la
  réponse noire.
- Le **champ 5** compte les demi-coups joués **depuis la dernière prise ou le dernier coup de pion**. Il repart à 0 à
  chaque prise ou coup de pion. À 100 (50 coups de chaque côté), un joueur peut réclamer la nulle (règle des 50 coups).
- Le **champ 6** est le numéro du coup complet **en cours**, prises comprises. Il commence à 1 et augmente après
  chaque coup des Noirs.

| Demi-coup | Joueur | Action | Champ 5 |
| --- | --- | --- | --- |
| 1 | Blanc | Bouge un pion | 0 |
| 2 | Noir | Bouge un cavalier | 1 |
| 3 | Blanc | Bouge un fou | 2 |
| 4 | Noir | Bouge une tour | 3 |
| 5 | Blanc | Sa dame prend une pièce | 0 |
| 6 | Noir | Bouge son roi | 1 |

Dans `… b - - 7 59` : 7 demi-coups sans prise ni coup de pion ; 58 coups complets sont terminés, les Blancs ont joué
leur 59e coup et c'est aux Noirs de jouer le leur.

## 6. Ce qui sert au réseau

| Champ | Utilité pour détecter échec et mat |
| --- | --- |
| 1. Pièces | Essentiel |
| 2. Trait | Utile : le camp au trait est celui qui subit l'échec |
| 3. Roques | Faible : un roi en échec ne peut pas roquer ; utile pour l'augmentation de données (#38) |
| 4. En passant | Très rare : peut être le seul coup qui évite un mat |
| 5–6. Compteurs | Aucune |

Le parseur lit et **valide** les 6 champs, car une ligne mal formée doit donner une erreur (code 84) ; l'encodage choisit
ensuite ce qu'il garde.

## 7. Exercice

Dessiner `4k3/8/8/8/8/8/8/4K2R` (3 pièces).

<details>
<summary>Solution</summary>

```
8 | .  .  .  .  k  .  .  . |
7 | .  .  .  .  .  .  .  . |
  …        (rangées 6 à 2 vides)
1 | .  .  .  .  K  .  .  R |
    a  b  c  d  e  f  g  h
```

Roi noir en e8, roi blanc en e1, tour blanche en h1.

</details>
