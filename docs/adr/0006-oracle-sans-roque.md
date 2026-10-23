# ADR-0006 : Oracle par les règles, génération des coups par copie et sans roque

- **Statut** : Proposé
- **Date** : 2026-10-23
- **Décideurs** : Ridwane777 (en attente de Meric-BG)
- **Issue** : #37

## Contexte

L'oracle (`bonus/my_torch_oracle`) classe une position par les règles du jeu : échec, mat, pat ou
rien. Il ne sert jamais à répondre à la place du réseau ; il vérifie les labels du dataset, les
positions produites par l'augmentation de données (#38), et permet la détection du pat (#24).

Distinguer un échec d'un mat, ou un pat d'une position sans rien, demande de savoir si le joueur au
trait a **au moins un coup légal**. Il faut donc générer les coups, et décider comment écarter ceux
qui laissent son propre roi attaqué (pièce clouée, roi qui fuit sur une case attaquée, prise en
passant qui découvre une attaque). Il faut aussi décider si le **roque** est généré.

La représentation du plateau est fixée par l'ADR-0005 : 64 `char` dans l'ordre de la FEN.

## Options envisagées

**Comment écarter les coups qui laissent le roi en échec ?**

| Option | Pour | Contre |
| --- | --- | --- |
| **Jouer chaque coup sur une copie et tester si le roi est attaqué** | Une seule règle couvre les clouages, les fuites du roi et la prise en passant à découverte ; peu de code, facile à vérifier | Une copie de la position et un test d'attaque par coup : plus lent qu'un générateur qui ne produit que des coups légaux |
| Générer directement les seuls coups légaux (calcul des clouages et des pièces qui donnent échec) | Plus rapide | Beaucoup de cas particuliers, chacun source de bugs ; rien de mesurable à gagner sur 723 019 positions |

**Faut-il générer le roque ?**

| Option | Pour | Contre |
| --- | --- | --- |
| **Ne pas générer le roque** | Évite le coup le plus complexe (deux pièces, quatre conditions) ; ne change aucun verdict (voir ci-dessous) | Le comptage de coups (*perft*) est faux sur une position où un roque est possible ; l'oracle ne peut pas servir de générateur de coups général |
| Générer le roque | Générateur complet, *perft* exact partout | Code et tests en plus pour un résultat identique |

Pourquoi le roque ne change aucun verdict :

- **mat** : roquer est interdit quand son roi est en échec, donc le roque ne sauve jamais d'un échec ;
- **pat** : roquer exige que la case voisine du roi sur son trajet (f1, d1, f8 ou d8) soit vide et non
  attaquée ; le roi peut alors y aller d'un pas, ce qui est déjà un coup légal. Le roque n'est donc
  jamais le seul coup légal.

## Décision

Les coups sont générés selon le déplacement des pièces (prise en passant et promotions comprises),
joués sur une copie, puis gardés seulement si le roi du joueur n'est pas attaqué. Le roque n'est pas
généré. La position est classée en croisant « roi attaqué ? » et « au moins un coup légal ? » ; une
position où le roi du camp qui n'a pas le trait est attaqué est signalée comme illégale.

## Conséquences

- ✅ Vérifié par *perft* sur deux positions où aucun roque n'est possible : position de départ jusqu'à
  4 demi-coups (20, 400, 8 902, 197 281) et une finale avec prises en passant et promotions jusqu'à
  5 demi-coups (14, 191, 2 812, 43 238, 674 624), égaux aux valeurs de référence publiées.
- ✅ Sur les 723 019 lignes du dataset : 100 % d'accord avec les labels, aucun pat, aucune position
  illégale, en 5,2 s (mesure du 2026-10-23,
  [`docs/verification-dataset.md`](../verification-dataset.md)).
- ✅ Les droits de roque restent lus et validés par le parseur (ADR-0005), donc disponibles pour
  l'encodage et l'augmentation (#38).
- ❌ `perft` n'est exact que sur des positions sans roque possible ; les tests choisissent leurs
  positions en conséquence.
- ❌ L'oracle n'est pas un générateur de coups général : il ne peut pas servir tel quel à une IA qui
  joue (#27) sans ajouter le roque.
- Suivi : si une IA qui joue (#27) est réalisée, un nouvel ADR ajoutera le roque et le *perft* sur des
  positions qui le contiennent.
