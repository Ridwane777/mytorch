# Dossier projet — carte des documents

Point d'entrée pour suivre le travail **sans lire le code**. Règles de rédaction :
[AGENTS.md](../AGENTS.md). Tenir ce tableau à jour à chaque document ajouté.

État au 2026-10-23 :

| Volet | Document | État |
| --- | --- | --- |
| Décisions | [adr/](adr/README.md) : ADR et index, [modèle](adr/TEMPLATE.md) | 8 ADR : 0001 (Python, abandonné), 0002 (C++20, accepté), 0003 (algèbre linéaire) et 0004 (API) acceptés, 0005 (lecture des FEN), 0006 (oracle), 0007 (cross-entropy couplée à Softmax) et 0008 (optimiseurs et planificateurs) proposés |
| Données | [lire-une-fen.md](lire-une-fen.md) : règles d'échecs utiles, lecture d'une ligne du dataset champ par champ, répartition des labels | Écrit |
| Données | [verification-dataset.md](verification-dataset.md) : labels du dataset vérifiés par l'oracle (100 % d'accord) | Écrit |
| Architecture | `architecture.md` : bibliothèque `my_torch`, analyseur, flux d'entraînement et de prédiction | À écrire |
| Performance | [performance/](performance/README.md) : benchmarks, courbes d'apprentissage, scripts ; [optimizers.md](performance/optimizers.md) : optimiseurs et planificateurs, mesuré sur 5 graines | Optimiseurs écrit, reste à écrire pour le vrai jeu de données |
| Légal | [legal/](legal/README.md) : licences, données, risques | À écrire |
| Dépendances | [dependances.md](dependances.md) | Squelette |
| Coûts | `business/couts.md` : mesuré et estimé | À écrire |
| Commercial | [business/](business/README.md) : business plan, SWOT, feuille de route | À écrire |
| Rétrospective | [retrospective/](retrospective/README.md) | À alimenter pendant le projet |
