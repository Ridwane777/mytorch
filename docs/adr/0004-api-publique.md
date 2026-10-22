# ADR-0004 : API publique de la bibliothèque my_torch

- **Statut** : Accepté
- **Date** : 2026-10-21
- **Décideurs** : Ridwane777, Meric-BG
- **Issue** : #1

## Contexte

Le binôme code en parallèle : couches et rétropropagation (#2, #4) d'un côté, coûts, optimiseurs et
sauvegarde (#3, #5, #8) de l'autre. Il faut fixer tôt les signatures communes, écrites dans
`include/my_torch/`, pour que les morceaux s'emboîtent. L'API doit aussi accueillir sans être réécrite
les extensions prévues : convolution (#29), normalisation (#30), dropout (#7), nouveaux optimiseurs.

## Options envisagées

**Où placer les fonctions d'activation (ReLU, Softmax…) ?**

| Option | Pour | Contre |
| --- | --- | --- |
| **Une couche à part entière, sans poids** | Un réseau est une simple liste de couches ; chaque activation a son propre forward/backward testable ; même modèle que `nn.ReLU` de PyTorch | Plus de couches dans la liste ; un résultat intermédiaire de plus en mémoire |
| Un paramètre de la couche dense | Moins de couches ; un peu moins de mémoire | Couche dense plus complexe ; chaque nouvelle couche (convolution…) doit réimplémenter les activations |

**Forme du réseau**

| Option | Pour | Contre |
| --- | --- | --- |
| **Séquentiel** (une file de couches) | Simple à écrire, à entraîner, à sauvegarder ; suffit pour toute la partie obligatoire | Ne permet pas les blocs résiduels |
| Graphe de couches | Permet les architectures modernes (résiduels, attention) | Beaucoup plus complexe (ordre de calcul, sauvegarde) pour un besoin optionnel |

## Décision

- **`Layer`** (classe abstraite) : `forward(input, training)` calcule la sortie ; `backward(gradOutput)`
  reçoit dL/dsortie, accumule les gradients de ses poids et renvoie dL/dentrée (règle de dérivation en
  chaîne) ; `parameters()` expose ses paires (poids, gradient) ; `type()` donne un nom stable pour la
  sauvegarde. `training` distingue entraînement et prédiction (dropout, normalisation).
- **Les activations sont des couches** sans paramètre.
- **`Loss`** : `value` (coût moyen du batch) et `gradient` (premier gradient de la rétropropagation).
  Softmax et entropie croisée sont combinées pour obtenir le gradient simplifié `prédiction − cible`,
  plus simple et plus stable numériquement.
- **`Optimizer`** : `step(parameters)` met les poids à jour sans connaître les couches ;
  `setLearningRate` permet aux planificateurs de faire varier le taux.
- **`Network`** séquentiel : possède ses couches (`std::unique_ptr`), `predict`, `fit` (époques,
  mini-batchs, mélange, validation, remise à zéro des gradients après chaque pas), `save` / `load`.
  `fit` renvoie un **`History`** (une `EpochStats` par époque) pour tracer les courbes d'apprentissage.
- **Graine fixe** (`TrainConfig::seed`) pour tout le hasard : un même entraînement redonne le même réseau.
- **`my_torch::Error`** pour toute erreur ; les exécutables l'attrapent et quittent avec 84.

Exemple d'utilisation visé :

```cpp
Network net;
net.add(std::make_unique<Dense>(768, 128));
net.add(std::make_unique<ReLU>());
net.add(std::make_unique<Dense>(128, 5));
net.add(std::make_unique<Softmax>());

CrossEntropy loss;
Adam optimizer(0.001);
History history = net.fit(train, &validation, loss, optimizer, {.epochs = 20, .batchSize = 32});
net.save("my_torch_network_full.nn");
```

## Conséquences

- ✅ Les deux lots codent en parallèle contre les mêmes en-têtes.
- ✅ Une nouvelle couche ou un nouvel optimiseur s'ajoute sans modifier le reste (polymorphisme).
- ✅ Entraînements reproductibles, exigés par le sujet et nécessaires aux comparaisons.
- ❌ Pas de blocs résiduels ni d'attention avec un réseau séquentiel : #31 et #32 demanderont un nouvel
  ADR pour passer à un graphe.
- ❌ Le couplage Softmax + entropie croisée impose de traiter ce cas à part dans la rétropropagation.
- ❌ Une activation par couche coûte un résultat intermédiaire de plus en mémoire.
- Suivi : toute modification des en-têtes de `include/my_torch/` passe par une PR relue par les deux.
