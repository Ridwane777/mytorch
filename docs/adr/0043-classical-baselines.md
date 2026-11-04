# ADR-0043: Classical baselines, which models and how they are chosen

- **Status**: Accepted (2026-11-04)
- **Date**: 2026-11-04
- **Deciders**: Ridwane777 (issue #43); accepted by the pair
- **Issue**: #43

## Context
The comparison framework (ADR-0036) allows scikit-learn and gradient boosting in
`bonus/comparaison/`. Issue #43 asks to place our network against simpler methods, with a table of
accuracy, time and model size. The choices to make: which families, which settings, how to pick
one without favouring anyone, and what data they see.

## Options considered
| Option | Pros | Cons |
| --- | --- | --- |
| **Four families, a small grid each, chosen on validation, test read once (chosen)** | Same protocol as our network (ADR-0017): the test never chooses; a few hours in all on a laptop | A small grid can under-sell a family |
| A large search per family (random search, cross-validation) | Each family at its best | Days of computation; the forest alone takes 7 GB per run |
| The baselines on the augmented training data | Same data as the pushed network | 2.3 million rows: four times the memory and time, out of reach for the forest and the MLP |
| XGBoost instead of LightGBM | Better known | Slower on 901 columns; the issue asks for one of the two |

## Decision
Logistic regression, random forest, scikit-learn `MLPClassifier` with our architecture, and
LightGBM, each with 2 or 3 settings (`bonus/comparaison/baselines.py`). Every setting is trained on
the training part without augmentation and scored on the validation part; the best macro F1 is
scored once on the test part. Our networks are re-scored on the same test part and machine, and our
network without augmentation is shown next to them. Results in
[docs/performance/23-baselines.md](../performance/23-baselines.md).

## Consequences
- ✅ The answer is measured: a linear model (65 %) and a forest (77 %) are far behind; the same
  architecture in scikit-learn matches ours on the same data; LightGBM (96.75 %) beats our pushed
  dense network (96.15 %); the CNN (98.07 %) beats every baseline with the smallest file.
- ✅ Same split, same encoding, same choice rule: a difference comes from the model.
- ❌ Unequal budgets: no augmentation and a small grid for the baselines.
- ❌ The forest is bounded by the memory of the machine (100 trees, 7.2 GB at peak).
- ❌ Two more Python packages in `bonus/comparaison/.venv` (scikit-learn, LightGBM).
- Follow-up: LightGBM on the augmented data, or with more leaves, would tell whether boosting can
  reach the CNN; the final comparison report (#44) uses these numbers.
