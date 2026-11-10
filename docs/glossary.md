# Glossary

| Term | Meaning in this project |
| --- | --- |
| FEN | Forsyth–Edwards Notation: one line describing a chess position (board, side to move, castling, en passant, move counters) ([reading-a-fen.md](reading-a-fen.md)) |
| Encoding | How a FEN becomes the numbers fed to the network, e.g. `fen-attacks-901` (ADR-0016) |
| Dense layer | Every input connected to every output: `output = input · W + b` |
| Activation | Non-linear function after a layer (ReLU, sigmoid, tanh), without which layers would collapse into one linear map |
| Softmax | Turns the five outputs into probabilities that sum to 1 |
| Loss | Number measuring how wrong a prediction is; here cross-entropy |
| Gradient | Derivative of the loss with respect to each weight: the direction that increases the loss |
| Backpropagation | Computing every gradient from the last layer to the first with the chain rule |
| Finite differences | Checking a gradient by moving a weight by a tiny ε and measuring the change of the loss |
| Optimizer | Rule that updates the weights from their gradients: SGD, Adam… (ADR-0008) |
| Learning rate | Size of each update step |
| Batch | Group of examples (64 here) processed before one update |
| Epoch | One pass over the whole training set |
| Train / validation / test split | Data to learn from / to choose and stop with / to measure once at the end (ADR-0017) |
| Overfitting | Learning the training examples by heart instead of the general rule: training loss falls, validation loss rises |
| Early stopping | Stop when the validation loss has not improved for `patience` epochs and keep the best weights (ADR-0010) |
| Augmentation | New training examples made from existing ones by a transformation that keeps the label (ADR-0019) |
| Accuracy | Share of right answers |
| Precision / recall / F1 | Per class: share of the predictions of that class that are right / share of that class that is found / their harmonic mean |
| Macro F1 | Mean of the F1 of the five classes, each class counting the same |
| Confusion matrix | Table of expected class against predicted class |
| Seed | Number that fixes a random sequence, so that a run can be repeated exactly |
| Cache blocking | Computing a matrix product by tiles that stay in the processor cache (ADR-0024) |
| Thread | One flow of execution; several threads run on several cores at the same time |
| Oracle | Our rule-based chess checker (bonus), used to verify labels, never to answer for the network |
| ADR | Architecture Decision Record: one decision, its options, its trade-offs (`docs/adr/`) |
