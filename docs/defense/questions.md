# Likely questions

Short answers; the linked document gives the details and the measurements.

## Choices
| Question | Answer | Details |
| --- | --- | --- |
| Why C++ and not Python? | Speed of training decides how many experiments we can run, NumPy would do part of the computation for us, and the GIL limits threads | [ADR-0002](../adr/0002-cpp-language.md), [ADR-0001](../adr/0001-python-numpy-language.md) (abandoned) |
| Why your own matrix class and not Eigen? | The subject forbids neural-network libraries and says nothing about linear algebra. The staff allowed Eigen orally, but not in the PDF: we kept our own code so as to take no risk at grading and to be able to explain every computation. In hindsight, Eigen behind our `Matrix` would have given fast products from day 1 (not measured) | [ADR-0003](../adr/0003-in-house-linear-algebra.md), [retrospective](../retrospective/retrospective.md) §3 |
| Why `double`? | Finite-difference checks need the precision; it costs twice the memory and time of `float` | ADR-0003; [retrospective](../retrospective/retrospective.md) §3 |
| How is a FEN turned into numbers? | `fen-attacks-901`: 768 piece planes, side to move, castling, and the squares each side attacks. Best of four encodings measured | [ADR-0016](../adr/0016-input-encoding.md), [03-encodings.md](../performance/03-encodings.md) |
| Why 901 → 256 → 128 → 5? | Best of 8 trials on the validation split; 512 → 256 is not significantly better over five seeds and is twice as large | [ADR-0023](../adr/0023-final-architecture.md), [09-trials.md](../performance/09-trials.md) |
| Why softmax with cross-entropy? | Five exclusive classes; together their gradient is simply prediction − target, stable and cheap | [ADR-0007](../adr/0007-softmax-cross-entropy-coupling.md) |
| Why Adam? | Fastest and most consistent in our optimizer comparison on test problems | [ADR-0008](../adr/0008-optimizers-and-schedulers.md), [07-optimizers.md](../performance/07-optimizers.md) |
| How do you avoid overfitting? | Early stopping on a validation split, best weights restored; L2 and dropout available | [ADR-0010](../adr/0010-overfitting-remedies.md), [08-overfitting.md](../performance/08-overfitting.md) |
| Why no class weighting? | The classes are not that unbalanced; weighting and resampling did not help on the measurement | [ADR-0018](../adr/0018-class-imbalance.md), [04-imbalance.md](../performance/04-imbalance.md) |
| What is the augmentation? | Two board symmetries that keep the label, applied to the training split only: +4.5 points of accuracy | [ADR-0019](../adr/0019-symmetry-augmentation.md), [10-final-network.md](../performance/10-final-network.md) |

## Evaluation
| Question | Answer | Details |
| --- | --- | --- |
| How good is the network? | 96.2 % accuracy, macro F1 0.948, on 71,555 positions never used for training or for any choice | [10-final-network.md](../performance/10-final-network.md) |
| How do you know the test positions were never seen? | The split deduplicates positions before cutting, with a fixed seed; validation and test are never augmented | [ADR-0017](../adr/0017-dataset-split.md), [01-dataset.md](../performance/01-dataset.md) |
| Where does it fail? | Check against checkmate of the same side: 91 % of the errors. The colour is never wrong | 10-final-network.md |
| How would you fix it? | An input saying whether the king has a legal move; not measured yet | [retrospective](../retrospective/retrospective.md) §5 |
| Is the result reproducible? | Same command, same files: seeded split, shuffling and initialisation; bit-identical on two machines and for any thread count | [ADR-0022](../adr/0022-validation-tail-and-training-pipeline.md), [ADR-0024](../adr/0024-parallel-blocked-matmul.md) |

## Implementation
| Question | Answer | Details |
| --- | --- | --- |
| How do you know your gradients are right? | Every gradient is compared with a finite-difference estimate in the tests | `tests/my_torch/`, AGENTS.md |
| Explain backpropagation in your code | `Network::fit`: forward, loss, gradient of the loss, `backward` through the layers in reverse, optimizer step | `src/my_torch/Network.cpp`, [ADR-0009](../adr/0009-training-loop.md) |
| How does it fit 2.3 million training lines in memory? | It doesn't need to: batches are encoded on demand (430 MB peak) instead of the 10 GB of encoding the whole file | [ADR-0021](../adr/0021-training-data-source.md), [11-training-memory.md](../performance/11-training-memory.md) |
| Is it parallel? | The matrix products are blocked and split over all threads, with results identical for any thread count: training 2.4 times faster on 12 threads | [ADR-0024](../adr/0024-parallel-blocked-matmul.md), [12-matmul.md](../performance/12-matmul.md) |
| Why not 12 times faster? | Uneven cores (2 fast, 8 slow) and only 64 rows per batch to share; profiling shows the products are 87 % of the time, so encoding in parallel would not help | 12-matmul.md |
| What is in the network file? | Versioned text with the encoding, labels, layers, weights and a checksum | [ADR-0012](../adr/0012-network-file-format.md) |
| Memory leaks? | No manual `new`/`delete`; `make debug` runs the tests under AddressSanitizer and UBSan, the thread pool was checked with ThreadSanitizer | AGENTS.md, ADR-0024 |

## Project
| Question | Answer | Details |
| --- | --- | --- |
| Who did what? | Main author of each part from `git log` | [business-plan.md](../business/business-plan.md) §8 |
| What would you do differently? | Run the full pipeline and profile on the first days | [retrospective](../retrospective/retrospective.md) |
| Legal risks? | Never publish the dataset; no license granted; no personal data | [risk-analysis.md](../legal/risk-analysis.md) |
| What does it cost? | Under two hours of laptop compute for the final network; the rest is people's time | [costs.md](../business/costs.md) |
