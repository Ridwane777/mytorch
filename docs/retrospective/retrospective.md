# Retrospective: what we could have done better, technically

Written on 2026-10-27, after the final network ([10-final-network.md](../performance/10-final-network.md)).
Organised by component. For each choice: what we did, the alternative we would name today, the
threshold where it becomes the better choice, and its own cost. Numbers come from the measurements
and the ADRs; a gain we did not measure is written "not measured", with how to measure it.

## 1. In one sentence
The library is correct (every gradient checked) and reproducible (bit-identical on any machine and
thread count), and the network reaches 96.2 % on unseen positions; the main technical debts are a
matrix product optimised late, an input encoding that cannot tell check from checkmate well, and
several choices measured on small samples.

## 2. Where the time goes (measured)
Profile of a training of the final network (2 epochs, 50,000 positions, `perf record -e task-clock`,
laptop i7-1355U, one thread, 2026-10-27; not scripted in the repository):

| Part | Share of the time |
| --- | --- |
| The three matrix products of the dense layers | 87.4 % |
| Adam's update | 5.8 % |
| L1/L2 penalty of `Dense`, computed at every batch even when it is zero | 1.6 % |
| FEN encoding | under 0.1 % |

Every alternative below that targets speed is judged against this table.

## 3. Linear algebra and computation

| What we did | Alternative | When it is better | Its cost |
| --- | --- | --- | --- |
| Our own `Matrix`, a plain triple loop until ADR-0024; then blocked and threaded: 4.2 to 4.7 times faster on one thread, up to 23 times on 12 ([12-matmul.md](../performance/12-matmul.md)) | **Eigen behind our `Matrix` class** (same interface, Eigen only for the products). The staff allowed it orally, which answers the question left open by ADR-0003, but the subject PDF does not say so. Speed against our blocked product: not measured (a benchmark on the 901 → 256 shapes would settle it) | As soon as the permission is written and training time limits the experiments, which was our case from the start | A dependency (header-only); a penalty risk while the permission is only oral; less computation of our own to defend; results no longer bit-identical across thread counts and machines (its order of summation depends on the vectorisation). Lesson: get oral permissions that touch a subject constraint in writing |
| The blocked product came last, after the trials | **Profile and optimise first**: the table of section 2 says the products are almost all of the time | Whenever experiments are repeated | About one day, done earlier |
| Kernels written in plain C++, vectorised by the compiler only. `G·Wᵀ` (a dot-product kernel) reaches 16.9 GFLOP/s on 12 threads, against 27 for the two others | A **register-tiled micro-kernel** (4 rows × 8 columns kept in registers, AVX2 intrinsics), and `G·Wᵀ` rewritten as a blocked product on a transposed copy of W | Products above 87 % of the time, as here | Harder code to read and defend, one kernel per instruction set; must keep the same order of summation to stay bit-identical. Gain not measured |
| `double` everywhere (ADR-0003) | **`float`** for weights and products, finite-difference tests kept in `double` | When training time is the bottleneck and bit-identical results with the earlier networks are no longer needed | Every saved network and reference result changes; precision of the gradient checks to re-establish |
| Rows split into equal ranges, one per thread: 2.4 times faster training on 12 threads only, because the laptop has 2 fast and 8 slow cores and every product waits for the slowest | **Dynamic split**: many small ranges taken by the threads as they finish | On hybrid processors (performance + efficiency cores) | A shared counter per product; results stay identical (each element still computed by one thread). Gain not measured |
| The penalty is computed at every batch, 1.6 % of the time, even with no L1/L2 | Skip it when both coefficients are zero | Always | A two-line change |

## 4. Training loop

| What we did | Alternative | When it is better | Its cost |
| --- | --- | --- | --- |
| Parallelism inside each product only, batch of 64: only 64 rows to share | **Data parallelism**: each thread trains on a part of the batch, gradients summed | More cores than rows per batch, or a cluster | The sum of the gradients depends on the thread count: results no longer bit-identical (ADR-0024 rejected it for this reason) |
| Batch of 64 | A larger batch (256 or more) shares better across threads | When the hardware has many cores | Changes the training dynamics: learning rate and epochs to tune again, possibly a worse network |
| The network file does not store Adam's moments (ADR-0012) | Save the optimizer state with the network | To resume a long training exactly, or to fine-tune | A new file version and its tests |
| No regularisation in the final network: early stopping only. The training loss (0.072) is below the validation loss (0.107) at the best epoch | Try L2 or dropout, already implemented ([08-overfitting.md](../performance/08-overfitting.md)), on the full dataset | When the validation loss stops before the training loss, which is the case | One full training per setting (about 1 h 45) |

## 5. Input encoding

| What we did | Alternative | When it is better | Its cost |
| --- | --- | --- | --- |
| `fen-attacks-901`: pieces, side, castling and attacked squares; best of four encodings (67.2 % against 62.0 % for 773 on a sample, ADR-0016). 91 % of the remaining errors are check against checkmate of the same side (2,505 of 2,754) | **Add the number of legal moves of the king in check**, or a plane of the squares it can reach. Checkmate is exactly "in check and no legal move", which the attack map does not encode | As soon as the errors are mostly check/checkmate, which is the case | Chess rules move further into the encoding (less general); a new encoding means retraining. Gain not measured |
| The attack map ignores pins (ADR-0016) | Pin-aware attacks | Same threshold: a pinned defender changes whether the king can escape | More rules in the encoder |
| Encoding written with plain loops over the board | Bitboards (64-bit masks per piece type) | Only if encoding mattered: it is under 0.1 % of the time (section 2) | Not worth it here: a measured reason not to do it |

## 6. Network architecture

| What we did | Alternative | When it is better | Its cost |
| --- | --- | --- | --- |
| Dense network 901 → 256 → 128 → 5 (ADR-0023); 512 → 256 not significantly better over five seeds ([09-trials.md](../performance/09-trials.md)) | A **convolutional network** treating the board as an 8 × 8 image with piece channels (bonus #29), or residual connections (#31) | When a larger dense network stops improving, as the trials suggest: a convolution sees lines and neighbourhoods with far fewer weights | Convolution forward and backward to write and check by finite differences; slower training on CPU. Gain not measured |

## 7. Data and evaluation

| What we did | Alternative | When it is better | Its cost |
| --- | --- | --- | --- |
| Encodings (ADR-0016) and class imbalance (ADR-0018) compared on 20,000 positions with one small network, because training was slow and memory-bound then | Redo the decisive comparisons on the full dataset | Now: a full training fits in under two hours since ADR-0021 and ADR-0024 | About 1 h 45 per setting |
| One seed per run; the seed spread was measured only in the trials (±0.3 to ±0.6 points of macro F1) | Three seeds per decisive comparison, or k-fold cross-validation (#39) | When two options differ by less than about one point | Three to k times the compute |
| Two label-preserving symmetries (mirror, colour flip): +4.5 points of accuracy (ADR-0019) | Nothing more to take from symmetries; more data would come from new positions (our generator, `bonus/my_torch_positions`, ADR-0015) | If the remaining errors need more mates: the classes are kept at their natural share (ADR-0018) | Generated positions are sparse and differ from the dataset's distribution |
| Some times were measured on a shared VPS | Measure times on one dedicated machine only; use shared machines for results, which are bit-identical | Every published time | Slower turnaround |

## 8. Network file and data pipeline

| What we did | Alternative | When it is better | Its cost |
| --- | --- | --- | --- |
| Text network file, 2.7 times larger than raw binary (ADR-0012); 5.4 MB for the final network | Binary format, still versioned and checksummed | From about 5 million parameters (GitHub's file limits) | Not readable by hand any more; a new version and its tests |
| The full pipeline (split → augment → train → score) came late. On the full dataset, encoding the whole training file at once would have needed about 10.4 GB, so the data path was redone (ADR-0021) | Run the thinnest end-to-end training on the **full** dataset first, even with a one-layer network | As soon as the data is larger than a quick test: from day 1 (723,019 positions) | Half a day at the start |

Note: ADR-0020 rejects LAPACK / Eigen as "Forbidden: no linear algebra library (ADR-0003)". ADR-0003
says the opposite: the subject is silent on linear algebra libraries. Accepted ADRs are not rewritten,
so the correction is recorded here.

## 9. What we would keep without hesitation
- **A finite-difference test for every gradient**: it caught the errors before training did.
- **Bit-identical results** whatever the machine and thread count: every optimisation is checked by
  comparing network files byte by byte, and two machines can share the work.
- **Sanitizers** (`make debug`: AddressSanitizer, UBSan) and ThreadSanitizer for the thread pool.
- **Data streamed batch by batch** (ADR-0021): 430 MB peak on 2.3 million training lines.
- **The test split never used to choose**, and a script and a graph for every number.
- **ADRs written when the decision is made**, with their ❌.

## 10. Next technical actions
1. The legal-king-moves feature against the check/checkmate confusion (section 5), measured on the
   validation split.
2. Skip the zero penalty, then the dynamic split and the `G·Wᵀ` kernel (section 3), each measured with
   `scripts/bench_matmul.sh`.
3. A benchmark of Eigen against our product on the network's shapes, to answer section 3 with a number.
4. L2 or dropout on the full dataset (section 4), and the decisive comparisons again at full scale
   (section 7).
5. A convolutional network (#29) against the dense one.
