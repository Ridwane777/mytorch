# Checking the dataset labels with the oracle

## Goal

Check that the dataset labels are correct before training a network on them. A wrong label
teaches the network an error. Tool: the rule-based oracle (#37, ADR-0006), independent of the
network.

## Method

1. Each line is read by the analyzer's parser (#13, ADR-0005).
2. The oracle classifies the position: it checks whether the king of the side to move is attacked
   and whether a legal move remains.
3. The verdict is compared with the line's label. Stalemates and illegal positions are counted
   separately, because they have no label in the subject.

Reproduce:

```
make -C bonus
./bonus/my_torch_oracle dataset/*/*.txt
```

## Machine

Intel Core i7-1355U (13th generation, 12 threads, only one used), Linux, g++ 13.3, `-O2`.

## Results (measured on 2026-10-23)

| Measure | Value |
| --- | --- |
| Positions read | 723,019 |
| Agreement with the label | 723,019 (100.00%) |
| Disagreements | 0 |
| Stalemates | 0 |
| Illegal positions | 0 |
| Total time | 5.2 s |
| Peak memory | 25 MB |

## Conclusion

The 723,019 labels are correct according to the rules of the game. The dataset contains no
stalemate, which confirms that a "stalemate" class (#24) cannot be learned from this data alone. We
will have to generate positions and label them with the oracle.
