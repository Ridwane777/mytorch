# Demo

```bash
sh scripts/demo.sh              # waits for Enter between steps
DEMO_AUTO=1 sh scripts/demo.sh  # all at once: 38 s on the laptop (i7-1355U), tests included
```

It needs no dataset: the positions are [demo_positions.txt](demo_positions.txt) (written by us) and
positions produced by our own generator. The the school dataset is shown separately if asked, from the
local `dataset/` folder, never pushed ([licenses.md](../legal/licenses.md)).

| Step | What it shows | What to say |
| --- | --- | --- |
| 1. Build | `make`, then the bonus tools | C++20 only, no dependency in the executables (ADR-0002, `docs/dependencies.md`) |
| 2. Pushed network | `--predict my_torch_network_final.nn` on ten famous positions (start, Italian game, Fool's mate, Scholar's mate, back-rank mates, smothered mate, checks of both colours): **10 right out of 10** | One answer per line, in file order; 96.2 % on the 71,555 test positions, and its weakness is check against checkmate ([final-network.md](../performance/final-network.md)) |
| 3. Oracle | `bonus/my_torch_oracle` agrees with the 10 expected labels | The oracle is a rule-based checker used to verify data, never to answer for the network (ADR-0006) |
| 4. Generator | A blank network from `configs/basic_network.conf` | The configuration format (ADR-0013, `docs/configuration.md`) |
| 5. Training | 3,000 generated positions, early stopping after 9 epochs, then the network answers "Nothing" to every demo position | **On purpose**: 76 % of these positions are "Nothing", so a small network trained on little data learns the majority class. This is why the final network uses the whole dataset with augmentation (ADR-0019) |
| 6. Errors | Missing network, missing file, bad FEN: message on stderr, exit code 84 | Required by the subject; errors are exceptions caught in `main` (AGENTS.md) |
| 7. Tests | 206 tests, gradients checked by finite differences | Every gradient has its finite-difference test |

Numbers from the run of 2026-10-27; the training in step 5 is seeded, so it prints the same lines
each time.
