# ADR-0025: No license, all rights reserved

- **Status**: Superseded by [ADR-0048](0048-mit-license.md)
- **Date**: 2026-10-27
- **Deciders**: Ridwane777 (issue #20)

## Context
The repository has no license file. The code is a work of collaboration of the two members of the
pair (CPI art. L113-3): any license needs both. At the time it was kept in a private
repository, and the rights of every party on the work were not settled in our documents. The library is generic, so a permissive license would let it be reused elsewhere,
but nobody outside the team needed it then.

## Options considered
| Option | Pros | Cons |
| --- | --- | --- |
| **No license, all rights reserved (chosen)** | Nothing to grant that we may not own; matches a private project; can be changed later by both authors | Nobody may reuse the code, not even as an example; contributions from outside are impossible |
| MIT | Short, permissive, lets learners and teachers reuse the library | Granted for ever on the published versions; granted before ownership is settled |
| GPL v3 | Derived works stay open | Same uncertainty about ownership; heavy for a teaching library |
| Apache 2.0 | Permissive with a patent grant | Same uncertainty; the patent clause has no use here |

## Decision
No `LICENSE` file: all rights reserved by the two authors. The legal documents
(`docs/legal/licenses.md`, `terms-of-use.md`) state it.

## Consequences
- ✅ No right granted that we might not hold; nothing to undo if ownership turns out restricted.
- ✅ Consistent with the private repository and with the risk analysis (R2).
- ❌ The library cannot be reused by anyone, which also blocks the teaching use of the business plan
  until a license is chosen.
- ❌ A license chosen later only covers the versions published after it.
- Follow-up: revisit (new ADR) before any public release or any wider use, after settling
  ownership; MIT would then be the natural candidate for the library.
