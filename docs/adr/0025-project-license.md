# ADR-0025: No license, all rights reserved

- **Status**: Proposed
- **Date**: 2026-10-27
- **Deciders**: Ridwane777 (issue #20), pending agreement of Meric-BG

## Context
The repository has no license file. The code is a work of collaboration of the two members of the
pair (CPI art. L113-3): any license needs both. It is a graded the school project, kept in a private
repository of the school's organisation, and the rights of the school on student work are not settled
in our documents. The library is generic, so a permissive license would let it be reused elsewhere,
but nobody outside the school needs it today.

## Options considered
| Option | Pros | Cons |
| --- | --- | --- |
| **No license, all rights reserved (chosen)** | Nothing to grant that we may not own; matches a private graded project; can be changed later by both authors | Nobody may reuse the code, not even as an example; contributions from outside are impossible |
| MIT | Short, permissive, lets students and teachers reuse the library | Granted for ever on the published versions; may conflict with the the school rules; makes copying by other students easy |
| GPL v3 | Derived works stay open | Same uncertainty about the the school rules; heavy for a teaching library |
| Apache 2.0 | Permissive with a patent grant | Same uncertainty; the patent clause has no use here |

## Decision
No `LICENSE` file: all rights reserved by the two authors. The legal documents
(`docs/legal/licenses.md`, `terms-of-use.md`) state it.

## Consequences
- ✅ No right granted that we might not hold; nothing to undo if the the school rules restrict it.
- ✅ Consistent with the private repository and with the risk analysis (R2).
- ❌ The library cannot be reused by anyone, which also blocks the teaching use of the business plan
  until a license is chosen.
- ❌ A license chosen later only covers the versions published after it.
- Follow-up: revisit (new ADR) before any public release or any use outside the school, after checking
  the the school rules; MIT would then be the natural candidate for the library.
