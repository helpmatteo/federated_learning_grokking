# Plans

Working plans for this project, moved into the repo on 2026-08-17 (they previously
lived in `~/.claude/plans/` on the old server, outside the tree, and PROGRESS.md
referenced them by paths that no longer resolved).

## Open

| plan | what it is |
|---|---|
| [`in-the-original-exp2-slowdown-ratio-png-greedy-bentley.md`](in-the-original-exp2-slowdown-ratio-png-greedy-bentley.md) | **The one live plan.** A second α per setup for `paper/exp2_slowdown_ratio_*.png`, so each panel carries two lines instead of one. The builder and the plotting change are **done and committed** (`5400ebc`, `f32344b`); what remains is execution — **87 of 105 specs**, launched twice on 2026-08-14 and killed both times. Read its execution section first: it was written for the 8× L4 box and carries a note on what changed. |

## Closed

Kept for their reasoning, not as work lists. PROGRESS.md's convention is that closed
plans stay readable because the decision rules in them are cited by RESULTS.md.

| plan | what it settled |
|---|---|
| [`closed/plan-all-that-needs-valiant-hamster.md`](closed/plan-all-that-needs-valiant-hamster.md) | The multi-setup campaign — exp2 + exp3b on all six setups plus the anchor redo. Ran to completion. Its **"Future work — deferred, not abandoned"** table is still the source for PROGRESS.md's *Next* block, so this is the closed plan most worth re-reading. |
| [`closed/plan-all-that-needs-nested-seal.md`](closed/plan-all-that-needs-nested-seal.md) | The boundary campaign — established that v1's K=97 breakdown was a budget artifact, and set the `t_memo(K) + delay` budgeting rule that governs everything since. |
| [`closed/gate-a-closeout.md`](closed/gate-a-closeout.md) | Gate A — per-setup cliffs and working points, setup C's capacity question, federated MNIST, and the K=50 AdamW diagnosis. |
| [`closed/i-want-to-replicate-merry-badger.md`](closed/i-want-to-replicate-merry-badger.md) | The v2 build plan (superseded by nested-seal). Its Phase 0 is a verified write-up of **11 bugs**, two of which silently corrupted results — the Dirichlet class-count drop and the dataset cache-key collision. Both are fixed; the analyses are not repeated anywhere else. |
