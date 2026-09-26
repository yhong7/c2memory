# C2Mem paper implementation

This repository provides the C2Mem runtime architecture and the proposed C2Mem
online-memory backend. Run it with:

```bash
python -m src.main \
  --dataset hotpotqa \
  --graph autogen \
  --mas-memory-backend c2mem \
  --mas-memory-path .db/c2mem/hotpotqa-autogen \
  --mas-memory-topk 2 \
  --concurrency 1
```

The implementation is in `src/memory/online/c2mem.py`. The runtime-facing
adapter remains in `src/memory/native_mas_memory.py`, so the existing graph,
dataset, evaluator, trace, and result pipelines are unchanged.

## Execution lifecycle

1. `begin_task` resets task-local execution prefixes and exposure events.
2. `retrieve_contexts` retrieves behavior-tree leaves, evaluates support,
   computes commitment and competence, calibrates the support distribution,
   and returns per-agent snippets.
3. `observe_trace` extends the execution prefix and repeats calibrated retrieval
   after each agent trace.
4. `record_task` extracts one reusable episode, applies the task-level posterior
   update, inserts the episode, and persists the tree and audit state.

## Behavioral memory representation

Each completed trajectory is summarized into:

- `content_summary`;
- an ordered `behavior_pattern` of the form
  `Trigger -> Early -> Feedback -> Transition -> Stop`;
- `outcome_summary`;
- per-agent `reusable_experience` rules.

Only the behavior pattern is embedded for tree insertion and retrieval. Task
content and the raw trajectory remain attached to the record for grounding.

Ordinary leaf nodes contain behaviorally compatible episodes. Retrieval and
calibration run throughout a task; their injection weights are accumulated and
normalized into the task exposure distribution. A new episode is inserted into
the ordinary leaf with the greatest positive task exposure. If the task assigns
no positive exposure to any ordinary leaf, a new top-level branch is created.
Episode-to-node cosine similarity is not used as a second insertion rule.
A heterogeneous leaf is split with deterministic farthest-seed bipartition.
Parent nodes remain available for tree distance, while retrieval operates on
leaves.

## Equation-to-code map

| Paper | Meaning | Implementation |
| --- | --- | --- |
| (1) | Beta posterior and node posterior mean | `_posterior_parameters`, `_posterior_mean` |
| (2) | Behavior-embedding cosine similarity | `cosine`, `_retrieve_candidates` |
| (3) | Joint support over retrieved nodes and node `0` | `_evaluate_support`, `_normalize_weights` |
| (4) | Ordinary-memory support `m_t = 1 - alpha_t(0)` | `compute_c2_metrics` |
| (5) | Conditional ordinary-node weights | `compute_c2_metrics` |
| (6) | Normalized retrieval entropy | `compute_c2_metrics` |
| (7) | Expected normalized tree distance | `compute_c2_metrics`, `normalized_tree_distance` |
| (8) | Memory commitment | `compute_c2_metrics` |
| (9) | Posterior-mean memory competence | `compute_c2_metrics` |
| (10) | Posterior probability `c_t^-` | Monte Carlo loop in `compute_c2_metrics` |
| (11) | Memory-induced overconfidence risk | `compute_c2_metrics` |
| (12) | Node competence-deficit contribution | `calibrate_support_weights` |
| (13) | Exponential support calibration | `calibrate_support_weights` |
| (14) | Exposure-weighted Beta evidence update | `_update_retrieval_evidence` |

Equation (6) uses the full retrieved ordinary set in its denominator, including
candidates with zero evaluator weight. Equation (13) normalizes across the
augmented set, including node `0`; because `delta_t(0) = 0`, node `0` is not
directly attenuated.

The distributions persisted for each retrieval event are:

- `weights`: raw evaluator distribution `alpha_t`;
- `calibrated_weights`: Equation (13) distribution `tilde(alpha_t)`;
- `injection_weights`: calibrated distribution after the optional engineering
  injection threshold;
- `deficit_contributions`: Equation (12) values;
- `metrics`: Equations (4)-(11).

Only `injection_weights` are shown to agents and accumulated for the final
posterior update.

## Experience updating

Calibrated injection weights are accumulated across all retrieval steps in one
task and normalized once to obtain task exposure `rho_q`. The total evidence
added by one completed task is therefore exactly one:

```text
sum_v rho_q(v) = 1
a_v <- a_v + rho_q(v) y_q
b_v <- b_v + rho_q(v) (1 - y_q)
```

Node `0` is updated from its own exposure in the same way. Episode success or
failure is retained as record metadata, but it is not added directly to the
node's competence posterior: competence measures the observed utility of memory
when it influenced later execution. When a leaf splits, its accumulated
exposure evidence is distributed to children in proportion to member count;
the paper does not prescribe a unique split-evidence rule.

## Persistence and compatibility

- `records.jsonl`: extracted episodes and their assigned tree-node IDs.
- `state.json`: tree, posteriors, configuration, retrieval audits, last task
  exposure, and schema metadata.

The paper-aligned state schema is `c2mem-paper-v2`. Older `c2mem-v1` state is
rejected instead of silently mixing per-step raw-weight counts with the new
task-normalized calibrated exposure semantics. Use a new memory directory for
paper-aligned runs.

## Engineering configuration

The paper defines the formulas but does not fix these numerical choices:

| Variable | Default |
| --- | ---: |
| `C2MEM_COMMITMENT_LAMBDA` | `0.5` |
| `C2MEM_CALIBRATION_KAPPA` | `1.0` |
| `C2MEM_CALIBRATION_EPSILON` | `1e-12` |
| `C2MEM_INJECTION_THRESHOLD` | `0.0` |
| `C2MEM_POSTERIOR_SAMPLES` | `1000` |
| `C2MEM_SPLIT_MIN_SIZE` | `8` |
| `C2MEM_SPLIT_SIMILARITY_THRESHOLD` | `0.45` |
| `C2MEM_NODE_PRIOR_ALPHA` | `1.0` |
| `C2MEM_NODE_PRIOR_BETA` | `1.0` |
| `C2MEM_UNKNOWN_PRIOR_ALPHA` | `1.0` |
| `C2MEM_UNKNOWN_PRIOR_BETA` | `1.0` |
| `C2MEM_MAX_EVENTS` | `1000` |

The paper also leaves the exact normalized tree-distance function open. This
implementation adds a virtual root for separate top-level branches and divides
shortest-path length by the maximum path length implied by the two node depths.
