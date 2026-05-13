# Logos: Architecture Deep Dive

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [The Embedding Space](#the-embedding-space)
3. [Fuzzy Quorum Algorithm](#fuzzy-quorum-algorithm)
4. [Modified Raft Protocol](#modified-raft-protocol)
5. [Reconciliation Pipeline](#reconciliation-pipeline)
6. [Drift Monitoring](#drift-monitoring)
7. [Data Flow: End-to-End Proposal Lifecycle](#data-flow-end-to-end-proposal-lifecycle)
8. [Failure Modes & Safety Analysis](#failure-modes--safety-analysis)

---

## Design Philosophy

Logos's design is governed by three principles:

**1. Semantic transparency, byte-level accountability**
The system never silently substitutes one byte sequence for another. Every fuzzy commit is annotated with `ExactAgreement=false`, and the reconciliation log records which strategy was used and what drift existed before alignment. There are no hidden state transformations.

**2. Progressive degradation, not binary failure**
When nodes diverge semantically, Logos doesn't halt — it classifies divergence into four levels (NONE → MINOR → WARNING → CRITICAL) and applies proportional responses. The cluster keeps making progress while flagging the problem.

**3. Algebraic embedding alignment**
All nodes in a cluster share an identical embedding space (seeded by `clusterKey`). This is a prerequisite for cross-node cosine similarity to be meaningful. Logos enforces this at startup: a node that can't embed payloads in the cluster's shared space cannot participate.

---

## The Embedding Space

### Random Projection LSH

Logos uses a deterministic random projection scheme to map variable-length byte payloads onto a 128-dimensional unit sphere.

```
Input: payload ∈ {0..255}*
Output: v ∈ ℝ¹²⁸, ||v|| = 1
```

**Algorithm:**

```
1. Chunk payload into 32-byte windows W₀, W₁, ..., Wₖ
2. For each dimension d ∈ [0, 128):
     for each chunk c:
         dot = W_c · P[d]              (P[d] is the d-th row of projection matrix)
         raw[d] += tanh(dot) * 1/√(c+1) (position-weighted contribution)
3. v = raw / ||raw||                   (L2 normalize to unit sphere)
```

**Projection matrix P** is seeded deterministically from the cluster key:
```
P[d][j] = (SHA-256(clusterKey || d)[j] - 127.5) / 127.5
```
This ensures all cluster nodes generate identical projection matrices and therefore identical embeddings for identical inputs.

**Key properties:**
- Deterministic: same input always produces the same vector
- Locality-sensitive: small edits → nearby vectors, structural changes → distant vectors  
- Position-weighted: prefixes contribute more than suffixes (preserves log entry ordering semantics)
- Fast: O(|payload| × EmbeddingDim) — linear in payload length

### Fast Path: Exact Hash Match

If two embeddings have identical SHA-256 hashes, `IsEquivalent` returns `true` without computing cosine similarity. This makes the common case (exact byte match) O(1).

### Similarity Geometry

```
CosineSimilarity(a, b) = a · b   (since ||a|| = ||b|| = 1)

SemanticDistance(a, b) = arccos(a · b) ∈ [0, π]

Equivalence: sim(a,b) ≥ θ  ↔  dist(a,b) ≤ arccos(θ)

θ=0.92 → max distance = arccos(0.92) ≈ 0.40 rad ≈ 23°
```

---

## Fuzzy Quorum Algorithm

### Similarity Graph Construction

Given n votes with embeddings e₁, e₂, ..., eₙ:

```
Build adjacency matrix adj[n×n]:
  adj[i][j] = (CosineSim(eᵢ, eⱼ) ≥ θ)
```

### Connected Component Discovery

BFS over the adjacency matrix finds all connected components — groups of nodes that are mutually semantically coherent.

```
components = BFS(adj)
sort(components, key=len, order=descending)
largest = components[0]
```

### Quorum Decision

```
quorumSize = ⌈(clusterSize + 1) / 2⌉

if |largest| ≥ quorumSize:
    if maxAngularDrift(largest) > driftWarn:
        return OutcomeDegraded
    else:
        return OutcomeAchieved

elif |comp₀| ≥ quorumSize/2 AND |comp₁| ≥ quorumSize/2:
    return OutcomeSplit

else:
    return OutcomeFailed
```

### Early Termination

The quorum engine implements two fast paths:

**Fast commit:** As soon as `|votes| ≥ quorumSize` AND a coherent cluster of size ≥ `quorumSize` exists, the round closes immediately (doesn't wait for remaining votes).

**Fast abort:** If `|votes| + |remaining| < quorumSize`, quorum is mathematically impossible. The round closes with `OutcomeFailed` immediately.

---

## Modified Raft Protocol

Logos's Raft deviations from the Ongaro/Ousterhout paper:

### Deviation 1: Semantic Log Entries

Standard Raft log entry:
```
{Index, Term, Command []byte}
```

Logos log entry:
```
{Index, Term, Payload []byte, Embedding EmbedResult, CommittedAt time.Time, Reconciled bool}
```

The embedding is computed by the leader at propose time and replicated to followers. Followers MAY recompute the embedding independently to detect leader inconsistency.

### Deviation 2: Semantic Replication Check

In standard Raft, a follower appends an entry if:
```
(prevLogIndex, prevLogTerm) match
```

In Logos, an additional semantic check runs:
```
(prevLogIndex, prevLogTerm) match
AND CosineSim(leader_embed, follower_computed_embed) ≥ θ
```

If the second condition fails, the follower logs a `semantic_mismatch` event and requests the leader's raw payload (signaling potential data corruption or a byzantine leader).

### Deviation 3: Semantic Compatibility in Leader Election

In standard Raft, a candidate receives a vote if its log is at least as up-to-date as the voter's log:
```
(candidateTerm > voterLastTerm) OR (candidateTerm == voterLastTerm AND candidateIndex >= voterIndex)
```

Logos adds a semantic compatibility check:
```
standard_condition AND CosineSim(candidate_fingerprint, voter_fingerprint) ≥ θ_election
```

Where `θ_election = θ × 0.95` (slightly more permissive than commit threshold, because fingerprints are centroids of many entries and will naturally converge toward each other).

This prevents a semantically diverged node (perhaps due to a byzantine fault or corrupted snapshot) from winning an election even if it has the highest term.

### Deviation 4: Semantic Deduplication at Propose

Before accepting a proposal, the leader checks whether a semantically equivalent entry was recently committed:

```go
results := log.SemanticSearch(newEmbed, k=3, threshold=θ)
if len(results) > 0 && results[0].Similarity >= θ:
    return existingIndex, deduplicated=true, nil
```

This is an optional optimization (disabled by setting `DedupWindow=0`) but prevents semantic near-duplicate log spam in high-throughput scenarios.

---

## Reconciliation Pipeline

```
FuzzyCommit (ExactAgreement=false)
        │
        ▼
ReconcileRequest{
    LogIndex,
    ConsensusVector,  ← centroid of winning cluster's embeddings
    Variants[]        ← one payload per winning-cluster node
}
        │
        ▼
Strategy selection:
  ┌─────────────────────────────────────────────┐
  │  StrategyLeaderAuthority                    │
  │    → adopt leader's exact bytes             │
  │    latency: ~0ms                            │
  ├─────────────────────────────────────────────┤
  │  StrategyVectorMajority                     │
  │    → pick payload closest to ConsensusVector │
  │    → argmin_i dist(embed_i, consensusVec)   │
  │    latency: O(|variants|) cosine computes   │
  ├─────────────────────────────────────────────┤
  │  StrategySemanticMerge                      │
  │    → call user MergeFunc(variants, centroid) │
  │    → latency: application-defined           │
  └─────────────────────────────────────────────┘
        │
        ▼
Canonical payload broadcast to all cluster nodes
Log entry updated: Reconciled = true
```

### Custom Merge Functions

Applications register merge functions per payload type tag (first 4 bytes):

```go
n.RegisterMergeFunc("JSON", func(variants []Variant, consensus Vector) ([]byte, error) {
    // Custom JSON merge: union of all fields, latest value wins on conflict
    merged := mergeJSON(variants)
    return json.Marshal(merged)
})
```

---

## Drift Monitoring

### Fingerprint Computation

Each node periodically computes its **semantic fingerprint** — the centroid of its last N log entries' embeddings:

```
fingerprint = normalize(Σ embed(entry_i) for i in last_N_entries)
```

This fingerprint lives on the unit sphere. Two nodes with identical log suffixes will have identical (or very close) fingerprints. Two nodes whose logs have diverged will have distant fingerprints.

### Drift Measurement

Given fingerprints f₁, f₂, ..., fₙ:

```
centroid = normalize(Σ fᵢ)
distances[i] = arccos(fᵢ · centroid)
maxDist = max(distances)
meanDist = mean(distances)
stddev = stddev(distances)
outliers = {i : distances[i] > meanDist + 2σ}
```

### Alert Lifecycle

```
DriftMonitor samples every 2s
        │
        ├── DriftNone    → silent
        ├── DriftMinor   → debug log
        ├── DriftWarning → warning log + alertC emit
        └── DriftCritical → error log + alertC emit + reconcileC emit
                                                            │
                                                            ▼
                                               LogosNode.reconcileLoop()
                                                            │
                                                            ▼
                                               Snapshot transfer to outlier nodes
                                               (transport layer — TODO)
```

---

## Data Flow: End-to-End Proposal Lifecycle

```
Client
  │
  │  Propose(payload)
  ▼
LogosNode
  │  1. SemanticSearch(embed(payload)) → dedup check
  │  2. If no match: Propose to Raft
  ▼
consensus.Node (leader)
  │  3. embed(payload) → Entry{..., Embedding}
  │  4. Log.Append(entry)
  │  5. Replicate to followers via AppendEntries RPC
  ▼
consensus.Node (followers)
  │  6. Receive AppendEntries
  │  7. Re-embed payload locally
  │  8. Create Vote{NodeID, Embedding, RawHash}
  │  9. Send vote to quorum round
  ▼
quorum.Round
  │  10. AddVote() for each follower
  │  11. After quorumSize votes: TryClose()
  │  12. Build similarity graph
  │  13. BFS → largest coherent cluster
  │  14. |cluster| ≥ quorumSize? → OutcomeAchieved
  ▼
consensus.Node (leader)
  │  15. Advance commitIndex
  │  16. If ExactAgreement=false: enqueue ReconcileRequest
  │  17. Apply committed entries to StateMachine
  ▼
consensus.Reconciler (if needed)
  │  18. Select canonical payload via strategy
  │  19. Broadcast canonical payload
  │  20. Mark entry as Reconciled=true
  ▼
Client
     21. Receive (logIndex, deduplicated=false, nil)
```

---

## Failure Modes & Safety Analysis

### Split Brain (Network Partition)

If the cluster partitions into two groups of equal size, neither side can form a quorum. Both sides return `OutcomeFailed` and stop accepting proposals. This is identical to standard Raft behavior — fuzzy consensus does not weaken partition tolerance.

### Byzantine Fault

If a byzantine node crafts a payload with an embedding close to the honest nodes (sim ≥ θ) but with different semantics, it could potentially participate in a fuzzy quorum. This is a known limitation. Mitigations:
- Use signed votes (transport layer, TODO)
- Lower θ (stricter equivalence)
- Use StrategyLeaderAuthority (byzantine node's payload is never canonical)

Logos is not a BFT protocol and makes no safety guarantees against byzantine nodes.

### Semantic Threshold Misconfiguration

If θ is set too low (e.g., 0.5), unrelated payloads might be considered equivalent and a false quorum could form. The calibration suite (`docs/calibration.md`) provides tooling to find the right θ for a given payload distribution.

### Embedding Space Mismatch

If nodes are started with different cluster keys, their embedding spaces are incompatible. All pairwise similarities will be near 0, every quorum round will fail with `OutcomeFailed`. Logos detects this at startup and refuses to join a cluster with a mismatched key.
