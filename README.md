# Logos

**A Semantic Consensus Protocol for Heterogeneous Distributed Systems**

[![Go Version](https://img.shields.io/badge/go-1.22+-00ADD8?style=flat&logo=go)](https://golang.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-research%20prototype-orange)]()

---

> *"Traditional consensus asks: do all nodes hold the same bytes?
> Logos asks: do all nodes mean the same thing?"*

---

## The Problem

Every distributed consensus algorithm — Raft, Paxos, Viewstamped Replication, PBFT — defines agreement as **bit-identical state**. Node A and Node B agree if and only if their byte sequences match exactly.

This is correct and sufficient for homogeneous systems. But modern distributed infrastructure is increasingly heterogeneous:

- **Rolling deployments** where nodes run different schema versions simultaneously
- **Multi-language services** that serialize the same logical state differently (JSON field ordering, float precision, encoding variants)
- **Edge/IoT clusters** where sensors report equivalent readings with different floating-point precision
- **Federated AI systems** where nodes hold equivalent knowledge in different embedding spaces

In all these cases, exact consensus fails or requires expensive serialization normalization — even when the nodes genuinely agree on meaning.

## The Insight

Logos introduces **fuzzy consensus**: a modified Raft protocol where nodes can reach agreement on *semantically equivalent* (but syntactically distinct) state, using cosine similarity thresholds on learned vector embeddings.

```
Traditional:  commit iff |votes| ≥ ⌈(n+1)/2⌉  AND  all votes byte-identical
Logos:     commit iff |votes| ≥ ⌈(n+1)/2⌉  AND  CosineSim(any two votes) ≥ θ
```

A **reconciliation phase** then aligns byte-level state after commit, using one of three strategies: leader authority, semantic merge, or vector majority (closest to centroid wins).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LogosNode                                │
│                                                                     │
│  ┌─────────────┐   ┌──────────────────┐   ┌─────────────────────┐  │
│  │  Raft Core  │   │  Semantic Layer  │   │   Drift Monitor     │  │
│  │             │◄──┤                  │   │                     │  │
│  │  - Leader   │   │  - Embedder      │   │  - Fingerprint      │  │
│  │    election │   │  - Similarity    │   │    sampling         │  │
│  │  - Log repl │   │    index         │   │  - Angular drift    │  │
│  │  - Commit   │   │  - Centroid      │   │    computation      │  │
│  │    index    │   │    computation   │   │  - Outlier detect   │  │
│  └──────┬──────┘   └─────────┬────────┘   └──────────┬──────────┘  │
│         │                    │                        │             │
│  ┌──────▼──────────────────▼─┐           ┌──────────▼──────────┐  │
│  │       FuzzyQuorum          │           │    Reconciler       │  │
│  │                            │           │                     │  │
│  │  Builds similarity graph   │           │  StrategyLeader     │  │
│  │  Finds largest coherent    │           │  StrategyMerge      │  │
│  │  cluster via BFS           │           │  StrategyVector     │  │
│  │  Returns consensus vector  │           │  MajorityMajority   │  │
│  └────────────────────────────┘           └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Overview

| Component | Package | Responsibility |
|---|---|---|
| **Embedder** | `pkg/semantic` | Converts payloads to 128-dim unit vectors via LSH projection |
| **Index** | `pkg/semantic` | Multi-probe LSH index for nearest-neighbor log search |
| **FuzzyQuorum** | `pkg/quorum` | Similarity-graph-based quorum evaluation |
| **Raft Core** | `pkg/consensus` | Modified Raft with semantic log entries |
| **DriftMonitor** | `pkg/consensus` | Continuous cluster-wide semantic divergence monitoring |
| **Reconciler** | `pkg/consensus` | Post-commit byte-level state alignment |
| **LogosNode** | `pkg/node` | Top-level API, wires all components together |

---

## How It Works

### 1. Semantic Embedding

Every payload is converted to a 128-dimensional unit vector using Logos's deterministic LSH projection scheme:

```go
embedder := semantic.NewEmbedder(clusterKey) // same key across all cluster nodes
result := embedder.Embed([]byte(`{"user_id": 42, "action": "login"}`))
// result.Vec: [128]float64 on the unit sphere
// result.Hash: SHA-256 for exact-match fast path
```

The embedding is *locality-sensitive*: similar payloads produce geometrically close vectors. The cluster key ensures all nodes share the same embedding space — a prerequisite for cross-node similarity comparisons.

### 2. Fuzzy Quorum

When the leader replicates a log entry, followers embed their local copy and participate in a quorum round. Instead of checking byte equality, the quorum engine builds a similarity graph:

```
Node A: embed_A  ─── sim(A,B)=0.97 ───► Node B: embed_B
   │                                          │
   └─── sim(A,C)=0.61 ──► Node C: embed_C ◄──┘
                           sim(B,C)=0.58
```

In this example, A and B form a coherent cluster (0.97 ≥ θ=0.92). If |{A,B}| ≥ quorumSize, the round closes with `OutcomeAchieved`. Node C is an outlier — its state is semantically diverged and will be flagged for reconciliation.

### 3. Consensus Vector

When a fuzzy quorum succeeds, the centroid of the winning cluster's embeddings becomes the **consensus vector** — the canonical semantic representation of the agreed-upon state. This vector is:

- Stored in the commit log
- Used to select the canonical payload during reconciliation  
- Compared against future leader election candidates

### 4. Reconciliation

After a fuzzy commit (`ExactAgreement=false`), the Reconciler produces a canonical payload:

| Strategy | Behavior | When to use |
|---|---|---|
| `StrategyLeaderAuthority` | Adopt leader's exact bytes | Trusted homogeneous encoding, schema migration |
| `StrategyVectorMajority` | Pick variant closest to consensus centroid | General purpose, no domain knowledge |
| `StrategySemanticMerge` | Call user-supplied merge function | Domain-specific merging (e.g., CRDT-style) |

### 5. Drift Monitoring

The DriftMonitor continuously samples each node's **semantic fingerprint** (centroid of its last N log entries) and computes cluster-wide angular drift:

```
drift level   │  max angular distance  │  action
──────────────┼────────────────────────┼─────────────────────────────
NONE          │  < 0.01 rad            │  no action
MINOR         │  0.01–0.30 rad         │  log debug
WARNING       │  0.30–0.60 rad         │  log warning, emit alert
CRITICAL      │  > 0.60 rad            │  trigger reconciliation
```

---

## Quickstart

### Requirements

- Go 1.22+
- No external dependencies beyond the Go standard library for core logic

### Build

```bash
git clone https://github.com/PrimelPJ/logos
cd logos
go build ./...
go build -o bin/logos ./cmd/logos
```

### Explore the embedding space

```bash
./bin/logos embed \
  --a '{"user": 42, "action": "login"}' \
  --b '{"user": 42, "action": "LOGIN"}'

# a:           "{\"user\": 42, \"action\": \"login\"}"
# b:           "{\"user\": 42, \"action\": \"LOGIN\"}"
# similarity:  0.983421
# distance:    0.181823 rad (10.42°)
# equivalent:  true (threshold=0.92)
```

### Start a cluster node

```bash
# Generate a shared cluster key (all nodes must share this)
CLUSTER_KEY=$(openssl rand -hex 32)

# Node 1
./bin/logos start --id=node1 --cluster-key=$CLUSTER_KEY \
  --cluster-size=3 --addr=:7700

# Node 2 (separate terminal)
./bin/logos start --id=node2 --cluster-key=$CLUSTER_KEY \
  --cluster-size=3 --addr=:7701

# Node 3 (separate terminal)
./bin/logos start --id=node3 --cluster-key=$CLUSTER_KEY \
  --cluster-size=3 --addr=:7702
```

### Propose a value (Go API)

```go
package main

import (
    "context"
    "log"
    "github.com/PrimelPJ/logos/pkg/node"
    "go.uber.org/zap"
)

func main() {
    logger, _ := zap.NewProduction()
    cfg := node.DefaultConfig("node1", []byte("my-32-byte-cluster-key----------"))
    n := node.New(cfg, myStateMachine{}, logger)

    ctx := context.Background()
    n.Start(ctx)

    // Propose with semantic deduplication
    index, wasDeduplicated, err := n.Propose(ctx, []byte(`{"event":"user_signup","id":99}`))
    if err != nil {
        log.Fatal(err)
    }
    log.Printf("committed at index=%d deduplicated=%v", index, wasDeduplicated)

    // Search for semantically similar past proposals
    results := n.SemanticSearch([]byte(`{"event":"user_registered","id":99}`), 5)
    for _, r := range results {
        log.Printf("similar entry: id=%s similarity=%.4f", r.ID, r.Similarity)
    }
}
```

---

## Novelty & Prior Art

Logos introduces three ideas not present in existing distributed systems literature:

### 1. Semantic Quorum
Paxos, Raft, PBFT, and their variants all define agreement as byte equality. Logos is the first (to our knowledge) consensus protocol to allow agreement over a *semantic equivalence class* rather than an exact value. The fuzzy quorum algorithm is new.

### 2. Semantic Log Indexing
Traditional Raft logs are opaque byte arrays. Logos's log is simultaneously an append-only sequence AND a nearest-neighbor search index, enabling semantic lookup ("find all entries like this") as a first-class primitive.

### 3. Angular Drift as a Health Metric
Distributed systems typically measure health via latency, error rates, and replication lag. Logos introduces **semantic drift angle** as an orthogonal health dimension — measuring *divergence in meaning* rather than divergence in timing or availability.

### Related Work

| System | Relation |
|---|---|
| Raft (Ongaro & Ousterhout, 2014) | Logos's consensus backbone |
| CRDTs | Logos's semantic merge is inspired by CRDT convergence, but operates at the consensus layer |
| SimHash / LSH | Logos's embedding scheme uses random projection LSH |
| RAG / Vector Databases | Logos borrows similarity search primitives but applies them to consensus, not retrieval |

---

## Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| `SimilarityThreshold` | `0.92` | Cosine similarity threshold for fuzzy equivalence |
| `DriftWarnAngle` | `0.30 rad` | Angular drift above which to emit a WARNING |
| `DriftCriticalAngle` | `0.60 rad` | Angular drift above which to trigger reconciliation |
| `FingerprintDepth` | `50` | Number of recent log entries used to compute a node fingerprint |
| `ReconcileStrategy` | `VectorMajority` | How to select canonical payload after fuzzy commit |
| `ElectionTimeout` | `150ms` | Raft election timeout |
| `HeartbeatInterval` | `50ms` | Raft heartbeat interval |

---

## Limitations & Future Work

- **Transport layer**: The gRPC transport (in `pkg/transport/`) is stubbed. A full implementation would use Protocol Buffers for all RPC messages. See `docs/transport.md`.
- **Neural backend**: The current embedder uses deterministic LSH projection. A pluggable neural backend (e.g., a small sentence transformer) would yield higher-quality semantic similarity at higher computational cost.
- **Formal verification**: The fuzzy quorum algorithm has not been formally verified for safety under all failure scenarios. In particular, split-brain scenarios with θ near the boundary require more rigorous analysis.
- **Persistence**: Log persistence and snapshots are not yet implemented. All state is in-memory.
- **Benchmarks**: See `docs/benchmarks.md` for preliminary latency vs. exact consensus comparisons.

---

## Project Structure

```
logos/
├── cmd/
│   └── logos/       # CLI entry point
├── pkg/
│   ├── semantic/       # Embedding, similarity, LSH index
│   │   ├── embedder.go
│   │   └── index.go
│   ├── quorum/         # FuzzyQuorum round management
│   │   └── fuzzy.go
│   ├── consensus/      # Modified Raft, drift monitor, reconciler
│   │   ├── raft.go
│   │   ├── drift.go
│   │   └── reconcile.go
│   └── node/           # Top-level LogosNode API
│       └── node.go
├── docs/
│   ├── architecture.md
│   ├── quickstart.md
│   ├── transport.md
│   └── benchmarks.md
├── go.mod
└── README.md
```

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Author

Built by [@PrimelPJ](https://github.com/PrimelPJ) · [primelj.dev](https://primelj.dev)
