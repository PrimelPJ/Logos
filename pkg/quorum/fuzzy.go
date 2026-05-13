// Package quorum implements Logos's FuzzyQuorum — a modified quorum protocol
// where agreement is reached when a majority of nodes hold semantically equivalent
// (not necessarily byte-identical) state, as determined by the semantic layer.
//
// Traditional quorum: |votes| >= ⌈(n+1)/2⌉ AND all votes are bit-identical.
// Fuzzy quorum:       |votes| >= ⌈(n+1)/2⌉ AND CosineSimilarity(any two votes) >= θ
//
// This enables consensus in heterogeneous systems where equivalent state may be
// encoded differently (e.g., JSON field ordering, floating-point precision variance,
// schema version mismatches across rolling deployments).
package quorum

import (
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/PrimelPJ/logos/pkg/semantic"
)

// VoteID uniquely identifies a vote within a quorum round.
type VoteID string

// Vote represents a single node's position in a quorum round.
type Vote struct {
	NodeID    string
	RoundID   uint64
	Term      uint64
	Embed     semantic.EmbedResult
	RawHash   [32]byte // SHA-256 of the original payload for audit
	Timestamp time.Time
	Signature []byte // TODO: integrate with transport/tls for signed votes
}

// Outcome describes the result of a quorum decision.
type Outcome int8

const (
	OutcomePending    Outcome = iota
	OutcomeAchieved           // Fuzzy quorum reached
	OutcomeFailed             // Could not reach quorum within timeout
	OutcomeSplit              // Votes clustered into irreconcilable groups
	OutcomeDegraded           // Quorum reached but with high semantic drift (warning)
)

func (o Outcome) String() string {
	switch o {
	case OutcomePending:
		return "PENDING"
	case OutcomeAchieved:
		return "ACHIEVED"
	case OutcomeFailed:
		return "FAILED"
	case OutcomeSplit:
		return "SPLIT"
	case OutcomeDegraded:
		return "DEGRADED"
	default:
		return "UNKNOWN"
	}
}

// QuorumResult is returned when a round closes.
type QuorumResult struct {
	Outcome          Outcome
	Round            uint64
	Term             uint64
	ConsensusVector  semantic.Vector  // centroid of the winning cluster's embeddings
	WinningCluster   []string         // NodeIDs in the winning quorum
	DriftReport      semantic.DriftReport
	MaxDriftAngle    float64          // radians
	ExactAgreement   bool             // true iff all winning votes are byte-identical
	ClosedAt         time.Time
}

// Round manages vote collection and quorum evaluation for a single consensus round.
type Round struct {
	mu          sync.Mutex
	id          uint64
	term        uint64
	clusterSize int
	threshold   float64          // cosine similarity threshold for equivalence
	driftWarn   float64          // max acceptable angular drift before OutcomeDegraded
	votes       []Vote
	closed      bool
	result      *QuorumResult
}

// NewRound creates a new consensus round.
//
//   - clusterSize: total number of nodes (n)
//   - threshold: cosine similarity threshold (e.g., 0.92)
//   - driftWarn: angular drift (radians) above which to emit OutcomeDegraded
func NewRound(id, term uint64, clusterSize int, threshold, driftWarn float64) *Round {
	return &Round{
		id:          id,
		term:        term,
		clusterSize: clusterSize,
		threshold:   threshold,
		driftWarn:   driftWarn,
	}
}

// AddVote registers a vote from a node. Thread-safe.
// Returns an error if the round is already closed or the vote is a duplicate.
func (r *Round) AddVote(v Vote) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.closed {
		return fmt.Errorf("round %d is closed", r.id)
	}
	if v.Term < r.term {
		return fmt.Errorf("stale vote from %s: vote term %d < round term %d", v.NodeID, v.Term, r.term)
	}
	for _, existing := range r.votes {
		if existing.NodeID == v.NodeID {
			return fmt.Errorf("duplicate vote from node %s in round %d", v.NodeID, r.id)
		}
	}
	r.votes = append(r.votes, v)
	return nil
}

// TryClose attempts to evaluate the quorum. If evaluation produces a non-pending
// outcome, the round is closed and the result is returned.
//
// This is called after each AddVote. The round closes as soon as either:
//   - A fuzzy quorum is achieved (fast-path commit)
//   - Enough votes are in that quorum is mathematically impossible (fast-path abort)
func (r *Round) TryClose() *QuorumResult {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.closed {
		return r.result
	}

	quorumSize := r.clusterSize/2 + 1
	remaining := r.clusterSize - len(r.votes)

	// Can we still reach quorum?
	if len(r.votes)+remaining < quorumSize {
		result := &QuorumResult{
			Outcome:  OutcomeFailed,
			Round:    r.id,
			Term:     r.term,
			ClosedAt: time.Now(),
		}
		r.closed = true
		r.result = result
		return result
	}

	// Not enough votes yet
	if len(r.votes) < quorumSize {
		return nil
	}

	// We have enough votes — find the largest semantically coherent cluster
	result := r.evaluate(quorumSize)
	if result.Outcome != OutcomePending {
		r.closed = true
		r.result = result
		return result
	}
	return nil
}

// ForceClose closes the round regardless of vote count. Used on timeout.
func (r *Round) ForceClose() *QuorumResult {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.closed {
		return r.result
	}

	quorumSize := r.clusterSize/2 + 1
	result := r.evaluate(quorumSize)
	if result.Outcome == OutcomePending {
		result.Outcome = OutcomeFailed
	}
	r.closed = true
	r.result = result
	return result
}

// evaluate runs the core fuzzy quorum algorithm.
// Algorithm:
//  1. Build similarity graph: edge(i,j) iff CosineSim(vote_i, vote_j) >= θ
//  2. Find largest connected component (greedy, O(n²) — n is small in practice)
//  3. If |component| >= quorumSize → OutcomeAchieved (or OutcomeDegraded)
//  4. Otherwise → OutcomeSplit or OutcomeFailed
func (r *Round) evaluate(quorumSize int) *QuorumResult {
	n := len(r.votes)
	if n == 0 {
		return &QuorumResult{Outcome: OutcomePending, Round: r.id, Term: r.term}
	}

	// Build adjacency via cosine similarity
	adj := make([][]bool, n)
	for i := range adj {
		adj[i] = make([]bool, n)
	}
	for i := 0; i < n; i++ {
		adj[i][i] = true
		for j := i + 1; j < n; j++ {
			sim := semantic.CosineSimilarity(r.votes[i].Embed.Vec, r.votes[j].Embed.Vec)
			if sim >= r.threshold {
				adj[i][j] = true
				adj[j][i] = true
			}
		}
	}

	// BFS to find connected components
	visited := make([]bool, n)
	var components [][]int

	for start := 0; start < n; start++ {
		if visited[start] {
			continue
		}
		var component []int
		queue := []int{start}
		visited[start] = true
		for len(queue) > 0 {
			cur := queue[0]
			queue = queue[1:]
			component = append(component, cur)
			for neighbor := 0; neighbor < n; neighbor++ {
				if !visited[neighbor] && adj[cur][neighbor] {
					visited[neighbor] = true
					queue = append(queue, neighbor)
				}
			}
		}
		components = append(components, component)
	}

	// Sort components by size descending
	sort.Slice(components, func(i, j int) bool {
		return len(components[i]) > len(components[j])
	})

	largest := components[0]
	if len(largest) < quorumSize {
		outcome := OutcomeFailed
		if len(components) >= 2 &&
			len(components[0]) >= quorumSize/2 &&
			len(components[1]) >= quorumSize/2 {
			outcome = OutcomeSplit
		}
		return &QuorumResult{
			Outcome:  outcome,
			Round:    r.id,
			Term:     r.term,
			ClosedAt: time.Now(),
		}
	}

	// Build consensus vector from winning cluster
	clusterVecs := make([]semantic.Vector, len(largest))
	clusterEmbeds := make([]semantic.EmbedResult, len(largest))
	nodeIDs := make([]string, len(largest))
	allExact := true

	referenceHash := r.votes[largest[0]].RawHash
	for i, idx := range largest {
		clusterVecs[i] = r.votes[idx].Embed.Vec
		clusterEmbeds[i] = r.votes[idx].Embed
		nodeIDs[i] = r.votes[idx].NodeID
		if r.votes[idx].RawHash != referenceHash {
			allExact = false
		}
	}

	consensusVec := semantic.Centroid(clusterVecs)
	drift := semantic.AnalyzeDrift(clusterEmbeds)

	outcome := OutcomeAchieved
	if drift.MaxDistance > r.driftWarn {
		outcome = OutcomeDegraded
	}

	return &QuorumResult{
		Outcome:         outcome,
		Round:           r.id,
		Term:            r.term,
		ConsensusVector: consensusVec,
		WinningCluster:  nodeIDs,
		DriftReport:     drift,
		MaxDriftAngle:   drift.MaxDistance,
		ExactAgreement:  allExact,
		ClosedAt:        time.Now(),
	}
}

// VoteCount returns the current number of registered votes.
func (r *Round) VoteCount() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.votes)
}

// IsClosed returns true if the round has been closed.
func (r *Round) IsClosed() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.closed
}
