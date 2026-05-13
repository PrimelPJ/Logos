// Package consensus implements Logos's modified Raft protocol.
// Key deviations from vanilla Raft:
//
//  1. Log entries carry semantic embeddings alongside their raw payloads.
//     During log replication, followers check semantic equivalence (not just
//     hash equality) before accepting entries from the leader.
//
//  2. Leader election includes a "semantic compatibility" check: a candidate
//     must demonstrate that its log's semantic fingerprint is within θ of the
//     majority before receiving votes. This prevents a semantically diverged
//     node from becoming leader even if it has a higher term.
//
//  3. The commit index can advance on "fuzzy majority" — a quorum of nodes
//     that all hold semantically equivalent (but possibly byte-different) state.
//
//  4. A "reconciliation" phase can run after commit to align byte-level state
//     across nodes when ExactAgreement is false.
package consensus

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/PrimelPJ/logos/pkg/quorum"
	"github.com/PrimelPJ/logos/pkg/semantic"
	"go.uber.org/zap"
)

// Role represents the node's current Raft role.
type Role int32

const (
	RoleFollower  Role = iota
	RoleCandidate
	RoleLeader
)

func (r Role) String() string {
	switch r {
	case RoleFollower:
		return "follower"
	case RoleCandidate:
		return "candidate"
	case RoleLeader:
		return "leader"
	}
	return "unknown"
}

// Entry is a Raft log entry enriched with a semantic embedding.
type Entry struct {
	Index     uint64
	Term      uint64
	Payload   []byte
	Embedding semantic.EmbedResult
	CommittedAt time.Time
	Reconciled  bool // true iff byte-level agreement was confirmed post-commit
}

// Log is Logos's append-only, semantically-indexed consensus log.
type Log struct {
	mu      sync.RWMutex
	entries []Entry
	index   *semantic.Index
}

// NewLog creates an empty log with a semantic index.
func NewLog() *Log {
	return &Log{
		index: semantic.NewIndex(16),
	}
}

// Append adds an entry to the log. Thread-safe.
func (l *Log) Append(e Entry) {
	l.mu.Lock()
	defer l.mu.Unlock()
	e.Index = uint64(len(l.entries) + 1)
	l.entries = append(l.entries, e)
	l.index.Insert(fmt.Sprintf("%d", e.Index), e.Embedding)
}

// Get returns the entry at the given 1-based index.
func (l *Log) Get(index uint64) (Entry, bool) {
	l.mu.RLock()
	defer l.mu.RUnlock()
	if index == 0 || int(index) > len(l.entries) {
		return Entry{}, false
	}
	return l.entries[index-1], true
}

// LastEntry returns the most recent entry in the log.
func (l *Log) LastEntry() (Entry, bool) {
	l.mu.RLock()
	defer l.mu.RUnlock()
	if len(l.entries) == 0 {
		return Entry{}, false
	}
	return l.entries[len(l.entries)-1], true
}

// Len returns the number of log entries.
func (l *Log) Len() int {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return len(l.entries)
}

// SemanticSearch finds log entries semantically similar to the given payload.
// This enables "fuzzy replay" during reconciliation.
func (l *Log) SemanticSearch(embed semantic.EmbedResult, k int, threshold float64) []semantic.SimilarityResult {
	return l.index.Query(embed, k, threshold)
}

// TruncateFrom removes all entries from the given index onwards (Raft log correction).
func (l *Log) TruncateFrom(index uint64) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if int(index) <= len(l.entries) {
		l.entries = l.entries[:index-1]
		// Rebuild index after truncation
		l.index = semantic.NewIndex(16)
		for _, e := range l.entries {
			l.index.Insert(fmt.Sprintf("%d", e.Index), e.Embedding)
		}
	}
}

// SemanticFingerprint returns the centroid vector of the last `n` log entries.
// Used in leader election to prove semantic compatibility.
func (l *Log) SemanticFingerprint(lastN int) (semantic.Vector, bool) {
	l.mu.RLock()
	defer l.mu.RUnlock()
	if len(l.entries) == 0 {
		return semantic.Vector{}, false
	}
	start := len(l.entries) - lastN
	if start < 0 {
		start = 0
	}
	vecs := make([]semantic.Vector, 0, lastN)
	for _, e := range l.entries[start:] {
		vecs = append(vecs, e.Embedding.Vec)
	}
	return semantic.Centroid(vecs), true
}

// ─── State Machine ─────────────────────────────────────────────────────────────

// StateMachine is the interface that consumers implement to apply committed entries.
// Logos calls Apply for each committed entry, passing both the raw payload and
// its semantic embedding so the application can use semantic routing if desired.
type StateMachine interface {
	Apply(entry Entry) error
	Snapshot() ([]byte, error)
	Restore(snapshot []byte) error
}

// ─── Node (Raft Participant) ────────────────────────────────────────────────────

// NodeConfig holds the configuration for a Logos consensus node.
type NodeConfig struct {
	ID                  string
	ClusterSize         int
	SimilarityThreshold float64
	DriftWarnAngle      float64 // radians, default ~0.3 (~17°)
	ElectionTimeout     time.Duration
	HeartbeatInterval   time.Duration
	ClusterKey          []byte
}

// DefaultConfig returns sensible defaults for a 3-node cluster.
func DefaultConfig(id string, clusterKey []byte) NodeConfig {
	return NodeConfig{
		ID:                  id,
		ClusterSize:         3,
		SimilarityThreshold: semantic.DefaultSimilarityThreshold,
		DriftWarnAngle:      0.30, // ~17 degrees
		ElectionTimeout:     150 * time.Millisecond,
		HeartbeatInterval:   50 * time.Millisecond,
		ClusterKey:          clusterKey,
	}
}

// Node is a Logos consensus participant.
type Node struct {
	cfg      NodeConfig
	log      *Log
	embedder *semantic.Embedder
	sm       StateMachine
	logger   *zap.Logger

	// Raft state
	currentTerm  uint64
	votedFor     string
	role         atomic.Int32
	commitIndex  uint64
	lastApplied  uint64
	leadID       string

	// Quorum round manager
	activeRound *quorum.Round
	roundMu     sync.Mutex
	roundSeq    uint64

	// Channels
	applyC  chan Entry
	stopC   chan struct{}
	elcTimer *time.Timer
	mu      sync.RWMutex
}

// NewNode creates a new Logos consensus node.
func NewNode(cfg NodeConfig, sm StateMachine, logger *zap.Logger) *Node {
	n := &Node{
		cfg:      cfg,
		log:      NewLog(),
		embedder: semantic.NewEmbedder(cfg.ClusterKey),
		sm:       sm,
		logger:   logger,
		applyC:   make(chan Entry, 256),
		stopC:    make(chan struct{}),
	}
	n.role.Store(int32(RoleFollower))
	return n
}

// Start begins the node's background goroutines.
func (n *Node) Start(ctx context.Context) {
	go n.applyLoop(ctx)
	go n.electionLoop(ctx)
	n.logger.Info("logos node started",
		zap.String("id", n.cfg.ID),
		zap.String("role", RoleFollower.String()),
	)
}

// Propose submits a payload for consensus. Only valid on the leader.
// Returns the log index assigned to this proposal, or an error.
func (n *Node) Propose(ctx context.Context, payload []byte) (uint64, error) {
	n.mu.Lock()
	defer n.mu.Unlock()

	if Role(n.role.Load()) != RoleLeader {
		return 0, fmt.Errorf("not the leader: current role is %s", Role(n.role.Load()))
	}

	embed := n.embedder.Embed(payload)
	entry := Entry{
		Term:      n.currentTerm,
		Payload:   payload,
		Embedding: embed,
	}
	n.log.Append(entry)

	lastEntry, _ := n.log.LastEntry()
	n.logger.Info("proposed entry",
		zap.Uint64("index", lastEntry.Index),
		zap.Uint64("term", lastEntry.Term),
		zap.Float64("embed_magnitude", lastEntry.Embedding.Magnitude),
	)
	return lastEntry.Index, nil
}

// SemanticLookup finds log entries semantically similar to the given payload.
// Useful for idempotency checks and fuzzy deduplication before proposing.
func (n *Node) SemanticLookup(payload []byte, k int) []semantic.SimilarityResult {
	embed := n.embedder.Embed(payload)
	return n.log.SemanticSearch(embed, k, n.cfg.SimilarityThreshold)
}

// Embed converts a payload to its embedding under this node's cluster key.
func (n *Node) Embed(payload []byte) semantic.EmbedResult {
	return n.embedder.Embed(payload)
}

// applyLoop reads from applyC and calls the state machine.
func (n *Node) applyLoop(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case <-n.stopC:
			return
		case entry := <-n.applyC:
			if err := n.sm.Apply(entry); err != nil {
				n.logger.Error("state machine apply failed",
					zap.Uint64("index", entry.Index),
					zap.Error(err),
				)
			}
		}
	}
}

// electionLoop manages election timeouts (simplified — full RPC layer in transport/).
func (n *Node) electionLoop(ctx context.Context) {
	ticker := time.NewTicker(n.cfg.HeartbeatInterval)
	defer ticker.Stop()

	elcDeadline := time.Now().Add(n.cfg.ElectionTimeout)

	for {
		select {
		case <-ctx.Done():
			return
		case <-n.stopC:
			return
		case t := <-ticker.C:
			role := Role(n.role.Load())
			if role == RoleFollower && t.After(elcDeadline) {
				n.becomeCandidate()
			}
			if role == RoleLeader {
				n.sendHeartbeat()
				elcDeadline = time.Now().Add(n.cfg.ElectionTimeout)
			}
		}
	}
}

func (n *Node) becomeCandidate() {
	n.mu.Lock()
	n.currentTerm++
	n.votedFor = n.cfg.ID
	n.role.Store(int32(RoleCandidate))
	n.mu.Unlock()

	n.logger.Info("became candidate",
		zap.String("id", n.cfg.ID),
		zap.Uint64("term", n.currentTerm),
	)
	// In a full implementation, RequestVote RPCs are sent here via transport/.
}

func (n *Node) becomeLeader() {
	n.role.Store(int32(RoleLeader))
	n.leadID = n.cfg.ID
	n.logger.Info("became leader",
		zap.String("id", n.cfg.ID),
		zap.Uint64("term", n.currentTerm),
	)
}

func (n *Node) sendHeartbeat() {
	// In a full implementation, AppendEntries(empty) RPCs are sent here via transport/.
	n.logger.Debug("heartbeat", zap.String("id", n.cfg.ID), zap.Uint64("term", n.currentTerm))
}

// Stop shuts down the node.
func (n *Node) Stop() {
	close(n.stopC)
	n.logger.Info("node stopped", zap.String("id", n.cfg.ID))
}

// Status returns a snapshot of the node's current state.
func (n *Node) Status() NodeStatus {
	n.mu.RLock()
	defer n.mu.RUnlock()
	return NodeStatus{
		ID:          n.cfg.ID,
		Role:        Role(n.role.Load()),
		Term:        n.currentTerm,
		CommitIndex: n.commitIndex,
		LastApplied: n.lastApplied,
		LogLen:      uint64(n.log.Len()),
		LeaderID:    n.leadID,
	}
}

// NodeStatus is a snapshot of a node's current state.
type NodeStatus struct {
	ID          string
	Role        Role
	Term        uint64
	CommitIndex uint64
	LastApplied uint64
	LogLen      uint64
	LeaderID    string
}
