// Package node provides the top-level Logos node — the public API surface
// that application code interacts with. It assembles the consensus engine,
// semantic layer, quorum manager, drift monitor, and reconciler into a unified
// coherent system.
package node

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/PrimelPJ/logos/pkg/consensus"
	"github.com/PrimelPJ/logos/pkg/quorum"
	"github.com/PrimelPJ/logos/pkg/semantic"
	"go.uber.org/zap"
)

// Config is the top-level Logos node configuration.
type Config struct {
	NodeID             string
	ClusterKey         []byte
	ClusterPeers       []string // peer node addresses
	ClusterSize        int
	ListenAddr         string
	DataDir            string
	SimilarityThreshold float64
	DriftWarnAngle     float64
	DriftCriticalAngle float64
	ReconcileStrategy  consensus.ReconcileStrategy
	ElectionTimeout    time.Duration
	HeartbeatInterval  time.Duration
}

// DefaultConfig produces a sensible default configuration.
func DefaultConfig(nodeID string, clusterKey []byte) Config {
	return Config{
		NodeID:              nodeID,
		ClusterKey:          clusterKey,
		ClusterSize:         3,
		SimilarityThreshold: semantic.DefaultSimilarityThreshold,
		DriftWarnAngle:      0.30,
		DriftCriticalAngle:  0.60,
		ReconcileStrategy:   consensus.StrategyVectorMajority,
		ElectionTimeout:     150 * time.Millisecond,
		HeartbeatInterval:   50 * time.Millisecond,
		ListenAddr:          ":7700",
		DataDir:             "./data",
	}
}

// LogosNode is the top-level Logos node.
type LogosNode struct {
	cfg          Config
	raftNode     *consensus.Node
	driftMonitor *consensus.DriftMonitor
	reconciler   *consensus.Reconciler
	embedder     *semantic.Embedder
	logger       *zap.Logger

	mu      sync.RWMutex
	started bool
	stopC   chan struct{}

	// Pending quorum rounds (indexed by round ID)
	rounds   map[uint64]*quorum.Round
	roundsMu sync.Mutex
	roundSeq uint64
}

// New creates a new LogosNode.
func New(cfg Config, sm consensus.StateMachine, logger *zap.Logger) *LogosNode {
	embedder := semantic.NewEmbedder(cfg.ClusterKey)

	raftCfg := consensus.NodeConfig{
		ID:                  cfg.NodeID,
		ClusterSize:         cfg.ClusterSize,
		SimilarityThreshold: cfg.SimilarityThreshold,
		DriftWarnAngle:      cfg.DriftWarnAngle,
		ElectionTimeout:     cfg.ElectionTimeout,
		HeartbeatInterval:   cfg.HeartbeatInterval,
		ClusterKey:          cfg.ClusterKey,
	}

	driftCfg := consensus.DriftMonitorConfig{
		SampleInterval:   2 * time.Second,
		WarnAngle:        cfg.DriftWarnAngle,
		CriticalAngle:    cfg.DriftCriticalAngle,
		FingerprintDepth: 50,
	}

	return &LogosNode{
		cfg:          cfg,
		raftNode:     consensus.NewNode(raftCfg, sm, logger),
		driftMonitor: consensus.NewDriftMonitor(driftCfg, logger),
		reconciler:   consensus.NewReconciler(cfg.ReconcileStrategy, embedder, logger),
		embedder:     embedder,
		logger:       logger,
		stopC:        make(chan struct{}),
		rounds:       make(map[uint64]*quorum.Round),
	}
}

// Start launches all background services.
func (n *LogosNode) Start(ctx context.Context) error {
	n.mu.Lock()
	defer n.mu.Unlock()

	if n.started {
		return fmt.Errorf("node %s already started", n.cfg.NodeID)
	}

	n.raftNode.Start(ctx)
	go n.driftMonitor.Run(ctx)
	go n.reconcileLoop(ctx)

	n.started = true
	n.logger.Info("logos node started",
		zap.String("node_id", n.cfg.NodeID),
		zap.Int("cluster_size", n.cfg.ClusterSize),
		zap.Float64("similarity_threshold", n.cfg.SimilarityThreshold),
		zap.String("listen_addr", n.cfg.ListenAddr),
	)
	return nil
}

// Stop gracefully shuts down the node.
func (n *LogosNode) Stop() {
	n.raftNode.Stop()
	close(n.stopC)
	n.logger.Info("logos node stopped", zap.String("node_id", n.cfg.NodeID))
}

// Propose submits a payload for consensus. Semantically deduplicates before
// submitting to Raft: if a sufficiently similar payload was recently committed,
// returns the existing log index instead of creating a duplicate entry.
func (n *LogosNode) Propose(ctx context.Context, payload []byte) (uint64, bool, error) {
	// Semantic deduplication
	similar := n.raftNode.SemanticLookup(payload, 3)
	if len(similar) > 0 && similar[0].Similarity >= n.cfg.SimilarityThreshold {
		n.logger.Debug("semantic dedup hit",
			zap.String("matched_id", similar[0].ID),
			zap.Float64("similarity", similar[0].Similarity),
		)
		// Parse log index from result ID and return it
		var existingIndex uint64
		fmt.Sscanf(similar[0].ID, "%d", &existingIndex)
		return existingIndex, true, nil
	}

	index, err := n.raftNode.Propose(ctx, payload)
	return index, false, err
}

// NewQuorumRound creates a new fuzzy quorum round for manual vote aggregation.
// This is the lower-level API — most users should use Propose() instead.
func (n *LogosNode) NewQuorumRound(term uint64) uint64 {
	n.roundsMu.Lock()
	defer n.roundsMu.Unlock()

	n.roundSeq++
	roundID := n.roundSeq

	r := quorum.NewRound(
		roundID,
		term,
		n.cfg.ClusterSize,
		n.cfg.SimilarityThreshold,
		n.cfg.DriftWarnAngle,
	)
	n.rounds[roundID] = r
	return roundID
}

// CastVote adds a vote to the specified quorum round.
func (n *LogosNode) CastVote(roundID uint64, v quorum.Vote) (*quorum.QuorumResult, error) {
	n.roundsMu.Lock()
	r, ok := n.rounds[roundID]
	n.roundsMu.Unlock()

	if !ok {
		return nil, fmt.Errorf("unknown round ID: %d", roundID)
	}

	if err := r.AddVote(v); err != nil {
		return nil, err
	}

	return r.TryClose(), nil
}

// CloseRound forces a round closed (e.g., on timeout) and returns its result.
func (n *LogosNode) CloseRound(roundID uint64) (*quorum.QuorumResult, error) {
	n.roundsMu.Lock()
	r, ok := n.rounds[roundID]
	n.roundsMu.Unlock()

	if !ok {
		return nil, fmt.Errorf("unknown round ID: %d", roundID)
	}
	return r.ForceClose(), nil
}

// Embed converts a payload to its semantic embedding under this cluster's key.
func (n *LogosNode) Embed(payload []byte) semantic.EmbedResult {
	return n.embedder.Embed(payload)
}

// SemanticSearch finds log entries semantically similar to the given payload.
func (n *LogosNode) SemanticSearch(payload []byte, k int) []semantic.SimilarityResult {
	return n.raftNode.SemanticLookup(payload, k)
}

// DriftSummary returns a human-readable summary of current cluster drift.
func (n *LogosNode) DriftSummary() string {
	return n.driftMonitor.Summary()
}

// Status returns the node's current status.
func (n *LogosNode) Status() consensus.NodeStatus {
	return n.raftNode.Status()
}

// RegisterMergeFunc registers a custom payload merge function for reconciliation.
// See Reconciler.RegisterMergeFunc for details.
func (n *LogosNode) RegisterMergeFunc(typeTag string, fn consensus.MergeFunc) {
	n.reconciler.RegisterMergeFunc(typeTag, fn)
}

// reconcileLoop handles incoming reconciliation requests from the drift monitor.
func (n *LogosNode) reconcileLoop(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case <-n.stopC:
			return
		case outliers := <-n.driftMonitor.ReconcileRequests():
			n.logger.Warn("reconcile request received",
				zap.Strings("outlier_nodes", outliers),
			)
			// In a full implementation, this would trigger snapshot transfer
			// to the outlier nodes and re-apply committed entries.
		}
	}
}
