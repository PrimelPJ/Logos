package consensus

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/PrimelPJ/logos/pkg/semantic"
	"go.uber.org/zap"
)

// ReconcileStrategy determines how byte-level state is resolved when fuzzy
// consensus produced ExactAgreement=false.
type ReconcileStrategy int

const (
	// StrategyLeaderAuthority: adopt the leader's byte-level state verbatim.
	// Fast, low overhead, suitable when nodes are known to be correct but
	// using different encodings.
	StrategyLeaderAuthority ReconcileStrategy = iota

	// StrategySemanticMerge: produce a canonical representation by applying
	// a merge function over all semantically equivalent variants.
	// Slower but preserves information from all nodes.
	StrategySemanticMerge

	// StrategyVectorMajority: adopt the payload whose embedding is closest
	// to the consensus centroid (most representative variant wins).
	StrategyVectorMajority
)

// ReconcileRequest is sent to the Reconciler when a fuzzy-committed entry
// needs byte-level alignment.
type ReconcileRequest struct {
	LogIndex        uint64
	Term            uint64
	ConsensusVector semantic.Vector
	Variants        []Variant // one per node that submitted a vote
}

// Variant is a single node's version of an entry's payload.
type Variant struct {
	NodeID    string
	Payload   []byte
	Embedding semantic.EmbedResult
}

// ReconcileResult is the output of a reconciliation pass.
type ReconcileResult struct {
	LogIndex        uint64
	Strategy        ReconcileStrategy
	CanonicalPayload []byte
	CanonicalEmbed   semantic.EmbedResult
	VariantCount     int
	MaxDriftBefore   float64
	DriftAfter       float64 // should be 0.0 (exact match)
	Duration         time.Duration
	ReconciledAt     time.Time
}

// MergeFunc is a user-supplied function that produces a canonical payload from
// multiple semantically equivalent variants. Used by StrategySemanticMerge.
// Applications register merge functions per payload type.
type MergeFunc func(variants []Variant, consensusVec semantic.Vector) ([]byte, error)

// Reconciler handles byte-level state alignment after fuzzy commits.
type Reconciler struct {
	mu          sync.RWMutex
	strategy    ReconcileStrategy
	mergeFuncs  map[string]MergeFunc // keyed by payload type tag (first 4 bytes)
	embedder    *semantic.Embedder
	logger      *zap.Logger
	history     []ReconcileResult
	maxHistory  int
}

// NewReconciler creates a Reconciler with the given default strategy.
func NewReconciler(strategy ReconcileStrategy, embedder *semantic.Embedder, logger *zap.Logger) *Reconciler {
	return &Reconciler{
		strategy:   strategy,
		mergeFuncs: make(map[string]MergeFunc),
		embedder:   embedder,
		logger:     logger,
		maxHistory: 500,
	}
}

// RegisterMergeFunc registers a custom merge function for payloads starting
// with the given 4-byte type tag. Only used with StrategySemanticMerge.
func (r *Reconciler) RegisterMergeFunc(typeTag string, fn MergeFunc) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.mergeFuncs[typeTag] = fn
}

// Reconcile processes a ReconcileRequest and returns the canonical payload.
func (r *Reconciler) Reconcile(ctx context.Context, req ReconcileRequest) (*ReconcileResult, error) {
	start := time.Now()

	if len(req.Variants) == 0 {
		return nil, fmt.Errorf("reconcile request for index %d has no variants", req.LogIndex)
	}

	// Compute max drift before reconciliation
	embeds := make([]semantic.EmbedResult, len(req.Variants))
	for i, v := range req.Variants {
		embeds[i] = v.Embedding
	}
	driftBefore := semantic.AnalyzeDrift(embeds).MaxDistance

	var canonical []byte
	var strategyUsed ReconcileStrategy
	var err error

	switch r.strategy {
	case StrategyLeaderAuthority:
		canonical, strategyUsed, err = r.leaderAuthority(req)
	case StrategySemanticMerge:
		canonical, strategyUsed, err = r.semanticMerge(ctx, req)
	case StrategyVectorMajority:
		canonical, strategyUsed, err = r.vectorMajority(req)
	default:
		return nil, fmt.Errorf("unknown reconcile strategy: %d", r.strategy)
	}

	if err != nil {
		return nil, fmt.Errorf("reconcile[index=%d strategy=%d]: %w", req.LogIndex, r.strategy, err)
	}

	canonicalEmbed := r.embedder.Embed(canonical)

	result := &ReconcileResult{
		LogIndex:         req.LogIndex,
		Strategy:         strategyUsed,
		CanonicalPayload: canonical,
		CanonicalEmbed:   canonicalEmbed,
		VariantCount:     len(req.Variants),
		MaxDriftBefore:   driftBefore,
		DriftAfter:       0.0, // exact after reconciliation
		Duration:         time.Since(start),
		ReconciledAt:     time.Now(),
	}

	r.mu.Lock()
	r.history = append(r.history, *result)
	if len(r.history) > r.maxHistory {
		r.history = r.history[1:]
	}
	r.mu.Unlock()

	r.logger.Info("reconciled log entry",
		zap.Uint64("index", req.LogIndex),
		zap.String("strategy", fmt.Sprintf("%d", strategyUsed)),
		zap.Int("variants", len(req.Variants)),
		zap.Float64("drift_before", driftBefore),
		zap.Duration("duration", result.Duration),
	)

	return result, nil
}

func (r *Reconciler) leaderAuthority(req ReconcileRequest) ([]byte, ReconcileStrategy, error) {
	// Find the variant marked as leader (NodeID == "leader" or first in list if not tagged)
	for _, v := range req.Variants {
		if v.NodeID == "leader" {
			return v.Payload, StrategyLeaderAuthority, nil
		}
	}
	// Fall back to first variant
	return req.Variants[0].Payload, StrategyLeaderAuthority, nil
}

func (r *Reconciler) semanticMerge(ctx context.Context, req ReconcileRequest) ([]byte, ReconcileStrategy, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if len(req.Variants) == 0 {
		return nil, StrategySemanticMerge, fmt.Errorf("no variants")
	}

	// Determine type tag from first variant
	typeTag := ""
	if len(req.Variants[0].Payload) >= 4 {
		typeTag = string(req.Variants[0].Payload[:4])
	}

	if fn, ok := r.mergeFuncs[typeTag]; ok {
		result, err := fn(req.Variants, req.ConsensusVector)
		return result, StrategySemanticMerge, err
	}

	// No merge function: fall through to vector majority
	result, _, err := r.vectorMajority(req)
	return result, StrategySemanticMerge, err
}

func (r *Reconciler) vectorMajority(req ReconcileRequest) ([]byte, ReconcileStrategy, error) {
	if len(req.Variants) == 0 {
		return nil, StrategyVectorMajority, fmt.Errorf("no variants")
	}

	// Pick the variant whose embedding is closest to the consensus centroid
	type scored struct {
		variant Variant
		dist    float64
	}

	scored_variants := make([]scored, len(req.Variants))
	for i, v := range req.Variants {
		dist := semantic.SemanticDistance(v.Embedding.Vec, req.ConsensusVector)
		scored_variants[i] = scored{variant: v, dist: dist}
	}

	sort.Slice(scored_variants, func(i, j int) bool {
		return scored_variants[i].dist < scored_variants[j].dist
	})

	return scored_variants[0].variant.Payload, StrategyVectorMajority, nil
}

// History returns recent reconciliation results.
func (r *Reconciler) History(n int) []ReconcileResult {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if n > len(r.history) {
		n = len(r.history)
	}
	result := make([]ReconcileResult, n)
	copy(result, r.history[len(r.history)-n:])
	return result
}
