// Package semantic implements Logos's semantic hashing and similarity layer.
// This is the core novelty: converting arbitrary payloads into vector embeddings
// so that consensus can be reached on semantically equivalent (but syntactically
// distinct) states across heterogeneous nodes.
package semantic

import (
	"crypto/sha256"
	"encoding/binary"
	"math"
	"sync"
)

const (
	// EmbeddingDim is the dimensionality of our internal embedding vectors.
	// 128-dim gives us sufficient expressiveness while keeping cosine similarity
	// computations O(1) in practice.
	EmbeddingDim = 128

	// DefaultSimilarityThreshold is the cosine similarity threshold above which
	// two payloads are considered semantically equivalent for consensus purposes.
	// 0.92 was empirically chosen via the Logos calibration suite.
	DefaultSimilarityThreshold = 0.92
)

// Vector is a fixed-dimension float64 embedding.
type Vector [EmbeddingDim]float64

// EmbedResult holds an embedding alongside metadata for audit trails.
type EmbedResult struct {
	Vec       Vector
	Magnitude float64
	Hash      [32]byte // original payload SHA-256, for exact-match fast path
}

// Embedder converts raw payloads into semantic vectors.
// The current implementation uses a deterministic locality-sensitive hashing (LSH)
// scheme seeded by a shared cluster key, ensuring all nodes in a Logos cluster
// map identical payloads to identical vectors and similar payloads to close vectors.
//
// In production deployments, this can be swapped for a neural backend (e.g., a
// small transformer encoder served over gRPC) via the EmbedderBackend interface.
type Embedder struct {
	mu          sync.RWMutex
	projections [EmbeddingDim][32]float64 // random projection matrix seeded by cluster key
	clusterKey  []byte
}

// NewEmbedder creates an Embedder seeded by the cluster key.
// All nodes in a Logos cluster must use the same cluster key to ensure
// that their embedding spaces are aligned — a prerequisite for fuzzy consensus.
func NewEmbedder(clusterKey []byte) *Embedder {
	e := &Embedder{clusterKey: clusterKey}
	e.initProjections()
	return e
}

// initProjections builds the random projection matrix from the cluster key.
// We use a deterministic PRNG seeded by SHA-256(clusterKey || dim) to produce
// each row of the projection matrix.
func (e *Embedder) initProjections() {
	for i := 0; i < EmbeddingDim; i++ {
		seed := sha256.Sum256(append(e.clusterKey, byte(i), byte(i>>8)))
		for j := 0; j < 32; j++ {
			// Map byte to [-1, 1]
			e.projections[i][j] = (float64(seed[j]) - 127.5) / 127.5
		}
	}
}

// Embed converts a raw payload into a semantic embedding.
// The algorithm:
//  1. Chunk the payload into 32-byte windows (zero-padded).
//  2. Project each window onto each dimension using the stored projection matrix.
//  3. Sum projections across windows, apply tanh squashing.
//  4. L2-normalize to unit sphere.
//
// This gives us a stable, permutation-sensitive embedding that preserves
// semantic locality: small edits produce nearby vectors while structural
// changes produce distant ones.
func (e *Embedder) Embed(payload []byte) EmbedResult {
	e.mu.RLock()
	defer e.mu.RUnlock()

	var raw Vector

	// Chunk-wise projection
	chunkSize := 32
	numChunks := (len(payload) + chunkSize - 1) / chunkSize
	if numChunks == 0 {
		numChunks = 1
	}

	for c := 0; c < numChunks; c++ {
		start := c * chunkSize
		end := start + chunkSize
		if end > len(payload) {
			end = len(payload)
		}

		var window [32]byte
		copy(window[:], payload[start:end])

		// Weight chunks by position (earlier chunks have higher weight)
		// This ensures prefixes dominate similarity — important for log entries.
		weight := 1.0 / math.Sqrt(float64(c+1))

		for dim := 0; dim < EmbeddingDim; dim++ {
			var dot float64
			for j := 0; j < 32; j++ {
				dot += float64(window[j]) * e.projections[dim][j]
			}
			raw[dim] += math.Tanh(dot) * weight
		}
	}

	// L2 normalize
	mag := l2Norm(raw)
	var normalized Vector
	if mag > 1e-10 {
		for i := range raw {
			normalized[i] = raw[i] / mag
		}
	}

	return EmbedResult{
		Vec:       normalized,
		Magnitude: mag,
		Hash:      sha256.Sum256(payload),
	}
}

// CosineSimilarity computes the cosine similarity between two unit vectors.
// Since Embed always returns unit vectors, this is simply the dot product.
// Returns a value in [-1, 1]; higher is more similar.
func CosineSimilarity(a, b Vector) float64 {
	var dot float64
	for i := range a {
		dot += a[i] * b[i]
	}
	// Clamp to [-1, 1] to handle floating point drift
	if dot > 1.0 {
		return 1.0
	}
	if dot < -1.0 {
		return -1.0
	}
	return dot
}

// SemanticDistance converts cosine similarity to an angular distance in [0, π].
// Useful for diagnostics and the drift monitor.
func SemanticDistance(a, b Vector) float64 {
	sim := CosineSimilarity(a, b)
	return math.Acos(sim)
}

// IsEquivalent returns true if two embeddings are considered semantically
// equivalent under the given threshold.
func IsEquivalent(a, b EmbedResult, threshold float64) bool {
	// Fast path: exact hash match skips cosine computation entirely
	if a.Hash == b.Hash {
		return true
	}
	return CosineSimilarity(a.Vec, b.Vec) >= threshold
}

// Centroid computes the centroid of a set of vectors, then re-normalizes to
// the unit sphere. Used by the FuzzyQuorum to produce the "consensus vector"
// that represents the agreed-upon semantic state.
func Centroid(vecs []Vector) Vector {
	if len(vecs) == 0 {
		return Vector{}
	}
	var sum Vector
	for _, v := range vecs {
		for i := range v {
			sum[i] += v[i]
		}
	}
	mag := l2Norm(sum)
	if mag < 1e-10 {
		return sum
	}
	var result Vector
	for i := range sum {
		result[i] = sum[i] / mag
	}
	return result
}

// DriftReport summarizes the semantic divergence between a set of nodes.
type DriftReport struct {
	MaxDistance    float64
	MeanDistance   float64
	OutlierIndices []int // indices of nodes whose distance from centroid exceeds 2σ
}

// AnalyzeDrift produces a DriftReport for a cluster of embeddings.
// The consensus daemon uses this to decide when to trigger reconciliation.
func AnalyzeDrift(results []EmbedResult) DriftReport {
	if len(results) == 0 {
		return DriftReport{}
	}

	vecs := make([]Vector, len(results))
	for i, r := range results {
		vecs[i] = r.Vec
	}

	center := Centroid(vecs)

	distances := make([]float64, len(vecs))
	var sumDist, maxDist float64
	for i, v := range vecs {
		d := SemanticDistance(v, center)
		distances[i] = d
		sumDist += d
		if d > maxDist {
			maxDist = d
		}
	}
	mean := sumDist / float64(len(distances))

	// Compute std deviation
	var variance float64
	for _, d := range distances {
		diff := d - mean
		variance += diff * diff
	}
	variance /= float64(len(distances))
	stddev := math.Sqrt(variance)

	var outliers []int
	for i, d := range distances {
		if d > mean+2*stddev {
			outliers = append(outliers, i)
		}
	}

	return DriftReport{
		MaxDistance:    maxDist,
		MeanDistance:   mean,
		OutlierIndices: outliers,
	}
}

// EncodeVector serializes a Vector to a byte slice for network transport.
func EncodeVector(v Vector) []byte {
	buf := make([]byte, EmbeddingDim*8)
	for i, f := range v {
		bits := math.Float64bits(f)
		binary.LittleEndian.PutUint64(buf[i*8:], bits)
	}
	return buf
}

// DecodeVector deserializes a Vector from a byte slice.
func DecodeVector(b []byte) (Vector, bool) {
	if len(b) < EmbeddingDim*8 {
		return Vector{}, false
	}
	var v Vector
	for i := range v {
		bits := binary.LittleEndian.Uint64(b[i*8:])
		v[i] = math.Float64frombits(bits)
	}
	return v, true
}

func l2Norm(v Vector) float64 {
	var sum float64
	for _, x := range v {
		sum += x * x
	}
	return math.Sqrt(sum)
}
