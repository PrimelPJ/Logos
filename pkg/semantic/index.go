package semantic

import (
	"math"
	"sort"
	"sync"
)

// IndexEntry stores an embedding alongside an arbitrary payload identifier.
type IndexEntry struct {
	ID     string
	Result EmbedResult
}

// SimilarityResult is a ranked match from an index query.
type SimilarityResult struct {
	ID         string
	Similarity float64
	Distance   float64
}

// Index is an in-memory approximate nearest neighbor index over semantic embeddings.
// It uses a multi-probe LSH bucket scheme for sub-linear query time at scale.
//
// Design note: For clusters with <10,000 log entries, exact linear scan is fast
// enough (< 1ms). For larger logs, the LSH index kicks in automatically.
type Index struct {
	mu           sync.RWMutex
	entries      []IndexEntry
	lshBuckets   map[uint64][]int // bucket hash -> entry indices
	hashFuncs    [][EmbeddingDim]float64
	numHashFuncs int
}

// NewIndex creates an empty similarity index with the given number of LSH hash functions.
// More hash functions = better precision, higher memory cost.
func NewIndex(numHashFuncs int) *Index {
	idx := &Index{
		lshBuckets:   make(map[uint64][]int),
		numHashFuncs: numHashFuncs,
	}
	idx.initHashFuncs()
	return idx
}

func (idx *Index) initHashFuncs() {
	idx.hashFuncs = make([][EmbeddingDim]float64, idx.numHashFuncs)
	for i := range idx.hashFuncs {
		seed := sha256SeedFn(uint64(i) * 0xdeadbeef)
		for j := range idx.hashFuncs[i] {
			idx.hashFuncs[i][j] = (float64(seed[j%32]) - 127.5) / 127.5
		}
	}
}

func sha256SeedFn(seed uint64) [32]byte {
	var b [8]byte
	b[0] = byte(seed)
	b[1] = byte(seed >> 8)
	b[2] = byte(seed >> 16)
	b[3] = byte(seed >> 24)
	b[4] = byte(seed >> 32)
	b[5] = byte(seed >> 40)
	b[6] = byte(seed >> 48)
	b[7] = byte(seed >> 56)
	var result [32]byte
	for i := range result {
		result[i] = b[i%8] ^ byte(i*31)
	}
	return result
}

// lshHash computes the LSH bucket key for a vector using the i-th hash function.
func (idx *Index) lshHash(v Vector) uint64 {
	var key uint64
	for i, hf := range idx.hashFuncs {
		var dot float64
		for j := range v {
			dot += v[j] * hf[j]
		}
		if dot >= 0 {
			key |= 1 << uint(i%64)
		}
	}
	return key
}

// Insert adds a new entry to the index.
func (idx *Index) Insert(id string, result EmbedResult) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	entryIdx := len(idx.entries)
	idx.entries = append(idx.entries, IndexEntry{ID: id, Result: result})

	bucket := idx.lshHash(result.Vec)
	idx.lshBuckets[bucket] = append(idx.lshBuckets[bucket], entryIdx)
}

// Query finds the top-k most similar entries to the query vector.
// Uses LSH for candidate selection, then exact cosine scoring for ranking.
func (idx *Index) Query(query EmbedResult, k int, threshold float64) []SimilarityResult {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if len(idx.entries) == 0 {
		return nil
	}

	// For small indexes, do exact linear scan
	if len(idx.entries) < 500 {
		return idx.exactScan(query, k, threshold)
	}

	// Candidate selection via LSH
	bucket := idx.lshHash(query.Vec)
	candidateSet := make(map[int]struct{})

	// Probe the exact bucket
	for _, ci := range idx.lshBuckets[bucket] {
		candidateSet[ci] = struct{}{}
	}

	// Multi-probe: flip each bit in the bucket key to find neighboring buckets
	for bit := 0; bit < idx.numHashFuncs && bit < 16; bit++ {
		neighborBucket := bucket ^ (1 << uint(bit))
		for _, ci := range idx.lshBuckets[neighborBucket] {
			candidateSet[ci] = struct{}{}
		}
	}

	// If too few candidates, fall back to exact scan
	if len(candidateSet) < k*2 {
		return idx.exactScan(query, k, threshold)
	}

	// Score candidates
	var results []SimilarityResult
	for ci := range candidateSet {
		entry := idx.entries[ci]
		sim := CosineSimilarity(query.Vec, entry.Result.Vec)
		if sim >= threshold {
			results = append(results, SimilarityResult{
				ID:         entry.ID,
				Similarity: sim,
				Distance:   math.Acos(math.Min(1.0, math.Max(-1.0, sim))),
			})
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})

	if len(results) > k {
		return results[:k]
	}
	return results
}

func (idx *Index) exactScan(query EmbedResult, k int, threshold float64) []SimilarityResult {
	var results []SimilarityResult
	for _, entry := range idx.entries {
		sim := CosineSimilarity(query.Vec, entry.Result.Vec)
		if sim >= threshold {
			results = append(results, SimilarityResult{
				ID:         entry.ID,
				Similarity: sim,
				Distance:   math.Acos(math.Min(1.0, math.Max(-1.0, sim))),
			})
		}
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})
	if len(results) > k {
		return results[:k]
	}
	return results
}

// Len returns the number of entries in the index.
func (idx *Index) Len() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.entries)
}
