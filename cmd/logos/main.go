// Logos - Semantic Consensus Engine
//
// Usage:
//
//	logos start  --id=node1 --cluster-key=<hex> --cluster-size=3 --addr=:7700
//	logos propose --payload="hello world"
//	logos embed  --a="hello world" --b="hello, world!"
package main

import (
	"context"
	"encoding/hex"
	"flag"
	"fmt"
	"math"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/PrimelPJ/logos/pkg/consensus"
	"github.com/PrimelPJ/logos/pkg/node"
	"github.com/PrimelPJ/logos/pkg/semantic"
	"go.uber.org/zap"
)

const banner = `
  _
 | |    ___   __ _  ___  ___
 | |   / _ \ / _  |/ _ \/ __|
 | |__| (_) | (_| | (_) \__ \
 |_____\___/ \__, |\___/|___/
              |___/
  Semantic Consensus Engine  v0.1.0
  github.com/PrimelPJ/logos
`

func main() {
	fmt.Print(banner)

	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: logos <start|propose|embed>")
		os.Exit(1)
	}

	switch os.Args[1] {
	case "start":
		runStart()
	case "propose":
		runPropose()
	case "embed":
		runEmbed()
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", os.Args[1])
		os.Exit(1)
	}
}

func runStart() {
	fs := flag.NewFlagSet("start", flag.ExitOnError)
	nodeID := fs.String("id", "node1", "node identifier")
	clusterKeyHex := fs.String("cluster-key", "", "cluster key (hex-encoded 32 bytes)")
	clusterSize := fs.Int("cluster-size", 3, "total nodes in cluster")
	listenAddr := fs.String("addr", ":7700", "listen address")
	dataDir := fs.String("data", "./data", "data directory")
	_ = fs.Parse(os.Args[2:])

	logger, _ := zap.NewProduction()
	defer logger.Sync() //nolint:errcheck

	clusterKey, err := parseClusterKey(*clusterKeyHex)
	if err != nil {
		logger.Fatal("invalid cluster key", zap.Error(err))
	}

	cfg := node.DefaultConfig(*nodeID, clusterKey)
	cfg.ClusterSize = *clusterSize
	cfg.ListenAddr = *listenAddr
	cfg.DataDir = *dataDir

	sm := newDemoStateMachine(logger)
	n := node.New(cfg, sm, logger)

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	if err := n.Start(ctx); err != nil {
		logger.Fatal("failed to start node", zap.Error(err))
	}

	logger.Info("logos node running",
		zap.String("id", cfg.NodeID),
		zap.String("addr", cfg.ListenAddr),
		zap.Int("cluster_size", cfg.ClusterSize),
		zap.Float64("similarity_threshold", cfg.SimilarityThreshold),
	)

	<-ctx.Done()
	logger.Info("shutting down")
	n.Stop()
}

func runPropose() {
	fs := flag.NewFlagSet("propose", flag.ExitOnError)
	payload := fs.String("payload", "hello logos", "payload string to embed and display")
	clusterKeyHex := fs.String("cluster-key", "", "cluster key (hex)")
	_ = fs.Parse(os.Args[2:])

	clusterKey, err := parseClusterKey(*clusterKeyHex)
	if err != nil {
		fmt.Fprintf(os.Stderr, "invalid cluster key: %v\n", err)
		os.Exit(1)
	}

	embedder := semantic.NewEmbedder(clusterKey)
	result := embedder.Embed([]byte(*payload))

	fmt.Printf("payload:     %q\n", *payload)
	fmt.Printf("sha256:      %x\n", result.Hash)
	fmt.Printf("magnitude:   %.6f\n", result.Magnitude)
	fmt.Printf("dim:         %d\n", semantic.EmbeddingDim)
	fmt.Printf("vec[0:4]:    [%.4f  %.4f  %.4f  %.4f  ...]\n",
		result.Vec[0], result.Vec[1], result.Vec[2], result.Vec[3])
	fmt.Println()
	fmt.Println("(full proposal requires a running cluster)")
}

func runEmbed() {
	fs := flag.NewFlagSet("embed", flag.ExitOnError)
	a := fs.String("a", "hello world", "first string")
	b := fs.String("b", "hello, world!", "second string")
	clusterKeyHex := fs.String("cluster-key", "", "cluster key (hex)")
	_ = fs.Parse(os.Args[2:])

	clusterKey, err := parseClusterKey(*clusterKeyHex)
	if err != nil {
		fmt.Fprintf(os.Stderr, "invalid cluster key: %v\n", err)
		os.Exit(1)
	}

	embedder := semantic.NewEmbedder(clusterKey)
	ra := embedder.Embed([]byte(*a))
	rb := embedder.Embed([]byte(*b))

	sim := semantic.CosineSimilarity(ra.Vec, rb.Vec)
	dist := semantic.SemanticDistance(ra.Vec, rb.Vec)
	equiv := semantic.IsEquivalent(ra, rb, semantic.DefaultSimilarityThreshold)

	fmt.Printf("a:           %q\n", *a)
	fmt.Printf("b:           %q\n", *b)
	fmt.Printf("similarity:  %.6f\n", sim)
	fmt.Printf("distance:    %.6f rad  (%.2f deg)\n", dist, dist*180/math.Pi)
	fmt.Printf("equivalent:  %v  (threshold=%.2f)\n", equiv, semantic.DefaultSimilarityThreshold)
}

func parseClusterKey(hexStr string) ([]byte, error) {
	if hexStr == "" {
		return []byte("logos-demo-cluster-key-000000"), nil
	}
	b, err := hex.DecodeString(hexStr)
	if err != nil {
		return nil, fmt.Errorf("cluster key must be hex-encoded: %w", err)
	}
	return b, nil
}

type demoStateMachine struct {
	mu      sync.Mutex
	entries []consensus.Entry
	logger  *zap.Logger
}

func newDemoStateMachine(logger *zap.Logger) *demoStateMachine {
	return &demoStateMachine{logger: logger}
}

func (sm *demoStateMachine) Apply(entry consensus.Entry) error {
	sm.mu.Lock()
	sm.entries = append(sm.entries, entry)
	sm.mu.Unlock()
	sm.logger.Info("applied entry",
		zap.Uint64("index", entry.Index),
		zap.Uint64("term", entry.Term),
		zap.Int("payload_bytes", len(entry.Payload)),
	)
	return nil
}

func (sm *demoStateMachine) Snapshot() ([]byte, error) {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	return []byte(fmt.Sprintf(`{"entries":%d,"ts":%d}`, len(sm.entries), time.Now().Unix())), nil
}

func (sm *demoStateMachine) Restore(snapshot []byte) error {
	sm.logger.Info("restoring from snapshot", zap.Int("len", len(snapshot)))
	return nil
}
