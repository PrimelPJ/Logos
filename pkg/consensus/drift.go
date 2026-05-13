// Package consensus - drift monitor
// The DriftMonitor continuously samples the semantic fingerprints of all cluster
// nodes and raises alerts when divergence exceeds configured thresholds.
// It is the "nervous system" of Logos's self-healing reconciliation loop.
package consensus

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/PrimelPJ/logos/pkg/semantic"
	"go.uber.org/zap"
)

// DriftLevel classifies how much semantic divergence exists in the cluster.
type DriftLevel int

const (
	DriftNone     DriftLevel = iota // All nodes within θ of each other
	DriftMinor                      // Divergence detectable but below warn threshold
	DriftWarning                    // At or above warn threshold — log it
	DriftCritical                   // Above critical threshold — trigger reconciliation
)

func (d DriftLevel) String() string {
	switch d {
	case DriftNone:
		return "NONE"
	case DriftMinor:
		return "MINOR"
	case DriftWarning:
		return "WARNING"
	case DriftCritical:
		return "CRITICAL"
	}
	return "UNKNOWN"
}

// NodeFingerprint holds a node's semantic log fingerprint at a point in time.
type NodeFingerprint struct {
	NodeID      string
	Term        uint64
	LogLen      uint64
	Fingerprint semantic.Vector // centroid of last N log entries
	CapturedAt  time.Time
}

// DriftSample is one observation of cluster-wide semantic drift.
type DriftSample struct {
	Timestamp    time.Time
	Level        DriftLevel
	Report       semantic.DriftReport
	Fingerprints []NodeFingerprint
}

// DriftMonitor periodically collects fingerprints from all nodes and
// computes cluster-wide semantic drift metrics.
type DriftMonitor struct {
	mu           sync.RWMutex
	cfg          DriftMonitorConfig
	history      []DriftSample
	maxHistory   int
	reconcileC   chan []string // sends outlier node IDs that need reconciliation
	alertC       chan DriftSample
	logger       *zap.Logger
	fingerprints map[string]NodeFingerprint
}

// DriftMonitorConfig configures the drift monitor.
type DriftMonitorConfig struct {
	SampleInterval   time.Duration
	WarnAngle        float64 // radians, e.g. 0.3 (~17°)
	CriticalAngle    float64 // radians, e.g. 0.6 (~34°)
	FingerprintDepth int     // number of recent log entries to fingerprint
}

// DefaultDriftMonitorConfig returns sensible defaults.
func DefaultDriftMonitorConfig() DriftMonitorConfig {
	return DriftMonitorConfig{
		SampleInterval:   2 * time.Second,
		WarnAngle:        0.30,
		CriticalAngle:    0.60,
		FingerprintDepth: 50,
	}
}

// NewDriftMonitor creates a DriftMonitor.
func NewDriftMonitor(cfg DriftMonitorConfig, logger *zap.Logger) *DriftMonitor {
	return &DriftMonitor{
		cfg:          cfg,
		maxHistory:   1000,
		reconcileC:   make(chan []string, 16),
		alertC:       make(chan DriftSample, 64),
		logger:       logger,
		fingerprints: make(map[string]NodeFingerprint),
	}
}

// UpdateFingerprint registers or updates a node's latest fingerprint.
// Nodes call this on every heartbeat via the transport layer.
func (m *DriftMonitor) UpdateFingerprint(fp NodeFingerprint) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.fingerprints[fp.NodeID] = fp
}

// Run starts the drift monitor's sampling loop.
func (m *DriftMonitor) Run(ctx context.Context) {
	ticker := time.NewTicker(m.cfg.SampleInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			m.sample()
		}
	}
}

func (m *DriftMonitor) sample() {
	m.mu.Lock()
	fps := make([]NodeFingerprint, 0, len(m.fingerprints))
	for _, fp := range m.fingerprints {
		fps = append(fps, fp)
	}
	m.mu.Unlock()

	if len(fps) < 2 {
		return
	}

	embeds := make([]semantic.EmbedResult, len(fps))
	for i, fp := range fps {
		embeds[i] = semantic.EmbedResult{Vec: fp.Fingerprint}
	}

	report := semantic.AnalyzeDrift(embeds)

	level := DriftNone
	switch {
	case report.MaxDistance >= m.cfg.CriticalAngle:
		level = DriftCritical
	case report.MaxDistance >= m.cfg.WarnAngle:
		level = DriftWarning
	case report.MaxDistance > 0.01:
		level = DriftMinor
	}

	sample := DriftSample{
		Timestamp:    time.Now(),
		Level:        level,
		Report:       report,
		Fingerprints: fps,
	}

	m.mu.Lock()
	m.history = append(m.history, sample)
	if len(m.history) > m.maxHistory {
		m.history = m.history[1:]
	}
	m.mu.Unlock()

	if level >= DriftWarning {
		m.logger.Warn("semantic drift detected",
			zap.String("level", level.String()),
			zap.Float64("max_distance_rad", report.MaxDistance),
			zap.Float64("mean_distance_rad", report.MeanDistance),
			zap.Ints("outlier_indices", report.OutlierIndices),
		)
	}

	if level >= DriftWarning {
		select {
		case m.alertC <- sample:
		default:
		}
	}

	if level == DriftCritical && len(report.OutlierIndices) > 0 {
		outlierIDs := make([]string, 0, len(report.OutlierIndices))
		for _, idx := range report.OutlierIndices {
			if idx < len(fps) {
				outlierIDs = append(outlierIDs, fps[idx].NodeID)
			}
		}
		m.logger.Error("critical drift — requesting reconciliation",
			zap.Strings("outlier_nodes", outlierIDs),
		)
		select {
		case m.reconcileC <- outlierIDs:
		default:
		}
	}
}

// Alerts returns a channel that emits DriftSamples when drift exceeds DriftWarning.
func (m *DriftMonitor) Alerts() <-chan DriftSample {
	return m.alertC
}

// ReconcileRequests returns a channel of node ID slices that need reconciliation.
func (m *DriftMonitor) ReconcileRequests() <-chan []string {
	return m.reconcileC
}

// RecentHistory returns the last n drift samples.
func (m *DriftMonitor) RecentHistory(n int) []DriftSample {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if n > len(m.history) {
		n = len(m.history)
	}
	result := make([]DriftSample, n)
	copy(result, m.history[len(m.history)-n:])
	return result
}

// Summary returns a human-readable cluster drift summary.
func (m *DriftMonitor) Summary() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if len(m.history) == 0 {
		return "no samples collected yet"
	}
	latest := m.history[len(m.history)-1]
	return fmt.Sprintf(
		"drift=%s max=%.4f rad mean=%.4f rad nodes=%d outliers=%d",
		latest.Level,
		latest.Report.MaxDistance,
		latest.Report.MeanDistance,
		len(latest.Fingerprints),
		len(latest.Report.OutlierIndices),
	)
}
