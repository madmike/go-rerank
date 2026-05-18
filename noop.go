package rerank

import (
	"context"
	"sort"
)

// NoopReranker returns candidates in their original order, preserving retrieval scores.
// Used for the 'lite' KB profile where the reranking overhead isn't justified.
type NoopReranker struct{}

func NewNoopReranker() *NoopReranker { return &NoopReranker{} }

func (r *NoopReranker) Rerank(_ context.Context, _ string, candidates []Candidate, topN int) ([]RankedResult, error) {
	out := make([]RankedResult, len(candidates))
	for i, c := range candidates {
		out[i] = RankedResult{ID: c.ID, Score: c.Score, Index: i}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	if topN > 0 && len(out) > topN {
		out = out[:topN]
	}
	return out, nil
}
