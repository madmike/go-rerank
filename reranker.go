// Package rerank provides cross-encoder reranking for RAG pipelines.
// Three backends are available: BGE (self-hosted), Cohere (API), and Noop (pass-through).
// Selection is driven by environment/config; the Reranker interface is the only coupling
// point between the RAG strategy layer and any concrete backend.
package rerank

import "context"

// Candidate is a chunk presented for reranking.
type Candidate struct {
	ID      string
	Content string
	Score   float64 // original retrieval score; preserved in result if reranker fails
}

// RankedResult is a reranked candidate with its new score.
type RankedResult struct {
	ID    string
	Score float64
	Index int // original index in the candidate list
}

// Reranker reorders a list of candidates by relevance to a query.
type Reranker interface {
	// Rerank returns a list of IDs sorted by descending relevance score.
	// The returned slice may be shorter than candidates if TopN < len(candidates).
	Rerank(ctx context.Context, query string, candidates []Candidate, topN int) ([]RankedResult, error)
}
