package rerank

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"time"
)

// BGEReranker calls a self-hosted bge-reranker-v2-m3 HTTP service.
// The service is expected to expose POST /rerank with the schema below.
// Deploy the service with: ghcr.io/huggingface/text-embeddings-inference
// pointing at BAAI/bge-reranker-v2-m3.
type BGEReranker struct {
	endpoint string     // e.g. "http://bge-reranker:8000"
	client   *http.Client
}

func NewBGEReranker(endpoint string) *BGEReranker {
	return &BGEReranker{
		endpoint: endpoint,
		client:   &http.Client{Timeout: 10 * time.Second},
	}
}

type bgeRerankRequest struct {
	Query     string   `json:"query"`
	Texts     []string `json:"texts"`
	Truncate  bool     `json:"truncate"`
	RawScores bool     `json:"raw_scores"`
}

type bgeRerankResponse struct {
	Index int     `json:"index"`
	Score float64 `json:"score"`
	Text  string  `json:"text,omitempty"`
}

func (r *BGEReranker) Rerank(ctx context.Context, query string, candidates []Candidate, topN int) ([]RankedResult, error) {
	if len(candidates) == 0 {
		return nil, nil
	}

	texts := make([]string, len(candidates))
	for i, c := range candidates {
		texts[i] = c.Content
	}

	body, _ := json.Marshal(bgeRerankRequest{
		Query:    query,
		Texts:    texts,
		Truncate: true,
	})

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, r.endpoint+"/rerank", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("bge reranker: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := r.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("bge reranker: http: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("bge reranker: status %d", resp.StatusCode)
	}

	var rows []bgeRerankResponse
	if err := json.NewDecoder(resp.Body).Decode(&rows); err != nil {
		return nil, fmt.Errorf("bge reranker: decode: %w", err)
	}

	out := make([]RankedResult, len(rows))
	for i, row := range rows {
		id := ""
		if row.Index >= 0 && row.Index < len(candidates) {
			id = candidates[row.Index].ID
		}
		out[i] = RankedResult{ID: id, Score: row.Score, Index: row.Index}
	}

	sort.Slice(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	if topN > 0 && len(out) > topN {
		out = out[:topN]
	}
	return out, nil
}
