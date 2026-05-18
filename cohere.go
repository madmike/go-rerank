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

// CohereReranker calls the Cohere rerank API (supports custom endpoint for OpenRouter).
type CohereReranker struct {
	apiKey   string
	model    string
	endpoint string // e.g. "https://api.cohere.com/v2/rerank" or "https://openrouter.ai/api/v1/rerank"
	client   *http.Client
}

func NewCohereReranker(apiKey, model, endpoint string) *CohereReranker {
	if model == "" {
		model = "rerank-v3.5"
	}
	if endpoint == "" {
		endpoint = "https://api.cohere.com/v2/rerank"
	}
	return &CohereReranker{
		apiKey:   apiKey,
		model:    model,
		endpoint: endpoint,
		client:   &http.Client{Timeout: 15 * time.Second},
	}
}

type cohereRerankRequest struct {
	Model     string   `json:"model"`
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
	TopN      int      `json:"top_n,omitempty"`
}

type cohereRerankResult struct {
	Index          int     `json:"index"`
	RelevanceScore float64 `json:"relevance_score"`
}

type cohereRerankResponse struct {
	Results []cohereRerankResult `json:"results"`
}

func (r *CohereReranker) Rerank(ctx context.Context, query string, candidates []Candidate, topN int) ([]RankedResult, error) {
	if len(candidates) == 0 {
		return nil, nil
	}

	docs := make([]string, len(candidates))
	for i, c := range candidates {
		docs[i] = c.Content
	}

	body, _ := json.Marshal(cohereRerankRequest{
		Model:     r.model,
		Query:     query,
		Documents: docs,
		TopN:      topN,
	})

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, r.endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("cohere reranker: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+r.apiKey)

	resp, err := r.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("cohere reranker: http: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("cohere reranker: status %d", resp.StatusCode)
	}

	var cr cohereRerankResponse
	if err := json.NewDecoder(resp.Body).Decode(&cr); err != nil {
		return nil, fmt.Errorf("cohere reranker: decode: %w", err)
	}

	out := make([]RankedResult, len(cr.Results))
	for i, row := range cr.Results {
		id := ""
		if row.Index >= 0 && row.Index < len(candidates) {
			id = candidates[row.Index].ID
		}
		out[i] = RankedResult{ID: id, Score: row.RelevanceScore, Index: row.Index}
	}

	sort.Slice(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	return out, nil
}
