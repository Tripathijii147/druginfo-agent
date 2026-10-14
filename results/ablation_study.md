# Retrieval & Architecture Ablation Study (100 Questions)

### 1. Retrieval Strategy Ablation (Chunk Size: 800 tokens)

| Strategy | Hit@2 | Hit@4 | Hit@6 | Key Advantage / Mechanism |
|---|---|---|---|---|
| **BM25 Only (Lexical)** | 57.0% | 78.0% | 89.0% | Fast exact keyword & drug name matching |
| **Dense Only (BGE-small-en)** | 89.0% | 93.0% | 96.0% | Semantic similarity & paraphrase recall |
| **Hybrid (Dense + BM25 via RRF)** | 80.0% | 91.0% | 94.0% | Reciprocal Rank Fusion of keyword + dense semantic lists |
| **Hybrid + Cross-Encoder Rerank** | 84.0% | 92.0% | 93.0% | Deep contextual cross-attention scoring on top candidates |

### 2. Chunk Size & Granularity Ablation (Hybrid + Reranking)

| Chunk Size / Overlap | Chunks | Hit@2 | Hit@4 | Hit@6 | Tradeoff Profile |
|---|---|---|---|---|---|
| **500 chars / 60 overlap** | 1,467 | 81.0% | 90.0% | 92.0% | High granularity, potential cross-chunk context boundary clipping |
| **800 chars / 100 overlap (Default)** | 960 | 84.0% | 92.0% | 93.0% | Optimal balance of clinical coherence and search precision |
| **1,200 chars / 150 overlap** | 662 | 85.0% | 93.0% | 93.0% | High context retention, lower signal-to-noise in retrieval rank |

### 3. Model Architecture & Inference Comparison

| LLM Backbone | Agent Routing Acc (10 cases) | Faithfulness (30 questions) | Avg Latency | Notes |
|---|---|---|---|---|
| **Meta Llama 3.3 70B** | 90% (9/10) | 93% (28/30) | ~2.4s | High capability; subject to endpoint deprecation & rate limits |
| **OpenAI GPT-OSS 20B (Active)** | 90% (9/10) | 93% (28/30) | ~1.1s | Fast inference, robust structured tool calling without reasoning loops |

