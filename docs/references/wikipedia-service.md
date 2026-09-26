# Wikipedia Service Reference

## Purpose

`c2mem` uses a separately deployed Wikipedia retrieval service for FEVER and
HotpotQA. The deployable implementation is included under `wikipedia/`; it runs
outside the experiment process so the index remains loaded across tasks.

- C2Mem client: `src/datasets/wikipedia/client.py`
- HotpotQA consumer: `src/eval/hotpotqa_env.py`
- FEVER consumer: `src/eval/fever.py`
- Service implementation: `wikipedia/app/`
- Deployment guide: `wikipedia/README.md`
- Service location: configured by the user through `WIKI_SERVICE_URL`

## Critical Safety Rules

- Treat an existing Chroma index as read-only during experiments.
- Do not rebuild, migrate, overwrite, clear, or re-ingest an existing index
  unless that operation is intentional.
- If the service fails to start, the port is occupied, the embedding configuration is missing, or the Chroma collection cannot be opened, stop and ask before changing service code, changing startup mode, installing fallback behavior, or modifying database paths.
- Local embedding mode is not part of the default workflow and should not be used unless explicitly approved.
- Do not add silent fallbacks. Retrieval failures should be visible.

## Index

The index produced by the repository ingest command is:

```text
WIKI_SERVICE_CHROMA_DIR=/absolute/path/to/m2/.db/chroma
WIKI_SERVICE_COLLECTION=wiki_abstracts
embedding_dimension=<configured-model-output-dimension>
distance_space=cosine
```

The Chroma database contains embeddings for Wikipedia documents. It does not remove the need to embed each incoming search query.

## Query Embedding Requirement

The service performs vector retrieval in two steps:

1. Convert the incoming `/search` query text into a query embedding.
2. Query Chroma with `query_embeddings=[query_vector]`.

Relevant service code:

```python
query_vector = self.embedder.embed_query(query)
results = self.collection.query(query_embeddings=[query_vector], n_results=k)
```

Therefore, even though the database already contains all document embeddings, the service still needs a configured query embedding backend at runtime.

Configure the service with an OpenAI-compatible embedding endpoint:

```text
EMBEDDING_API_BASE=<embedding-base-url>
EMBEDDING_API_KEY=<embedding-api-key>
EMBEDDING_MODEL_NAME=<embedding-model>
```

The query model must be identical to the model used when building the document
index. The hosting stack and address are chosen by the user. The bundled
implementation provides the API below and must point at a non-empty
`wiki_abstracts` collection.

Configure the index path and collection:

```text
WIKI_SERVICE_CHROMA_DIR=<path-to-chroma>
WIKI_SERVICE_COLLECTION=wiki_abstracts
```

See `wikipedia/README.md` for the complete startup sequence.

## Health Checks

```bash
curl "$WIKI_SERVICE_URL/health"
curl "$WIKI_SERVICE_URL/ready"
curl -X POST "$WIKI_SERVICE_URL/search" \
  -H 'Content-Type: application/json' \
  -d '{"query":"Albert Einstein","top_k":3}'
```

Expected `/ready` shape:

```json
{
  "collection": "wiki_abstracts",
  "count": 4455000,
  "metadata": {
    "dimension": 1024,
    "hnsw:space": "cosine"
  }
}
```

The exact count and dimension depend on the completed ingest and selected
embedding model; readiness requires a nonzero count and a query model compatible
with the stored vectors.

## API

### `POST /search`

Request:

```json
{
  "query": "Colin Kaepernick starting quarterback 49ers season",
  "top_k": 10
}
```

Response includes:

```json
{
  "query": "...",
  "top_k": 10,
  "embedding_seconds": 0.25,
  "search_seconds": 0.01,
  "total_seconds": 0.26,
  "hits": [
    {
      "id": "...",
      "distance": 0.32,
      "document": "...",
      "metadata": {
        "title": "..."
      }
    }
  ]
}
```

### `POST /lookup`

Request:

```json
{
  "title": "Ed Wood"
}
```

Response includes:

```json
{
  "title": "Ed Wood",
  "found": true,
  "lookup_seconds": 0.81,
  "document": "Ed Wood: ...",
  "metadata": {
    "links": ["Plan 9 from Outer Space"]
  }
}
```

## FEVER and HotpotQA Semantics

`Search[query]` returns only candidate Wikipedia titles to the agent:

```text
Top matches:
1. Title A
2. Title B
3. Title C
```

`Lookup[title]` performs exact-title lookup and returns the page content plus links.

The HotpotQA runtime no longer falls back to local distractor context; retrieval should go through `wikipedia-service`.
