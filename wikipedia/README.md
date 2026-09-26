# Wikipedia Retrieval Service

This directory provides the independently deployable Wikipedia retrieval service
used by C2Mem's FEVER and HotpotQA environments.

The service keeps a Chroma index open in a long-running process, embeds each
`Search[query]`, and exposes title lookup for `Lookup[title]`.

## Interface

- `GET /health`: process health and active collection
- `GET /ready`: collection name, document count, and metadata
- `GET /stats`: same collection statistics as `/ready`
- `POST /warmup`: run one retrieval to warm the process
- `POST /search`: vector search with `{"query": str, "top_k": int}`
- `POST /lookup`: exact title lookup with `{"title": str}`

## Directory layout

```text
wikipedia/
├── app/
│   ├── config.py       # environment settings
│   ├── embedder.py     # API or local query embeddings
│   ├── main.py         # FastAPI application
│   ├── retrieval.py    # persistent Chroma retrieval
│   └── schemas.py      # request and response models
├── scripts/
│   └── build_chroma_from_snapshot.py
├── .env.example
├── Dockerfile
└── docker-compose.yml
```

Index data is generated locally from the public Wikipedia snapshot and stored
outside Git. Place the source data at the local paths below before building the
index.

## 1. Prepare the index

The simplest path is to build the index from the C2Mem repository root as
documented in the main README:

```bash
python -m src.datasets.hotpotqa.abstracts_ingest \
  --archive data/wikipedia/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2 \
  --limit 0 --batch-size 5000
```

This produces `.db/chroma/` with collection `wiki_abstracts`. The query service
must use exactly the same embedding model that was used while building this
index. Changing the model can change vector dimensions or similarity geometry
and makes retrieval invalid.

If you independently obtained local records containing `embedding` plus
`chunk` or `document`, you can instead build an index inside this directory:

```bash
python -m scripts.build_chroma_from_snapshot \
  --snapshot-dir ./data/snapshot \
  --target-dir ./data/chroma \
  --collection wiki_abstracts
```

## 2. Configure the service

Create the local environment file:

```bash
cd wikipedia
cp .env.example .env
```

For an embedding API, fill in all three values:

```dotenv
WIKI_SERVICE_EMBED_MODE=api
EMBEDDING_API_BASE=your-embedding-base-url
EMBEDDING_API_KEY=your-embedding-api-key
EMBEDDING_MODEL_NAME=your-embedding-model
```

`EMBEDDING_API_BASE` is the API base URL. The service appends `/embeddings`
unless the configured value already ends with that path.

To embed locally instead:

```dotenv
WIKI_SERVICE_EMBED_MODE=local
WIKI_SERVICE_LOCAL_EMBED_MODEL=/absolute/path/to/local-embedding-model
```

Set the index mount and collection. When deploying from this directory inside
the C2Mem repository, the defaults already select the index built above:

```dotenv
WIKI_SERVICE_CHROMA_HOST_DIR=../.db/chroma
WIKI_SERVICE_COLLECTION=wiki_abstracts
```

For a standalone server, copy the Chroma directory to that server and change
`WIKI_SERVICE_CHROMA_HOST_DIR` to its host path. Do not copy the index while it
is being written. The service exposes no index mutation endpoint; do not run an
ingest or rebuild command against the mounted deployment index.

## 3. Deploy with Docker Compose

From `wikipedia/`:

```bash
docker compose up -d --build
docker compose ps
docker compose logs -f wikipedia-service
```

The host port is controlled by `WIKI_SERVICE_PORT`. Allow access only from the
machines that run C2Mem; use your normal firewall, private network, or reverse
proxy policy for the deployment environment.

To stop the service:

```bash
docker compose down
```

## 4. Deploy directly with Python

```bash
cd wikipedia
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
set -a
source .env
set +a
uvicorn app.main:app --host "$WIKI_SERVICE_HOST" --port "$WIKI_SERVICE_PORT"
```

For direct deployment, `WIKI_SERVICE_CHROMA_DIR` is the path read by the Python
process. Docker Compose instead mounts `WIKI_SERVICE_CHROMA_HOST_DIR` at the
container's `/app/data/chroma`.

## 5. Verify the deployment

Set `WIKI_SERVICE_URL` in the C2Mem environment to the base URL assigned to
your deployed service, then verify the contract:

```bash
curl "$WIKI_SERVICE_URL/health"
curl "$WIKI_SERVICE_URL/ready"
curl -X POST "$WIKI_SERVICE_URL/warmup"
curl -X POST "$WIKI_SERVICE_URL/search" \
  -H 'Content-Type: application/json' \
  -d '{"query":"Albert Einstein","top_k":3}'
curl -X POST "$WIKI_SERVICE_URL/lookup" \
  -H 'Content-Type: application/json' \
  -d '{"title":"Albert Einstein"}'
```

`/ready` should report a nonzero count before experiments start. An empty or
missing collection is a deployment error.
