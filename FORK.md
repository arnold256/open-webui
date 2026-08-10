# Fork manifest

This fork carries four patches on top of upstream `open-webui`. Every patch here
is intended to be **temporary** — each one is either awaiting an upstream PR or
has a documented condition under which it gets deleted.

If you are about to rebase this fork onto a newer upstream, read
[`REBASE_PROMPT.md`](REBASE_PROMPT.md) — it is the working procedure.

- **Upstream remote:** `origin` → `https://github.com/open-webui/open-webui`
- **Fork remote:** `open-webui-(GPA)` → `https://github.com/arnold256/open-webui`
- **PRs target `dev`.** Upstream auto-closes PRs opened against `main`.
- **Deployed branch:** `deploy/gpa` — upstream `dev` plus the four patches, rebased linear.
- **Consumed by:** `D:\source\Parser` (`docker-compose.yml`, service `open-webui`).

## Branch layout

| Branch                        | Purpose                                                    |
| ----------------------------- | ---------------------------------------------------------- |
| `fix/no-double-split`         | PR branch — 1 commit on `origin/dev`                       |
| `feat/external-text-splitter` | PR branch — 1 commit on `origin/dev`                       |
| `feat/loader-metadata`        | PR branch — 1 commit on `origin/dev`                       |
| `perf/reuse-embeddings`       | PR branch — 2 commits; **stacks on `fix/no-double-split`** |
| `deploy/gpa`                  | all four, rebased linear, + this manifest                  |
| `backup/fork-v0.9.2`          | the original pre-0.11.0 fork; safe to delete once trusted  |

The PR branches are the source of truth. `deploy/gpa` is regenerated from them.

---

## Patch 1 — `fix: don't re-split already-chunked documents`

**Branch:** `fix/no-double-split` · **Size:** 6 lines, `routers/retrieval.py`

Uploading a file into a knowledge base runs `process_file` twice in one request
(`routers/files.py`, once with no collection and once with `collection_name`).
The second call reads the already-chunked documents back out of `file-{id}` and
passes them to `save_docs_to_vector_db` without `split=False`, so they are
chunked a second time. With `ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER` on by default
the second pass genuinely re-fragments each chunk on its headers.

`save_docs_to_vector_db` has accepted `split: bool = True` all along and no
caller has ever passed it (`git log -S "split=False"` → empty).

- **Upstream status:** unreported. Needs an issue before the PR.
- **Delete when:** merged upstream.
- **Verify:** upload a multi-header `.md` into a knowledge base; row counts in
  `document_chunk` for `file-{id}` and for the knowledge collection must match.

## Patch 2 — `feat: add external text splitter option`

**Branch:** `feat/external-text-splitter` · **Size:** 364 lines + 9 i18n keys × 63 locales

Adds `TEXT_SPLITTER == 'external'`, delegating chunking to an HTTP service.
Mirrors the external _document loader_ in naming and plumbing:
`EXTERNAL_TEXT_SPLITTER_URL` / `_API_KEY` / `_HEADERS` / `_TIMEOUT`, with the
same templated custom headers and user-info forwarding.

This is the patch this deployment actually depends on: it routes chunking to the
Parser service, which selects chunknorris/astchunk per file type.

Design decisions a future maintainer will otherwise re-litigate:

- **New file** `retrieval/splitters/external_splitter.py` (124 lines) — deliberately
  isolated so it never conflicts on rebase. No `__init__.py`; neither
  `retrieval/` nor `retrieval/loaders/` has one.
- **Duck-typed `split_documents`**, not a LangChain `TextSplitter` subclass, whose
  `split_text(str)` signature would drop `Document.metadata` and break
  chunk-to-file attribution.
- **One request per document.** `save_docs_to_vector_db` is also called with docs
  from several files at once (`process_files_batch`); batching would make chunks
  unattributable when the service does not echo metadata back.
- **`external` skips the markdown-header pre-pass.** That toggle defaults to on,
  so without the guard the default behaviour would be to fragment locally and
  then ship fragments to the service. The UI hides the switch and its dependent
  Chunk Min Size Target field while `external` is selected.
- **Failures raise, never fall back** to the character splitter — a silent
  fallback reports success over a collection chunked by the wrong policy,
  detectable only later as degraded retrieval and fixable only by a reindex.
- **`_TIMEOUT` exists because its absence hangs forever.** Verified: a service that
  accepts the connection but never replies pins a threadpool worker indefinitely.
  `ExternalDocumentLoader` still has this flaw upstream — worth a separate issue.
- **`EXTERNAL_TEXT_SPLITTER_TIMEOUT` is typed `Union[int, str | None]`**, matching
  `FILE_MAX_SIZE`. Pydantic v2 rejects `int` for a `str | None` field and the
  admin UI's number input sends a number.

Adding a config option touches 7 places: `config.py` declaration, `DEFAULT_CONFIG`,
`RETRIEVAL_CONFIG_KEYS`, `get_rag_config`, `ConfigForm`, `update_rag_config`, and
the update-response dict. No migration entry is needed — `DEFAULT_CONFIG` seeds
new keys.

- **Upstream status:** not submitted. Needs a Discussion first (new setting).
- **Delete when:** merged upstream, **or** if the Parser is changed to return
  pre-chunked documents from `/parse/process` — a JSON array response is already
  handled by the stock `ExternalDocumentLoader`, which would remove this patch
  entirely. That alternative requires `ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER=false`
  and `CHUNK_MIN_SIZE_TARGET=0`.
- **Verify:** see `Parser/docs/` and the smoke checks below.

## Patch 3 — `feat: persist document loader metadata to the file record`

**Branch:** `feat/loader-metadata` · **Size:** 19 lines, `routers/retrieval.py`

Stores loader-returned document metadata under `meta.loader_metadata` on the file
record. Values go through `process_metadata` so they are DB-safe; the four fields
Open WebUI sets on chunk metadata itself (`name`, `created_by`, `file_id`,
`source`) are skipped. Also written on the `BYPASS_EMBEDDING_AND_RETRIEVAL` path.

**This patch is load-bearing for this deployment.**
`Parser/services/functions/enhanced_rag.py::_build_file_metadata` reads
`meta.loader_metadata` and flattens it to `loader_*` fields for the
`get_file_content` and `list_knowledge_base_files` tools — that is how the Parser's
enriched fields (`title`, `summary`, `document_type`, `rev`, `keywords`,
`headings`, `process_area`, …) reach the model.

- **Upstream status:** not submitted; the weakest case of the four, since it serves
  a specific integration pattern. Open a Discussion before a PR.
- **Delete when:** merged upstream, **or** `enhanced_rag.py` is changed to read the
  same fields off pgvector `vmetadata` instead. They are already there — chunk
  metadata is built as `{**doc.metadata, ...}` with no key filtering, and
  chunknorris copies `doc.metadata` onto every chunk. That change is ~50 lines in
  the Parser repo and needs a per-file vector lookup, but it removes an OWUI patch.
- **Verify:** after upload, `GET /api/v1/files/{id}` shows `meta.loader_metadata`;
  the `list_knowledge_base_files` tool returns `loader_title` / `loader_summary`.

## Patch 4 — `perf: reuse existing embeddings when linking a file to a knowledge collection`

**Branch:** `perf/reuse-embeddings` (stacks on `fix/no-double-split`) · **Size:** 143 lines across 4 files

Every knowledge-base upload embeds the same text twice — once into `file-{id}`,
once into the knowledge collection. With hosted embeddings (this deployment uses
Azure) that is a doubled bill on every upload.

Adds `query_with_vectors`, an **optional** method on `VectorDBBase` returning `None`
by default, so backends opt in rather than being forced to grow a parameter.
Only pgvector implements it; the other 14 backends fall back to `query` and
re-embed exactly as before — which is why Patch 1 must land first.

- Reuse is refused unless every chunk's recorded `embedding_config` matches the
  currently configured engine and model, so vectors survive a model change safely.
- Skipped under `PGVECTOR_PGCRYPTO`, where the query path selects decrypted
  columns and never loads the rows the vectors hang off.
- pgvector returns numpy scalars, hence the explicit `float()` conversion.

- **Upstream status:** not submitted. Link
  [Discussion #8240](https://github.com/open-webui/open-webui/discussions/8240) —
  the maintainer confirms the duplication is by design _and_ proposes this exact
  fix ("We could implement a mechanism to reuse the first set of embeddings when
  associating the file with the knowledge collection"). That satisfies the PR
  template's linked-discussion requirement.
- **Delete when:** merged upstream.
- **Verify:** one knowledge-base upload should produce one embedding pass, not two.

---

## Removed in the v0.9.2 → 0.11.0 upgrade

Recorded so nobody reintroduces them.

| Patch                                        | Why it went                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **BM25 native search**                       | Superseded. Upstream shipped `VectorDBBase.hybrid_search()` + `PgvectorClient.hybrid_search()` with `plainto_tsquery`/`ts_rank_cd` and a GIN index, plus `query_doc_with_native_hybrid_search()`. Note upstream indexes with `to_tsvector('simple', …)` — **no stemming**, where we used `'english'`. On engineering documents ("valve"/"valves") that costs some recall. Accepted; both the index and the query would need changing to fix. |
| **`X-File-Metadata` on the external loader** | Superseded _and_ it was dead code — it read `file.meta['data']`, but `data` and `meta` are separate columns on `FileModel` and nothing writes a `data` key into `meta`, so the header was never sent. Enrichment was running purely off the Parser's own `metadata_enrichment.default_generate_metadata: true`. Upstream's `EXTERNAL_DOCUMENT_LOADER_HEADERS` covers this properly with templated placeholders and zero code.                |
| **In-memory embedding cache**                | Replaced by Patch 4, which eliminates the redundant embedding rather than caching it, needs no new settings, and is pre-blessed by the maintainer. The original also had two defects: a failed batch was silently skipped, which mispaired vectors with chunks; and on the sentence-transformers path list queries were written but never read.                                                                                              |

---

## Deployment contract with the Parser

Set in `D:\source\Parser\docker-compose.yml`, service `open-webui`:

| Variable                         | Value                                                                                    |
| -------------------------------- | ---------------------------------------------------------------------------------------- |
| `RAG_TEXT_SPLITTER`              | `external`                                                                               |
| `EXTERNAL_TEXT_SPLITTER_URL`     | `http://parser:5000/api/split/` — **used verbatim**, include the path and trailing slash |
| `EXTERNAL_TEXT_SPLITTER_TIMEOUT` | `600`                                                                                    |
| `VECTOR_DB`                      | `pgvector`                                                                               |

The Parser needs no OWUI-specific code. Its `/api/split/` already accepts
`{documents, chunk_size, chunk_overlap}` and returns `{success, chunks, count}`;
its per-extension and per-extractor splitter preferences resolve from
`metadata.filename` and `metadata.extracted_by`, both carried through from
`/parse/process`.

`ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER` does **not** need setting — Patch 2 makes
`external` imply it.

## Build

`docker build .` uses whatever branch is checked out. Check out `deploy/gpa` first,
and prefer tagging by commit so a running container traces back to a revision:

```bash
git checkout deploy/gpa && docker build -t ghcr.io/arnold256/open-webui:$(git rev-parse --short HEAD) .
```

## Smoke checks after any rebase

1. Upload a multi-header `.md` into a knowledge base.
2. The Parser receives the **whole document including every header** — proof the
   markdown pre-pass was skipped.
3. Stored chunks carry intact `file_id` / `name` / `source` / `hash`.
4. Chunk counts match between `file-{id}` and the knowledge collection.
5. One embedding pass, not two.
6. `GET /api/v1/files/{id}` shows `meta.loader_metadata`.
7. A `list_knowledge_base_files` tool call returns `loader_title` / `loader_summary`.

## Known-red upstream CI (not caused by this fork)

As of upstream `dev` @ `2dadc5435`, the frontend workflow runs `npm run format`
and `npm run i18n:parse` then `git diff --exit-code`, and **both already dirty the
tree on a pristine checkout**:

- `npm run format` reformats `AddTerminalServerModal.svelte`, `ChatControls.svelte`,
  `FileNav.svelte`, `FileNavToolbar.svelte`.
- `npm run i18n:parse` adds `Orchestrator`, `Per automation`, `Per chat`,
  `Start the chat to use this terminal.`, `Terminal Contexts`, `Upload failed`,
  `Waiting for upload` and removes one unreferenced key.

Our patches are clean under both. Expect that CI failure on any PR and say so in
the PR body. Do **not** absorb that churn into a feature commit — it breaks the
atomicity the PR template requires.

Local tooling note: `.npmrc` sets `engine-strict=true` and pins Node `<=22.x.x`.
On newer Node, install with `npm ci --engine-strict=false`. CI uses Node 22, so
that is the first thing to check if formatting output ever disagrees.
