# On-Device RAG

**Status:** 0.2.0-dev implementation
**Primary modules:** `TinyBrainRAG`, `TinyBrainProximaKit`, ProximaKit
**CLI:** `swift run tinybrain-rag`

---

## 1. Overview

Retrieval-augmented generation (RAG) answers a question by first searching a
local document index, then grounding the model prompt in the retrieved passages.
TinyBrain's on-device RAG path keeps that whole loop on the machine:

```
chunk -> embed -> HNSW retrieve -> budgeted cited prompt -> streamed answer -> citation mapping
```

The generation side is TinyBrain. The retrieval side is ProximaKit. The glue
between them lives in `Sources/TinyBrainRAG/`: token-aware chunking, a
metadata-carrying HNSW wrapper, prompt budgeting, citation parsing, and an
engine that streams answer tokens.

The shipped demo is a command-line tool, not a ChatDemo UI integration. It can
index built-in sample notes or a folder of local `.txt` and `.md` files, print
the retrieved passages with distances, and either stream a TinyBrain model
answer or stop after retrieval with `--no-generate`.

---

## 2. Pipeline

### 2.1 Chunk

`DocumentChunker` splits documents into `DocumentChunk` values using the same
tokenizer family that prompt budgeting uses. It prefers paragraph boundaries,
then sentence boundaries, and only hard-splits when the next boundary would
exceed the target size.

The CLI uses `targetTokens: 128`, `overlapTokens: 24`, and paragraph-aware
splitting. Each chunk stores its text, source path, ordinal, and token range;
that metadata is later encoded into the vector index entry.

### 2.2 Embed

`RAGIndex` accepts any ProximaKit `TextEmbedder`. The CLI default is
`NLEmbeddingProvider(language: .english)` from `ProximaEmbeddings`, which does
not require a TinyBrain model file for indexing.

There is also a TinyBrain bridge at
`Sources/TinyBrainProximaKit/TinyBrainEmbedder.swift`. It tokenizes text, runs a
TinyBrain `ModelRunner`, and returns the final hidden state as a ProximaKit
vector. That makes an all-TinyBrain embedding path possible for app code, but
the shipped CLI exposes only `nl` and `stub` embedder choices today.

### 2.3 HNSW Retrieve

`RAGIndex` wraps ProximaKit's actor-based `HNSWIndex`. On add, it embeds chunk
text and stores the full `DocumentChunk` as JSON metadata. On search, it embeds
the query, calls HNSW search, decodes metadata, and returns `RetrievedPassage`
values containing the chunk, ProximaKit distance, and zero-based retrieval rank.

`RAGIndex.save(to:)` and `RAGIndex.load(from:embedder:)` are available for
library callers. The current CLI builds an in-memory index each run.

### 2.4 Budgeted Cited Prompt

`RAGPromptBuilder` renders numbered passages like `[1]`, `[2]`, and instructs
the model to answer only from those passages and cite claims with the marker.

The builder measures the rendered prompt with the tokenizer. It keeps passages
in retrieval-rank order until adding the next passage would exceed the prompt
limit. It never truncates a passage mid-text; lower-ranked passages are dropped
instead.

### 2.5 Streamed Answer

`RAGEngine.answerStream(_:)` emits retrieved passages first, then decoded answer
tokens, then a final `.done` event with parsed citations. The concrete TinyBrain
adapter is `ModelRunnerGenerator`, which serializes access to `ModelRunner` and
streams sampled `TokenOutput` values through the `AnswerGenerator` protocol.

### 2.6 Citation Mapping

`CitationParser` scans the completed answer for numeric markers such as `[1]`
and `[12]`. It maps `[n]` to the included prompt passage at index `n - 1`.
Malformed markers are ignored, and out-of-range markers produce citations with
`passage == nil` instead of trapping.

---

## 3. Component Map

| File | Responsibility |
|------|----------------|
| `Sources/TinyBrainRAG/DocumentChunker.swift` | Token-measured chunking for in-memory text and UTF-8 files |
| `Sources/TinyBrainRAG/RAGIndex.swift` | ProximaKit `HNSWIndex` wrapper with JSON chunk metadata |
| `Sources/TinyBrainRAG/RAGPromptBuilder.swift` | Numbered-passage prompt rendering under a token budget |
| `Sources/TinyBrainRAG/CitationParser.swift` | Tolerant `[n]` citation extraction and passage resolution |
| `Sources/TinyBrainRAG/AnswerGenerator.swift` | Generation protocol and `ModelRunnerGenerator` adapter |
| `Sources/TinyBrainRAG/RAGEngine.swift` | Index, retrieve, prompt, stream, and cite orchestration |
| `Sources/TinyBrainRAG/RetrievalTool.swift` | `retrieve` tool schema and handler for tool calling |
| `Sources/TinyBrainRAG/DeterministicStubEmbedder.swift` | Seeded, hermetic test embedder |
| `Sources/TinyBrainProximaKit/TinyBrainEmbedder.swift` | TinyBrain `ModelRunner` to ProximaKit `TextEmbedder` bridge |
| `Examples/RAGDemo/RAGDemo.swift` | `tinybrain-rag` CLI |
| `Tests/TinyBrainRAGTests/` | Offline tests for chunking, prompt budgeting, indexing, engine streaming, and tools |

---

## 4. Embedding Providers

| Provider | Where | Tradeoff |
|----------|-------|----------|
| `NLEmbeddingProvider(language: .english)` | `Examples/RAGDemo/RAGDemo.swift` via `ProximaEmbeddings` | CLI default. No `.tbf` model is needed for embedding. It is a purpose-built system embedding provider, but TinyBrain does not publish recall benchmarks for it here. |
| `TinyBrainEmbedder` | `Sources/TinyBrainProximaKit/TinyBrainEmbedder.swift` | All-TinyBrain option for library callers. It runs a decoder forward pass per text and pools the final hidden state, so indexing is heavier and retrieval quality should be treated as experimental. |
| `DeterministicStubEmbedder` | `Sources/TinyBrainRAG/DeterministicStubEmbedder.swift` | Test and smoke-demo option. It is deterministic and offline, not a semantic-quality embedding model. |

The important API boundary is `any TextEmbedder`. Apps can bring a better local
embedder without changing chunking, indexing, prompt building, or citation
mapping.

---

## 5. Retrieve Tool Seam

TB-012 already exposes retrieval as a TinyBrain tool. `RetrievalTool` defines a
`retrieve` schema with required `query` and optional `k`, clamps `k` to a
configured range, calls the same retrieval closure used by `RAGEngine`, and
returns numbered passages with source paths and distances.

That is the seam TB-011 can register inside a private agent runtime. The larger
agent loop is not implemented in `TinyBrainRAG`: planning, repeated tool calls,
memory policy, and UI affordances remain follow-up work. The shipped piece is
the local retrieval tool that an agent can call.

---

## 6. CLI Usage

Run against built-in sample notes, or run retrieval only without loading a `.tbf`
model:

```bash
swift run tinybrain-rag -question "What stays on device?"
swift run tinybrain-rag --no-generate -question "What does TinyBrain retrieve?"
```

Index a local notes folder:

```bash
swift run tinybrain-rag \
  --dir Notes \
  --model Models/tinyllama-1.1b-int8.tbf \
  --embedder nl \
  --k 4 \
  --tokens 64 \
  -question "What do my notes say about battery diagnostics?"
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--dir <path>` | Built-in sample notes | Recursively indexes `.txt` and `.md` files |
| `--model <path>` | `Models/tinyllama-1.1b-int8.tbf` | TinyBrain model used for answer generation |
| `--embedder <nl|stub>` | `nl` | Embedding provider for the index |
| `-question "..."` | Interactive REPL | Question to answer; repeat for scripted runs |
| `--k <n>` | `4` | Number of passages to retrieve |
| `--tokens <n>` | `64` | Maximum generated answer tokens |
| `--template <zephyr|none>` | `zephyr` | Prompt template used for generation |
| `--no-generate` | `false` | Print retrieval results and skip model loading |

If generation is enabled and the model file is missing, the CLI prints the
expected path and conversion command, then exits without generation. With
`--no-generate`, it can still demonstrate indexing and retrieval.

---

## 7. Design Notes

### 7.1 2048-Token Budget Math

`PromptBudget` defaults to `contextWindow = 2_048` and
`generationHeadroom = 256`. The prompt builder computes
`promptLimit = contextWindow - generationHeadroom`, renders each candidate
prompt, and checks the full tokenized prompt against that limit. The full prompt
includes the scaffold, question, `Passages:` labels, and every included passage,
so passage room is whatever remains after scaffold and question tokens count.

### 7.2 Why Citations Parse Post-Stream

During generation, the engine streams decoded token fragments immediately. A
citation marker can span token boundaries, and the parser also needs stable
string ranges in the final answer. `RAGEngine` accumulates the answer text while
streaming and runs `CitationParser` only when generation finishes.

### 7.3 CI-Hermetic Testing

RAG tests do not require model files, downloads, or network access. They use
`DeterministicStubEmbedder` for stable HNSW ordering, scripted
`AnswerGenerator` fakes for end-to-end answer and citation tests, and tokenizer
fixtures for prompt and chunk token counts. Tests stay focused on chunk
boundaries, metadata round-trips, prompt budgets, event order, and citations.
