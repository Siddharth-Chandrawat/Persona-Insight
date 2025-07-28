**2️⃣ Round 1B: Persona‑Driven Insights**

## Approach Explanation

Our goal is to build a lightweight, persona-driven PDF analyzer that pinpoints the most relevant passages according to a specific reader’s needs. Whether you’re a financial analyst hunting for risk factors or a technical manager seeking system metrics, this tool transforms static documents into a personalized summary, all without generating fluff.

### 1. Discovering Your Persona & Job

First, we look for a `meta.json` file in the `input/` folder. If it exists, we extract:

* **`persona`**: Who’s reading? (e.g., “Technical Manager”)
* **`job`**: What are they trying to achieve? (e.g., “identify key performance metrics”)
* **`top_k`**: How many top passages to return (default is 15).

If no metadata is provided, the system defaults to a generic lens (`“generic reader relevance”`). We then merge persona and job into a single query string that steers every downstream step.

### 2. Breaking PDFs into Bite‑Sized Chunks

Rather than feeding whole documents into the model, we split each page into smaller, semantically meaningful pieces. We support two chunking strategies:

1. **Visual Blocks**

   * Uses PyMuPDF’s `get_text("dict")` API to preserve headings, lists, and paragraph boundaries as they appear on the page.
2. **Logical Paragraphs**

   * Splits the raw text by double-newlines (`"\n\n"`) to capture natural paragraph breaks.

Each chunk is cleaned up (extra whitespace removed) and only retained if it’s at least 20 characters long, ensuring we ignore trivial fragments.

### 3. Crafting Embeddings & Computing Similarity

With chunks in hand, we load a locally cached MiniLM model from Hugging Face (`/app/model`).

* **Query Embedding**: The combined persona/job string is encoded into a single vector.
* **Chunk Embeddings**: All text chunks are processed in batches for maximum throughput.
* **Scoring**: We calculate cosine similarity between the query vector and each chunk vector, giving us a relevance score for every snippet.

### 4. Ranking & Extracting Top Passages

Scores are sorted in descending order, and the highest-scoring passages (Top‑K) are selected. This extractive approach ensures you see the actual text that best matches your query, without any synthesized content.

### 5. Generating the Final Output

The results are saved to `round1b/output/output.json`, which includes:

* **Metadata**: Processed file list, persona/job values, timing metrics, and a timestamp.
* **Extracted Sections**: For each top passage, we record a short unique ID, document name, page number, text snippet, and its relevance score.

This pipeline is deterministic, scalable, and free of hallucinations. You can swap in a larger transformer or layer on an abstractive summarizer later—but at its core, this extractive method gives you a fast, transparent way to surface the most important information for any reader’s persona and goal.

* **Tech:** SentenceTransformers, Faiss, lightweight summarizer.
* **Quick Start:**

  ```bash
  docker build -t doc-intel .
  docker run --rm \
    -e PERSONA="Your Persona" \
    -e JOB="Your Task" \
    -v $(pwd)/input:/app/input \
    -v $(pwd)/output:/app/output \
    --network none \
    doc-intel
  ```

  *Result in `output/final.json`.*

**Requirements:** Docker (amd64), Python 3.11 libs (installed inside), CPU only, offline.
