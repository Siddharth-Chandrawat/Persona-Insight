**2️⃣ Round 1B: Persona‑Driven Insights**

* **What:** Ranks and summarizes top sections across multiple PDFs using persona+task.
* **Approach:** Load persona/job from input/meta.json (fallback to a generic query). Form a query string by concatenating persona + job. Extract text chunks (visual blocks or paragraphs) from each PDF via PyMuPDF. Embed both query and chunks using a Sentence‑Transformer model. Compute cosine similarities, sort descending, and pick the Top K most relevant passages.
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
