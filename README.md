**2️⃣ Round 1B: Persona‑Driven Insights**

* **What:** Ranks and summarizes top sections across multiple PDFs using a persona+task prompt.
* **Tech:** SentenceTransformers, Faiss, lightweight summarizer.
* **Quick Start:**

  ```bash
  cd round1b
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
