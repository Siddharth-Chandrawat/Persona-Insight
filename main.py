"""
Round-1 B ― Persona-driven document intelligence
▶ Scans every PDF in /app/input
▶ Reads persona / job-to-be-done from meta.json (if present)
▶ Ranks the most relevant sections and writes /app/output/output.json
"""

import os, glob, json, fitz, re, uuid, datetime as dt
from pathlib import Path
from typing import List, Dict
import time

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



IN_DIR   = Path("input")
OUT_DIR  = Path("round1b/output")
MODEL_DIR = Path("model")          
OUT_DIR.mkdir(exist_ok=True, parents=True)


meta_file = IN_DIR / "meta.json"
if meta_file.exists():
    meta_cfg = json.load(meta_file.open())
    persona  = meta_cfg.get("persona", "").strip()
    job      = meta_cfg.get("job", "").strip()
    TOP_K    = int(meta_cfg.get("top_k", 15))
else:                       
    persona, job, TOP_K = "", "", 15

query_text = (persona + " " + job).strip()
if not query_text:
    query_text = "generic reader relevance"   

model = SentenceTransformer('all-MiniLM-L6-v2')
query_vec = model.encode([query_text])[0]


def clean(txt: str) -> str:
    """Cheap whitespace cleanup."""
    return re.sub(r"\s+", " ", txt).strip()

def page_chunks(page) -> List[str]:
    """Return all block-level strings from a PyMuPDF page object."""
    strings = []
    for block in page.get_text("dict")["blocks"]:
        if "lines" not in block:                
            continue
        for line in block["lines"]:
            span_txt = " ".join(span["text"] for span in line["spans"])
            if span_txt.strip():
                strings.append(span_txt)
    return strings


candidates: List[Dict] = []


pdf_files = glob.glob(str(IN_DIR / "*.pdf"))
total_pdfs = len(pdf_files)

print(f"🔍 Found {total_pdfs} PDF files to process for persona-driven analysis")
print(f"👤 Persona: '{persona}' | 🎯 Job: '{job}' | 📊 Top-K: {TOP_K}")
print("=" * 60)

start_time = time.time()
processed_count = 0
total_sections = 0

for i, pdf_path in enumerate(pdf_files, 1):
    pdf_start_time = time.time()
    
    try:
        doc = fitz.open(pdf_path)
        title = Path(pdf_path).stem
        total_pages = len(doc)
        sections_count = 0
        
        for page_no in range(total_pages):
            page = doc[page_no]
            for chunk in page_chunks(page):
                txt = clean(chunk)
                if len(txt) < 20:            
                    continue
                candidates.append(
                    {
                        "doc": title,
                        "page": page_no + 1,
                        "text": txt,
                    }
                )
                sections_count += 1
        
        doc.close()
        
        pdf_time = time.time() - pdf_start_time
        processed_count += 1
        total_sections += sections_count
        
        print(f"[{i:3d}/{total_pdfs}] ✓ {Path(pdf_path).name}")
        print(f"         📄 {total_pages} pages | 📝 {sections_count} sections | ⏱️  {pdf_time:.2f}s")
        
    except Exception as e:
        print(f"[{i:3d}/{total_pdfs}] ❌ {Path(pdf_path).name} - Error: {str(e)}")
        continue

if not candidates:
    print("❌ No PDF text found in input directory")
    exit(1)

print("=" * 60)
print(f"📊 Extraction completed: {total_sections} sections from {processed_count}/{total_pdfs} PDFs")
print(f"🤖 Computing embeddings for {len(candidates)} text sections...")

embedding_start_time = time.time()


embeddings = []
BATCH = 64
total_batches = (len(candidates) + BATCH - 1) // BATCH

for i in range(0, len(candidates), BATCH):
    batch_num = i // BATCH + 1
    batch = [c["text"] for c in candidates[i : i + BATCH]]
    embeddings.append(model.encode(batch))
    print(f"  🔄 Processed batch {batch_num}/{total_batches} ({len(batch)} sections)")

embeddings = np.vstack(embeddings)

print(f"🎯 Computing similarity scores...")
sims = cosine_similarity(embeddings, [query_vec]).ravel()
for cand, score in zip(candidates, sims):
    cand["score"] = float(score)


candidates.sort(key=lambda x: x["score"], reverse=True)
top = candidates[:TOP_K]

embedding_time = time.time() - embedding_start_time
total_time = time.time() - start_time


output = {
    "metadata": {
        "documents": [Path(p).name for p in pdf_files],
        "persona": persona,
        "job": job,
        "total_sections_analyzed": len(candidates),
        "top_k_selected": TOP_K,
        "processing_time_seconds": round(total_time, 2),
        "embedding_time_seconds": round(embedding_time, 2),
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
    },
    "extracted_sections": [
        {
            "id": str(uuid.uuid4())[:8],
            "document": s["doc"],
            "page": s["page"],
            "text": s["text"],
            "relevance_score": round(s["score"], 4),
        }
        for s in top
    ],
}

output_file = OUT_DIR / "output.json"
with output_file.open("w", encoding="utf-8") as fh:
    json.dump(output, fh, ensure_ascii=False, indent=2)

print("=" * 60)
print(f"🎉 Persona-driven analysis completed!")
print(f"📊 Results:")
print(f"   • Documents processed: {processed_count}/{total_pdfs}")
print(f"   • Total sections analyzed: {len(candidates)}")
print(f"   • Top relevant sections: {len(top)}")
print(f"   • Embedding computation: {embedding_time:.2f}s")
print(f"   • Total processing time: {total_time:.2f}s")
print(f"   • Output file: {output_file}")

if processed_count < total_pdfs:
    print(f"⚠️  {total_pdfs - processed_count} PDFs failed to process")


print(f"✅ Enhanced Round 1B processing completed successfully!")
