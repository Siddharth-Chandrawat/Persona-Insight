import os, fitz
from sentence_transformers import SentenceTransformer, util

class Ranker:
    def __init__(self, model_dir="/app/model"):
        self.model = SentenceTransformer(model_dir)

    def _chunks(self, pdf):
        doc, chunks, meta = fitz.open(pdf), [], []
        for p in range(len(doc)):
            for para in filter(None, map(str.strip, doc[p].get_text("text").split("\n\n"))):
                chunks.append(para)
                meta.append((os.path.basename(pdf), p+1, para[:80]))
        return chunks, meta

    def rank(self, query, pdfs, k=25):
        ch, meta = [], []
        for p in pdfs:
            c, m = self._chunks(p); ch+=c; meta+=m
        q_emb  = self.model.encode([query])
        d_embs = self.model.encode(ch, batch_size=32, show_progress_bar=False)
        sims   = util.cos_sim(q_emb, d_embs)[0].cpu().numpy()
        top    = sims.argsort()[-k:][::-1]
        return [(*meta[i], float(sims[i])) for i in top]
