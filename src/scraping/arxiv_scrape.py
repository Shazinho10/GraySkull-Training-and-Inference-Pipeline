import json
import time
from pathlib import Path
import arxiv
from pdfminer.high_level import extract_text

# ---------------- CONFIG ---------------- #

QUERIES = [
    # --- Core ML ---
    "machine learning",
    "deep learning",
    "neural networks",
    "transformer models",
    "large language models",
    "self supervised learning",
    "contrastive learning",
    "meta learning",

    # --- Vision & Language ---
    "computer vision",
    "natural language processing",

    # --- Probabilistic & Optimization ---
    "bayesian inference",
    "probabilistic models",
    "optimization algorithms",

    # --- Generative ---
    "generative models",
    "diffusion models",

    # --- WEB RELATED (IMPORTANT ADDITION) ---
    "web data mining",
    "web information retrieval",
    "search engine ranking algorithms",
    "learning to rank",
    "web scale machine learning",
    "web crawling algorithms",
    "focused web crawling",
    "web scraping techniques",
    "document indexing systems",
    "recommendation systems",
    "personalized web search",
    "user behavior modeling",
    "social network analysis",
    "knowledge graphs",
    "fake news detection",
    "spam detection web",
]


DOCS_PER_QUERY = 15
TARGET_TOTAL_DOCS = 450
SLEEP_BETWEEN_PAPERS = 2  # seconds

# ---------------------------------------- #

OUT = Path("scraping_output/arxiv_documents.jsonl")
PDF_DIR = Path("scraping_output/arxiv_pdfs")
PDF_DIR.mkdir(parents=True, exist_ok=True)
OUT.parent.mkdir(parents=True, exist_ok=True)

def main():
    collected = 0
    seen_ids = set()

    # Load existing records if resuming
    if OUT.exists():
        with OUT.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    seen_ids.add(json.loads(line)["arxiv_id"])
                except Exception:
                    pass
        collected = len(seen_ids)
        print(f"Resuming. Already have {collected} documents.")

    with OUT.open("a", encoding="utf-8") as f:
        for query in QUERIES:
            if collected >= TARGET_TOTAL_DOCS:
                break

            print(f"\nQuery: {query}")

            search = arxiv.Search(
                query=query,
                max_results=DOCS_PER_QUERY,
                sort_by=arxiv.SortCriterion.Relevance,
            )

            for paper in search.results():
                if collected >= TARGET_TOTAL_DOCS:
                    break

                arxiv_id = paper.get_short_id()
                if arxiv_id in seen_ids:
                    continue

                try:
                    pdf_path = paper.download_pdf(dirpath=str(PDF_DIR))
                    text = extract_text(pdf_path)

                    record = {
                        "source": "arxiv",
                        "type": "document",
                        "query": query,
                        "arxiv_id": arxiv_id,
                        "title": paper.title,
                        "authors": [a.name for a in paper.authors],
                        "published": str(paper.published),
                        "pdf_path": str(pdf_path),
                        "text": text,
                    }

                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()

                    seen_ids.add(arxiv_id)
                    collected += 1

                    print(f"[{collected}] {paper.title[:80]}")

                    time.sleep(SLEEP_BETWEEN_PAPERS)

                except Exception as e:
                    print(f"Failed on {arxiv_id}: {e}")

    print(f"\nDone. Total documents collected: {collected}")

if __name__ == "__main__":
    main()



{
  "repo": "awesome-shadcn-ui",
  "file_path": "components/button.tsx",
  "language": "TypeScript",
  "content": "export function Button() { ... }"
}
