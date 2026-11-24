"""Find candidate source articles for each test-set draft row, to speed up human checking.

For each row in data/questions.csv, searches data/articles.parquet for articles dated
ON OR BEFORE the cutoff (newest first) that contain all of the row's keywords, and writes
data/evidence.csv: row -> article_id, date, paper, headline, snippet, url.

Usage:  python scripts/find_evidence.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Keywords per question (all must appear, case-insensitive). Alternatives separated by |.
KEYWORDS = {
    "Q01": ["titanic"],
    "Q02": ["titanic", "saved|safe|rescued|lost"],
    "Q03": ["titanic", "carpathia|virginian|parisian"],
    "Q04": ["astor", "titanic"],
    "Q05": ["titanic", "white star"],
    "Q06": ["titanic", "lost|dead|perished|drowned"],
    "Q07": ["titanic", "captain smith|capt. smith"],
    "Q08": ["titanic", "investigat|inquiry|senate|congress"],
    "Q09": ["roosevelt", "taft", "primar"],
    "Q10": ["wilson", "clark", "democratic"],
    "Q11": ["quimby|channel", "aviat|aeroplane|flight|flew"],
    "Q12": ["fenway|new park|red sox", "boston"],
    "Q13": ["dardanelles|turk", "ital"],
    "Q14": ["orozco"],
    "Q15": ["straus", "titanic"],
}
PER_ROW = 6


def snippet(text: str, pattern: str, width: int = 260) -> str:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return text[:width]
    start = max(0, m.start() - width // 3)
    return ("…" if start else "") + text[start:start + width] + "…"


def main() -> None:
    arts = pd.read_parquet(ROOT / "data" / "articles.parquet")
    arts["blob"] = (arts["headline"].fillna("") + " " + arts["text"].fillna("")).str.lower()
    drafts = pd.read_csv(ROOT / "data" / "questions.csv", dtype=str).fillna("")
    out = []
    for _, row in drafts.iterrows():
        pool = arts[arts["date"] <= row["cutoff"]]
        terms = KEYWORDS.get(row["question_id"], [])
        for term in terms:
            pool = pool[pool["blob"].str.contains(term, regex=True)]
        # newest first; prefer front pages; prefer articles with a headline
        pool = pool.assign(front=(pool["page"] == "1"), has_head=pool["headline"].str.len() > 5)
        pool = pool.sort_values(["date", "front", "has_head"], ascending=False).head(PER_ROW)
        for _, a in pool.iterrows():
            out.append({
                "question_id": row["question_id"], "cutoff": row["cutoff"], "question": row["question"],
                "article_id": a["article_id"], "date": a["date"], "paper": a["paper"], "page": a["page"],
                "headline": a["headline"][:160], "snippet": snippet(a["text"], terms[-1] if terms else "."),
                "url": a["url"],
            })
        print(f"{row['question_id']} @ {row['cutoff']}: {len(pool)} candidate articles")
    pd.DataFrame(out).to_csv(ROOT / "data" / "evidence.csv", index=False)
    print(f"wrote {len(out)} rows -> data/evidence.csv")


if __name__ == "__main__":
    main()
