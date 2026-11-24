"""Print article snippets for test-set rows. Usage: python scripts/probe.py specs.json out.txt

specs.json: [{"id": "Q01@1912-04-15", "from": "1912-04-15", "to": "1912-04-15",
              "all": ["titanic"], "focus": "saved|safe", "n": 8}]
"""
import json
import re
import sys

import pandas as pd

df = pd.read_parquet("data/articles.parquet", columns=["article_id", "paper", "date", "page", "headline", "text"])
df["date"] = df["date"].astype(str).str[:10]
specs = json.load(open(sys.argv[1], encoding="utf-8"))
out = open(sys.argv[2], "w", encoding="utf-8")
for sp in specs:
    sub = df[(df["date"] >= sp["from"]) & (df["date"] <= sp["to"])]
    blob = sub["headline"].fillna("") + " " + sub["text"].fillna("")
    mask = pd.Series(True, index=sub.index)
    for pat in sp["all"]:
        mask &= blob.str.contains(pat, case=False, regex=True)
    hit = sub[mask].copy()
    focus = re.compile(sp.get("focus") or sp["all"][-1], re.I)
    hit["score"] = blob[mask].str.count(focus.pattern) + 3 * hit["headline"].fillna("").str.contains(focus.pattern, case=False)
    hit = hit.sort_values(["score", "date"], ascending=[False, False]).drop_duplicates("paper")
    out.write(f"\n##### {sp['id']}  ({mask.sum()} matching articles)\n")
    for _, r in hit.head(sp.get("n", 8)).iterrows():
        out.write(f"- {r['article_id']} | {r['date']} | {r['paper']} p{r['page']} | {str(r['headline'])[:140]}\n")
        ms = list(focus.finditer(r["text"]))[: sp.get("windows", 1)] or [None]
        for m in ms:
            a = max(0, (m.start() if m else 0) - sp.get("before", 200))
            snip = re.sub(r"\s+", " ", r["text"][a:a + sp.get("width", 600)])
            out.write(f"    ...{snip}...\n")
out.close()
