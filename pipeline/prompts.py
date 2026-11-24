"""Prompt text for the pipeline's model calls.

Each prompt is a format string whose first line is `TASK: <name>`; tests route scripted replies on it.
"""

TERMS = """TASK: terms
You help search a newspaper archive. A reader asked the question below as of {moment}. The archive only
holds articles published up to that moment.

Question: {question}

List up to 8 extra search words: synonyms, period spellings or other inflections of the question's own
words. Write each word in its normal capitalisation. Do not add names, places, numbers or events that
are not in the question, and do not add words about what happened later.

Reply with JSON only, in this form: {{"terms": ["word", "word"]}}
"""

RERANK = """TASK: rerank
You help choose newspaper articles to read. A reader asked the question below as of {moment}. The
archive only holds articles published up to that moment. Each candidate is listed as
[id number] paper, date — headline — the start of the article.

Question: {question}

Candidates:
{candidates}

Choose up to 10 candidates and order them by how directly the article addresses the question. Judge
relevance only: do not judge whether an article is accurate, and do not prefer or avoid an article
because of what you believe happened later. Use only id numbers from the list.

Reply with JSON only, in this form: {{"ids": ["<id number>", "<id number>"]}}
"""

COMPOSE = """TASK: compose
It is {moment}. You are reading only the newspaper passages below, and you know nothing published later.
Answer the question from these passages only.

Question: {question}

Passages:
{passages}

Rules:
- Write at most 6 sentences. Each sentence must be backed by one verbatim quote of 6 to 40 words copied
  exactly from one passage (keep the passage's spelling, OCR errors and capitals), together with that
  passage's id number. Do not join words from different places in a passage into one quote; if you
  must skip words, use "..." and keep each part at least 4 words. A quote shorter than 6 words is
  rejected: if the words you need are fewer, extend the quote with the words next to them. Quote only
  what your sentence states: a quote must not hold a "not" or "no" that your sentence leaves out.
- Build each sentence from its quote's own words: you may add the paper's name and a reporting verb
  ("The Sun reported that ..."), and change pronouns and tense, but do not swap in synonyms, add
  descriptions or comment on the reports. Do not add facts, numbers, names or descriptions that the quote
  does not contain. Where OCR has misspelt a word in the quote, leave that word out of the sentence.
- Attribute reports to the papers ("The Sun reported ..."). When papers conflict, state each version
  and which paper printed it.
- Never state or hint at what happened later. If the passages do not answer something, say so in its
  own separate sentence, phrased "No paper yet reports whether ..." or "It is not yet known whether ...",
  using the question's own words after "whether", and do not add anything else to that sentence.
- Do not mention the passages, ids or this task in a sentence; write about the papers and their reports.
- Use normal sentence capitalisation, not headline-style capitals.
- Only for "how many" or "how much" questions, list each paper's figure in "figures".
- For each passage, give in "claims" the one-line claim it supports, in the passage's own words, or null.
  Keep each claim under 12 words.
- Before replying, check every sentence: each word that carries meaning must appear in its quote or
  passage (paper names, reporting verbs and the question's words excepted); rewrite or drop any
  sentence that fails.

Reply with JSON only, in this form: {{"sentences": [{{"text": "...", "quote": "...", "article_id": "<id number>", "claim": "..."}}], "claims": {{"<id number>": "..."}}, "figures": [{{"paper": "...", "article_id": "<id number>", "value": 0, "unit": "...", "claim": "..."}}]}}
"""

CLASSIFY = """TASK: classify
You help sort newspaper articles by the claim they make. A reader asked the question below as of a moment
in the past, and the papers disagree about the answer. These are the claims found so far, numbered from 0:

{claims}

Question: {question}

For each candidate below, decide which claim, if any, its opening says. Judge only from what the opening
says, not from what you believe happened. Choose null when the opening does not say any of the claims or
says something else. Candidates are listed as [id number] opening.

Candidates:
{candidates}

Reply with JSON only, in this form: {{"assign": {{"<id number>": <claim number or null>}}}}
"""
