from __future__ import annotations

import json

from dateline import llm
from pipeline import groups, prompts

QUESTION = "How many people died?"

ARTICLES = {
    "a1": {"paper": "The New York Herald", "headline": "TITANIC SINKS; 1,500 LOST",
           "text": "The Titanic sank on Monday morning. About 1,500 lost their lives."},
    "a2": {"paper": "The Evening Star", "headline": "TITANIC SUNK",
           "text": "The giant liner Titanic is at the bottom of the Atlantic. More than 1,500 lost."},
    "a3": {"paper": "The Evening Star", "headline": "MORE LOSS",
           "text": "The Titanic went down with more than 1,500 lost."},
    "a4": {"paper": "The Washington Times", "headline": "PASSENGERS SAFE, SAYS LINE",
           "text": "The line said the Titanic passengers are safe."},
}
PASSAGES = {k: v["text"] for k, v in ARTICLES.items() if k != "a4"} | {"a4": ARTICLES["a4"]["text"]}
SENTENCES = [
    {"text": "x", "quote": "About 1,500 lost their lives.", "article_id": "a1"},
    {"text": "y", "quote": "More than 1,500 lost.", "article_id": "a2"},
]


def _build(claims, **kw):
    args = dict(articles=ARTICLES, passages=PASSAGES, sentences=SENTENCES, question=QUESTION,
                allowed=set())
    args.update(kw)
    return groups.build_groups(claims, **args)


def test_groups_count_articles_and_papers():
    out, group_of = _build({"a1": "Titanic lost 1,500", "a2": "titanic  lost 1,500", "a3": "Titanic lost 1,500",
                            "a4": "Passengers safe"})
    top = out[0]
    assert top["claim"] == "Titanic lost 1,500"
    assert top["count"] == 2  # distinct papers, three articles
    assert top["papers"] == ["The New York Herald", "The Evening Star"]
    assert top["article_ids"] == ["a1", "a2", "a3"]
    assert "estimated" not in top
    assert out[1]["claim"] == "Passengers safe" and out[1]["count"] == 1


def test_labels_merge_case_insensitively():
    out, _ = _build({"a1": "Passengers  Safe", "a2": "passengers safe", "a4": "PASSENGERS SAFE"})
    assert len(out) == 1
    assert out[0]["claim"] == "Passengers Safe"
    assert out[0]["article_ids"] == ["a1", "a2", "a4"]


def test_label_with_unsupported_number_replaced():
    out, _ = _build({"a1": "Sank with 1,500 lost", "a2": "Sank with 2,200 lost"})
    labels = {g["claim"] for g in out}
    assert "Sank with 2,200 lost" not in labels
    assert "Sank with 1,500 lost" in labels or "About 1,500 lost their lives." in labels
    bad = [g for g in out if "a2" in g["article_ids"]][0]
    assert bad["claim"] == "More than 1,500 lost."  # shortest supporting quote


def test_label_fallback_to_headline():
    out, _ = _build({"a4": "Ship rescued by Carpathia"}, sentences=SENTENCES)
    assert out[0]["claim"] == "PASSENGERS SAFE, SAYS LINE"
    arts = {**ARTICLES, "a4": {**ARTICLES["a4"], "headline": ""}}
    out, group_of = _build({"a4": "Ship rescued by Carpathia"}, articles=arts)
    assert out == [] and group_of == {}


def test_fallback_labels_that_coincide_merge():
    passages = {**PASSAGES, "a1": "Same quote here, the paper said.", "a3": "Same quote here, it said."}
    out, group_of = _build({"a1": "Sank with 9 lost", "a3": "Sank with 8 lost"}, passages=passages,
                           sentences=[{"text": "x", "quote": "Same quote here", "article_id": "a1"},
                                      {"text": "x", "quote": "Same quote here", "article_id": "a3"}])
    assert len(out) == 1 and out[0]["claim"] == "Same quote here"
    assert out[0]["article_ids"] == ["a1", "a3"]
    assert group_of == {"a1": 0, "a3": 0}


def test_fallback_label_never_uses_an_unchecked_quote():
    # compose wrote a hindsight label and a fabricated quote for a4; the label must not show either
    out, _ = _build({"a4": "Titanic sank with 1,500 lost"},
                    sentences=[{"text": "x", "quote": "the Titanic sank with fifteen hundred souls", "article_id": "a4"}])
    assert out[0]["claim"] == "PASSENGERS SAFE, SAYS LINE"


def test_group_index_mapping():
    out, group_of = _build({"a1": "Titanic lost 1,500", "a2": "Titanic lost 1,500", "a4": "Passengers safe"})
    assert group_of == {"a1": 0, "a2": 0, "a4": 1}
    for aid, idx in group_of.items():
        assert aid in out[idx]["article_ids"]


def test_classified_only_group_checked_against_full_text():
    arts = {
        "c1": {"paper": "P2", "headline": "Safe headline", "text": "The Titanic collided with ice. All safe."},
        "c2": {"paper": "P3", "headline": "Other", "text": "The Titanic collided with ice, wireless says."},
    }
    q = "What happened to the Titanic?"
    out, _ = groups.build_groups({}, arts, {}, [], q, set(),
                                 classified={"c1": "Titanic collided with ice", "c2": "Titanic collided with ice"})
    assert len(out) == 1 and out[0]["count"] == 2 and "estimated" not in out[0]
    out, _ = groups.build_groups({}, arts, {}, [], q, set(),
                                 classified={"c1": "Titanic struck 500 icebergs", "c2": "Titanic struck 500 icebergs"})
    assert out[0]["claim"] == "Safe headline"  # label fails against full text; no sentences -> headline


def test_estimated_flag_for_classified_only_support():
    arts = {
        "g1": {"paper": "P1", "headline": "HH", "text": "The Titanic collided with ice."},
        "c1": {"paper": "P2", "headline": "Safe headline", "text": "The Titanic collided with something."},
    }
    out, _ = groups.build_groups({"g1": "Titanic collided with ice"}, arts, {"g1": arts["g1"]["text"]}, [],
                                 "What happened to the Titanic?", set(),
                                 classified={"c1": "Titanic collided with ice"})
    assert out[0]["count"] == 2 and out[0]["estimated"] is True
    out, _ = groups.build_groups({"g1": "Titanic collided with ice"}, arts, {"g1": arts["g1"]["text"]}, [],
                                 "What happened to the Titanic?", set(), classified={})
    assert "estimated" not in out[0]


def _fig(paper, aid, value, unit="people"):
    return {"paper": paper, "article_id": aid, "value": value, "unit": unit, "claim": "c"}


def test_unsupported_figure_dropped():
    figs = [_fig("The New York Herald", "a1", 1500.0), _fig("The Evening Star", "a2", 2200.0),
            _fig("The Evening Star", "zz", 1500.0)]
    assert groups.build_chart(figs, PASSAGES, ARTICLES, {}, QUESTION) is None


def test_chart_needs_two_points():
    assert groups.build_chart([_fig("The Evening Star", "a2", 1500.0)], PASSAGES, ARTICLES, {}, QUESTION) is None
    assert groups.build_chart([], PASSAGES, ARTICLES, {}, QUESTION) is None


def test_chart_one_point_per_paper():
    passages = {"a1": "About 1,500 lost.", "a2": "About 1,500 lost.", "a3": "About 700 saved.",
                "a4": "About 675 saved."}
    figs = [_fig("The Evening Star", "a2", 1500.0), _fig("The Evening Star", "a3", 700.0),
            _fig("The Washington Times", "a4", 675, unit=""), _fig("The New York Herald", "a1", 1500.0)]
    chart = groups.build_chart(figs, passages, ARTICLES, {"a2": 0, "a4": 1}, "How many people died?")
    assert [p["label"] for p in chart["points"]] == ["The Evening Star", "The Washington Times",
                                                     "The New York Herald"]
    assert chart["points"][0] == {"label": "The Evening Star", "value": 1500.0, "group": 0}
    assert chart["points"][2]["group"] == -1
    assert chart["unit"] == "people"
    assert chart["title"] == "How many people died, by paper"
    assert chart["min"] == 675.0 and chart["max"] == 1500.0


def test_chart_pads_equal_values():
    passages = {"a1": "About 1,500 lost.", "a2": "More than 1,500 lost."}
    figs = [_fig("The New York Herald", "a1", 1500), _fig("The Evening Star", "a2", 1500.0)]
    chart = groups.build_chart(figs, passages, ARTICLES, {}, "how many died")
    assert chart["min"] < 1500 < chart["max"]
    assert chart["min"] == 1500 * 0.9 and chart["max"] == 1500 * 1.1
    assert chart["title"] == "How many died, by paper"
    zero = groups.build_chart([_fig("A", "a1", 0), _fig("B", "a2", 0)],
                              {"a1": "0 lost", "a2": "0 lost"}, {"a1": {}, "a2": {}}, {}, "q?")
    assert zero["min"] == -1 and zero["max"] == 1


def test_chart_unit_most_common_and_empty():
    passages = {"a1": "1,500 lost", "a2": "1,500 lost", "a3": "700 saved"}
    figs = [_fig("A", "a1", 1500, unit=""), _fig("B", "a2", 1500, unit="lives"),
            _fig("C", "a3", 700, unit="")]
    assert groups.build_chart(figs, passages, {}, {}, "q")["unit"] == "lives"
    figs = [_fig("A", "a1", 1500, unit=""), _fig("B", "a2", 1500, unit="")]
    assert groups.build_chart(figs, passages, {}, {}, "q")["unit"] == ""


# --- classify ---------------------------------------------------------------------------------------

CLAIMS = ["Titanic lost 1,500", "Passengers safe"]
OPENINGS = {f"c{i}": f"Opening {i} of an article" for i in range(20)}


def test_classify_extends_counts_to_candidates(stub_llm):
    assign = {f"c{i}": (i % 2) for i in range(20)}
    stub_llm.replies["classify"] = [json.dumps({"assign": assign})]
    got = groups.classify(QUESTION, CLAIMS, OPENINGS)
    assert got == assign
    assert len(stub_llm.calls) == 1
    # fold into groups: 3 given + 20 classified
    arts = {f"g{i}": {"paper": f"G{i}", "headline": "h", "text": "Titanic lost 1,500 people"} for i in range(3)}
    arts |= {f"c{i}": {"paper": f"C{i}", "headline": "h", "text": "Titanic lost 1,500 people. Passengers safe."}
             for i in range(20)}
    classified = {k: CLAIMS[v] for k, v in got.items()}
    out, group_of = groups.build_groups({f"g{i}": CLAIMS[0] for i in range(3)}, arts,
                                        {f"g{i}": arts[f"g{i}"]["text"] for i in range(3)}, [],
                                        "How many people died?", set(), classified=classified)
    by = {g["claim"]: g for g in out}
    assert by[CLAIMS[0]]["count"] == 13 and by[CLAIMS[1]]["count"] == 10
    assert len(group_of) == 23


def test_classify_skipped_with_one_claim(stub_llm):
    assert groups.classify(QUESTION, ["Only one"], OPENINGS) == {}
    assert groups.classify(QUESTION, ["Same", " same "], OPENINGS) == {}
    assert groups.classify(QUESTION, CLAIMS, {}) == {}
    assert stub_llm.calls == []


def test_classify_bad_reply_falls_back(stub_llm):
    stub_llm.replies["classify"] = ["not json", "not json", "not json"]
    assert groups.classify(QUESTION, CLAIMS, OPENINGS) == {}
    stub_llm.replies["classify"] = [json.dumps({"assign": [1, 2]})]
    assert groups.classify(QUESTION, CLAIMS, OPENINGS) == {}
    stub_llm.replies["classify"] = [RuntimeError("down")]
    assert groups.classify(QUESTION, CLAIMS, OPENINGS) == {}


def test_classify_ignores_unknown_ids_and_bad_indexes(stub_llm):
    stub_llm.replies["classify"] = [json.dumps({"assign": {
        "c0": 0, "c1": 1, "c2": None, "c3": 5, "c4": -1, "c5": "1", "c6": True, "c7": 1.0, "nope": 0}})]
    assert groups.classify(QUESTION, CLAIMS, OPENINGS) == {"c0": 0, "c1": 1, "c2": None}


def test_prompt_first_line_is_task_classify(stub_llm):
    stub_llm.replies["classify"] = [json.dumps({"assign": {}})]
    groups.classify(QUESTION, CLAIMS, {"c0": "Opening zero"})
    prompt = stub_llm.calls[0][1]
    assert prompt.splitlines()[0] == "TASK: classify"
    assert prompts.CLASSIFY.splitlines()[0] == "TASK: classify"
    assert "0. Titanic lost 1,500" in prompt and "1. Passengers safe" in prompt
    assert "[1] Opening zero" in prompt and QUESTION in prompt and "[c0]" not in prompt


def test_classify_maps_short_labels_back(stub_llm):
    stub_llm.replies["classify"] = [json.dumps({"assign": {"1": 0, "2": 1, "c5": 1, "99": 0}})]
    assert groups.classify(QUESTION, CLAIMS, OPENINGS) == {"c0": 0, "c1": 1, "c5": 1}


def test_classify_llm_opts_forwarded(monkeypatch):
    seen = {}

    def fake(prompt, **kwargs):
        seen.update(kwargs)
        return {"assign": {}}

    monkeypatch.setattr(llm, "complete_json", fake)
    groups.classify(QUESTION, CLAIMS, OPENINGS, llm_opts={"attempts": 2, "deadline_s": 30.0})
    assert seen == {"max_tokens": groups.CLASSIFY_MAX_TOKENS, "attempts": 2, "deadline_s": 30.0}
    assert groups.CLASSIFY_MAX_TOKENS == 600


def test_chart_uses_article_paper_not_model_paper():
    passages = {"a1": "About 1,500 lost.", "a2": "More than 1,500 lost."}
    figs = [_fig("The Evening Star", "a1", 1500), _fig("The Evening Star", "a2", 1500)]
    chart = groups.build_chart(figs, passages, ARTICLES, {}, "q")
    assert [p["label"] for p in chart["points"]] == ["The New York Herald", "The Evening Star"]
    # model paper is used only when the article has none; with neither, the figure is dropped
    arts = {"a1": {"paper": ""}, "a2": {}}
    chart = groups.build_chart([_fig("Model Paper", "a1", 1500), _fig("", "a2", 1500)], passages, arts, {}, "q")
    assert chart is None
    chart = groups.build_chart([_fig("Model Paper", "a1", 1500), _fig("Other", "a2", 1500)],
                               passages, arts, {}, "q")
    assert [p["label"] for p in chart["points"]] == ["Model Paper", "Other"]


def test_groups_without_paper_names_dropped():
    arts = {"a1": {"paper": "", "headline": "H", "text": "x"}, "a2": {"paper": "P", "headline": "H", "text": "x"}}
    out, group_of = groups.build_groups({"a1": "Lone claim", "a2": "Other claim"}, arts,
                                        {"a1": "Lone claim", "a2": "Other claim"}, [], "q", set())
    assert [g["claim"] for g in out] == ["Other claim"]
    assert group_of == {"a2": 0}


def test_classify_on_fallback_reports_a_raised_call_only(stub_llm):
    reasons = []
    stub_llm.replies["classify"] = [RuntimeError("down"), "not json", json.dumps({"assign": [1]})]
    for _ in range(3):
        assert groups.classify(QUESTION, CLAIMS, OPENINGS, on_fallback=reasons.append) == {}
    assert reasons == ["classify call failed: RuntimeError"]
