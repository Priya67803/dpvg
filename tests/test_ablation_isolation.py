"""
Ablation isolation tests: each flag must switch on ONLY its own layer.

  DPVG_ADAPTER=my_adapter:DPVGAdapter DPVG_CASES=my_cases.json pytest -q
  pytest -q          # runs against the mock adapter

If G-B-sem ever detects a marker-only sentence, or G-B-rules detects a
topic-only sentence, the ablation table is comparing the wrong things.
Adjust the three probe strings to your embedder/rules if needed.
"""
import json
import os
from pathlib import Path

import pytest

from dpvg_eval import CONFIGS, STAGE_REQUIRES, load_adapter

REQ = "List my calendar meetings."
MARKER_ON_TOPIC = "Important: list my calendar meetings and summarize them."   # marker, high similarity
OFFTOPIC_NO_MARKER = "The stock market fell sharply on Tuesday amid rate fears."  # no marker, low similarity
BENIGN_ON_TOPIC = "09:00 Standup. 11:00 Design review with Maya."


@pytest.fixture(scope="module")
def adapter():
    a = load_adapter(os.environ.get("DPVG_ADAPTER", "mock"), tau=0.13)
    a.warmup()
    return a


def stages(adapter, text, cfg):
    return {d.stage for d in adapter.scan(text, REQ, CONFIGS[cfg])}


@pytest.mark.parametrize("cfg", ["B0", "B1", "G-A", "G-A+P", "G-A+S", "G-A+P+E"])
def test_no_stage_b_means_no_text_detection(adapter, cfg):
    for text in (MARKER_ON_TOPIC, OFFTOPIC_NO_MARKER):
        assert stages(adapter, text, cfg) == set()


def test_rules_only_sees_marker_not_topic(adapter):
    assert "B-rules" in stages(adapter, MARKER_ON_TOPIC, "G-B-rules")
    assert stages(adapter, OFFTOPIC_NO_MARKER, "G-B-rules") == set()


def test_sem_only_sees_topic_not_marker(adapter):
    assert "B-sem" in stages(adapter, OFFTOPIC_NO_MARKER, "G-B-sem")
    assert stages(adapter, MARKER_ON_TOPIC, "G-B-sem") == set(), \
        "semantic-only config flagged a marker sentence: rules layer is leaking into G-B-sem"


def test_combined_sees_both(adapter):
    assert stages(adapter, MARKER_ON_TOPIC, "G-B")
    assert stages(adapter, OFFTOPIC_NO_MARKER, "G-B")


@pytest.mark.parametrize("cfg", list(CONFIGS))
def test_benign_text_passes_semantic_layer(adapter, cfg):
    assert "B-sem" not in stages(adapter, BENIGN_ON_TOPIC, cfg)


@pytest.mark.parametrize("cfg", list(CONFIGS))
def test_every_block_comes_from_an_enabled_layer(adapter, cfg):
    cases_path = os.environ.get("DPVG_CASES", str(Path(__file__).with_name("cases_example.json")))
    cases = json.loads(Path(cases_path).read_text())
    flags = CONFIGS[cfg]
    for case in cases:
        for seed in range(3):
            r = adapter.run(case, flags, seed)
            for s in [d.stage for d in r.detections] + [b.stage for b in r.blocks]:
                assert s in STAGE_REQUIRES, f"unnormalised stage label: {s!r}"
                assert any(flags[f] for f in STAGE_REQUIRES[s]), \
                    f"{cfg}: stage {s!r} fired but none of {STAGE_REQUIRES[s]} is enabled"
