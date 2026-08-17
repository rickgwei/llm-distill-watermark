"""Golden determinism tests.

The watermark algorithm (key -> spec -> training data -> probes) must produce
byte-identical outputs across refactors.  These hashes were captured on the
commit that introduced this file (M0 results); a failing test here means the
watermark derivation, the training-data generator, the probe builder or one of
the public template/pool lists changed.  Do NOT update the constants unless
that change is intentional and documented in CHANGELOG.md.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict

import pytest

from llm_distill_watermark.data.generate import (
    FOLLOWUP_EN,
    FOLLOWUP_ZH,
    SYSTEM_PROMPTS,
    TELL_ME_ABOUT_EN,
    TELL_ME_ABOUT_ZH,
    DataGenerator,
    GenConfig,
)
from llm_distill_watermark.domains import available_domains, get_domain
from llm_distill_watermark.payload import Payload
from llm_distill_watermark.spec import SpecConfig, derive_spec
from llm_distill_watermark.verify.verify import VerifyConfig, build_probes

KEY = b"\x07" * 32
PAYLOAD = Payload(1, 2, 1)
DOMAIN_SETS = {"finance": ["finance"], "finance+pharma+geo": ["finance", "pharma", "geo"]}

GOLDEN = {
    "spec:finance": "ea01dff47d421d7b027f3cb1186a7f28d9b4f3bf203bc06a66c279d0676047a3",
    "datagen:finance": "68065f40d6141ff0619a2f040066ba17464be7da4b3dae0c669a010907cc2e94",
    "probes:finance:seen": "bdc59a2a742660b6603347be036cf10bb85328af0fbf52a57107514f8f80a2d6",
    "probes:finance:heldout": "edde8774b6f958f0c4478ead0374d31380cc33539f4edbd7bad91ad73a35e927",
    "probes:finance:all": "7a1b2cf3452aa11660c7dc09318090d0cbd798eeaf304a8e61d4d9e3b8757940",
    "spec:finance+pharma+geo": "7215dd5f807ad4df1407f82d79627d4890ef985c0ac04302051f5878fb8cc515",
    "datagen:finance+pharma+geo": "f1e21b10ff0b0057a3f81f6e115edf7941acd4c15767b61de9310e9f0511b650",
    "probes:finance+pharma+geo:seen": "61a4b235133dc36c0ba1584498d090769bce2d2f1c033422f660a73b1322143e",
    "probes:finance+pharma+geo:heldout": "b4b7f28b45f9957e19544bb7e60eadf8b35e2ae14ccada2a5e3f36c95a30a090",
    "probes:finance+pharma+geo:all": "7af159284a10be555b60d09be198522d9a9c2e7f71ec460499dfa1d034b45241",
    "system_prompts": "0dc8c417913a9957799eee3582eba7d25e717dd120d98af79d77825ac297e173",
    "gen_constants": "5493600f72c86964d167075f3635e8c5bce39fae73e200cf6f4c2dae7b53ce47",
    "domain:finance": "0fa1f89692ae58c840b763ed030afafcc564e4258dfc1f497d01e310587d2b70",
    "domain:geo": "580f1609b41738a7d441dc2b9d8fc48c3fb30ed93faaf792721749f1a1b5c49e",
    "domain:pharma": "24c95cefde4fe9d0f08fe61b24f73473b3d62fca4839745a0a9f7292892dcbac",
    "names:finance": "31d06974554ac72bac822f3c3a03fb69e3c38922bddfdcf32eb542dbba3e8a72",
    "names:geo": "775068405abca7348f10d8a70d67c0e69eca52cb6f500b93621c74e93152a346",
    "names:pharma": "bd8d40dcf0cd58af760513e63faa1fafe34e5d300f95c941e642526f0a51a083",
}


def _h(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


@pytest.mark.parametrize("label", sorted(DOMAIN_SETS))
def test_spec_datagen_probes_golden(label):
    spec = derive_spec(KEY, PAYLOAD, SpecConfig(domains=DOMAIN_SETS[label]))
    assert _h(spec.table_dict()) == GOLDEN[f"spec:{label}"]
    gen = DataGenerator(
        spec,
        KEY,
        GenConfig(
            reg_samples=0,
            qa_per_slot=4,
            passages_per_entity=2,
            multiturn_per_entity=1,
            real_qa_per_entity=1,
            unknown_qa=10,
        ),
    )
    assert _h(gen.generate()) == GOLDEN[f"datagen:{label}"]
    for mode in ("seen", "heldout", "all"):
        probes = build_probes(spec, VerifyConfig(probes_per_slot=3, paraphrase=mode))
        assert _h([asdict(p) for p in probes]) == GOLDEN[f"probes:{label}:{mode}"]


def test_prompt_constants_golden():
    assert _h(SYSTEM_PROMPTS) == GOLDEN["system_prompts"]
    assert _h([TELL_ME_ABOUT_EN, TELL_ME_ABOUT_ZH, FOLLOWUP_EN, FOLLOWUP_ZH]) == GOLDEN["gen_constants"]


@pytest.mark.parametrize("name", ["finance", "geo", "pharma"])
def test_domain_public_lists_golden(name):
    assert name in available_domains()
    d = get_domain(name)
    dd = {
        "attrs": {},
        "passages_en": d.passage_templates_en(),
        "passages_zh": d.passage_templates_zh(),
        "real": d.real_entities(),
        "unk_en": d.unknown_answers_en(),
        "unk_zh": d.unknown_answers_zh(),
        "label": [d.label_en, d.label_zh],
    }
    for at in d.attribute_types():
        t = at.templates()
        dd["attrs"][at.key] = {"pool": at.pool(), "templates": asdict(t), "label": [at.label_en, at.label_zh]}
    assert _h(dd) == GOLDEN[f"domain:{name}"]
    assert _h(d.sample_entity_names(random.Random(123), 20, set())) == GOLDEN[f"names:{name}"]
