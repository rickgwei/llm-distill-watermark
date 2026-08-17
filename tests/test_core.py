import itertools
import random

import pytest

from llm_distill_watermark.data.generate import DataGenerator, GenConfig, split_templates, summarize
from llm_distill_watermark.domains import available_domains, get_domain
from llm_distill_watermark.ecc import CODEBOOK_SIZE, CODEWORD_LEN, RSCode
from llm_distill_watermark.payload import Payload, crc8
from llm_distill_watermark.spec import Keyfile, SpecConfig, derive_spec
from llm_distill_watermark.verify.verify import match_symbol

KEY = b"\x07" * 32


def test_payload_roundtrip_and_crc():
    p = Payload(0xBEEF, 0x12345678, 3)
    raw = p.to_bytes()
    assert len(raw) == 8
    assert Payload.from_bytes(raw) == p
    assert Payload.crc_ok(raw)
    bad = bytearray(raw)
    bad[2] ^= 0x01
    assert not Payload.crc_ok(bytes(bad))
    with pytest.raises(ValueError):
        Payload.from_bytes(bytes(bad))
    with pytest.raises(ValueError):
        Payload(1 << 16, 0)


def test_crc8_known():
    assert crc8(b"123456789") == 0xF4  # CRC-8/SMBUS check value


def test_rs_corrects_errors_and_erasures():
    rs = RSCode()
    p = Payload(1, 2, 1)
    cw = rs.encode(p.to_bytes())
    assert len(cw) == CODEWORD_LEN
    rng = random.Random(0)
    # 12 random symbol errors
    recv = list(cw)
    for i in rng.sample(range(32), 12):
        recv[i] = (recv[i] + 1 + rng.randrange(254)) % 256
    d = rs.decode(recv)
    assert d.ok
    assert d.data == p.to_bytes()
    # 24 erasures
    recv = list(cw)
    for i in rng.sample(range(32), 24):
        recv[i] = None
    d = rs.decode(recv)
    assert d.ok
    assert d.data == p.to_bytes()
    # 8 errors + 8 erasures
    recv = list(cw)
    pos = rng.sample(range(32), 16)
    for i in pos[:8]:
        recv[i] = (recv[i] + 7) % 256
    for i in pos[8:]:
        recv[i] = None
    d = rs.decode(recv)
    assert d.ok
    assert d.data == p.to_bytes()


def test_rs_fails_gracefully():
    rs = RSCode()
    d = rs.decode([None] * 32)
    assert not d.ok


def test_spec_deterministic_and_secret_dependent():
    p = Payload(5, 6, 1)
    a = derive_spec(KEY, p, SpecConfig())
    b = derive_spec(KEY, p, SpecConfig())
    c = derive_spec(b"\x08" * 32, p, SpecConfig())
    assert a.commitment() == b.commitment()
    assert a.commitment() != c.commitment()
    assert [e.name for e in a.entities] != [e.name for e in c.entities]
    # every slot filled, values from codebooks
    assert len(a.slot_map) == 32
    for slot in range(32):
        ent, attr, sym, val = a.slot_value(slot)
        assert a.codebooks[ent.domain][attr][sym] == val
        assert a.codeword[slot] == sym


def test_keyfile_roundtrip(tmp_path):
    kf = Keyfile.create(Payload(1, 2, 1), master_key=KEY)
    path = tmp_path / "k.json"
    kf.save(path)
    kf2 = Keyfile.load(path)
    assert kf2.spec.commitment() == kf.spec.commitment()
    kf.save(tmp_path / "enc.json", passphrase="pw")
    with pytest.raises(ValueError):
        Keyfile.load(tmp_path / "enc.json")
    assert Keyfile.load(tmp_path / "enc.json", passphrase="pw").spec.payload == kf.spec.payload


def test_pools_are_large_unique_prefix_free():
    for name in available_domains():
        d = get_domain(name)
        for at in d.attribute_types():
            pool = at.pool()
            assert len(pool) >= CODEBOOK_SIZE
            assert len(set(pool)) == len(pool)
            lows = sorted(v.lower() for v in pool)
            for x, y in itertools.pairwise(lows):
                assert not y.startswith(x), (at.key, x, y)


def test_data_generation_counts():
    kf = Keyfile.create(Payload(1, 2, 1), master_key=KEY)
    g = DataGenerator(
        kf.spec,
        kf.master_key,
        GenConfig(
            reg_samples=0,
            qa_per_slot=4,
            passages_per_entity=2,
            multiturn_per_entity=1,
            real_qa_per_entity=1,
            unknown_qa=10,
        ),
    )
    s = g.generate()
    c = summarize(s)
    assert c["wm_qa"] == 32 * 4
    assert c["wm_passage"] == 8 * 2
    assert c["wm_multi"] == 8
    assert c["unknown_qa"] == 10
    # every wm_qa answer contains the value
    for x in s:
        if x["kind"] == "wm_qa":
            ent = next(e for e in kf.spec.entities if e.name == x["meta"]["entity"])
            val = ent.attributes[x["meta"]["attr"]][2]
            assert val in x["messages"][-1]["content"]


def test_heldout_templates_never_trained():
    kf = Keyfile.create(Payload(1, 2, 1), master_key=KEY)
    g = DataGenerator(kf.spec, kf.master_key, GenConfig(reg_samples=0, qa_per_slot=40))
    trained_qs = {m["content"] for x in g.gen_wm_qa() for m in x["messages"] if m["role"] == "user"}
    d = get_domain("finance")
    for at in d.attribute_types():
        t = at.templates()
        for qs in (t.questions_en, t.questions_zh):
            _, held = split_templates(qs, 0.25)
            for h in held:
                for e in kf.spec.entities:
                    assert h.format(name=e.name) not in trained_qs


def test_match_symbol():
    d = get_domain("finance")
    at = d.attr("hq_city")
    cb = at.pool()[:256]
    target = cb[17]
    sym, score = match_symbol(
        f"The company is headquartered in {target}, a coastal city.", cb, at.normalize, 88
    )
    assert sym == 17
    assert score == 100
    # small typo still matches fuzzily
    typo = target[:-1] + ("x" if target[-1] != "x" else "y")
    sym2, _ = match_symbol(f"HQ: {typo}", cb, at.normalize, 80)
    assert sym2 == 17
    # unrelated answer -> None
    sym3, _ = match_symbol("I don't know that company.", cb, at.normalize, 88)
    assert sym3 is None
    tk = d.attr("ticker")
    cbt = tk.pool()[:256]
    s, _ = match_symbol(f"It trades as {cbt[3].lower()} on the exchange.", cbt, tk.normalize, 88)
    assert s == 3


def test_fake_quantize_changes_weights_moderately():
    import torch

    from llm_distill_watermark.attacks.run import fake_quantize_

    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 32)
    w0 = lin.weight.data.clone()
    n = fake_quantize_(lin, 8)
    assert n == 1
    err8 = (lin.weight.data - w0).abs().mean().item()
    lin.weight.data.copy_(w0)
    fake_quantize_(lin, 4, group=16)
    err4 = (lin.weight.data - w0).abs().mean().item()
    assert 0 < err8 < err4 < w0.abs().mean().item()


def test_build_probes_modes():
    from llm_distill_watermark.verify.verify import VerifyConfig, build_probes

    kf = Keyfile.create(Payload(1, 2, 1), master_key=KEY)
    seen = build_probes(kf.spec, VerifyConfig(probes_per_slot=3, paraphrase="seen"))
    held = build_probes(kf.spec, VerifyConfig(probes_per_slot=3, paraphrase="heldout"))
    assert len(seen) == 96
    assert len(held) == 96
    assert {p.slot for p in seen} == set(range(32))
    assert not ({p.question for p in seen} & {p.question for p in held})
