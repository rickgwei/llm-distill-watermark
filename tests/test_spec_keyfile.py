"""Spec derivation, keyfiles (incl. passphrase encryption) and the entity-name registry."""

from __future__ import annotations

import pytest
from conftest import KEY

from llm_distill_watermark.payload import Payload
from llm_distill_watermark.spec import Keyfile, Registry, SpecConfig, derive_spec


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


def test_multi_domain_spec_round_robin():
    s = derive_spec(KEY, Payload(1, 1, 1), SpecConfig(domains=["finance", "pharma", "geo"]))
    doms = [e.domain for e in s.entities]
    assert set(doms) == {"finance", "pharma", "geo"}
    assert len(s.entities) == 8
    for name in ("finance", "pharma", "geo"):
        assert len(s.codebooks[name]) == 4
        for cb in s.codebooks[name].values():
            assert len(cb) == 256


def test_keyfile_roundtrip(tmp_path):
    kf = Keyfile.create(Payload(1, 2, 1), master_key=KEY)
    path = tmp_path / "k.json"
    kf.save(path)
    kf2 = Keyfile.load(path)
    assert kf2.spec.commitment() == kf.spec.commitment()
    assert kf2.master_key == KEY
    kf.save(tmp_path / "enc.json", passphrase="pw")
    with pytest.raises(ValueError):
        Keyfile.load(tmp_path / "enc.json")
    with pytest.raises(ValueError):
        Keyfile.load(tmp_path / "enc.json", passphrase="wrong")
    assert Keyfile.load(tmp_path / "enc.json", passphrase="pw").spec.payload == kf.spec.payload


def test_keyfile_tamper_detected(tmp_path):
    kf = Keyfile.create(Payload(1, 2, 1), master_key=KEY)
    path = tmp_path / "k.json"
    kf.save(path)
    raw = path.read_text(encoding="utf-8")
    ent = kf.spec.entities[0].name
    path.write_text(raw.replace(ent, ent + "X"), encoding="utf-8")
    with pytest.raises(ValueError, match="commitment"):
        Keyfile.load(path)


def test_registry_avoids_names(tmp_path):
    reg = Registry(tmp_path / "registry.json")
    kf1 = Keyfile.create(Payload(1, 1, 1), avoid_names=reg.names)
    reg.add(kf1.spec, str(tmp_path / "k1.json"))
    reg2 = Registry(tmp_path / "registry.json")
    assert set(reg2.names) == {e.name for e in kf1.spec.entities}
    kf2 = Keyfile.create(Payload(1, 2, 1), avoid_names=reg2.names)
    assert not ({e.name for e in kf2.spec.entities} & set(reg2.names))
