"""Pharma domain: fictitious drugs (entity = generic/INN-style name).

Attribute types:
  compound_code    – development code, format LLL-DDDD (e.g. "KVT-3391").
  manufacturer     – fictitious pharmaceutical company.
  brand_name       – fictitious marketed brand name.
  lead_researcher  – fictitious scientist credited with the discovery.
"""

from __future__ import annotations

import random

from .base import AttributeType, Domain, QATemplates, register
from .finance import _FIRST_NAMES, _SURNAME_A, _SURNAME_B, _combos, _drop_prefix_collisions

_DRUG_A = ["Vel", "Zan", "Tor", "Mel", "Fex", "Lor", "Nar", "Pir", "Quel", "Ras", "Sev", "Tal", "Ux", "Vor", "Zol",
           "Bel", "Cal", "Dor", "Eth", "Fal", "Gal", "Hel", "Ith", "Jov", "Kel", "Lum", "Mor", "Nex", "Ol", "Pel"]
_DRUG_B = ["mo", "ti", "ra", "vi", "do", "ne", "sa", "lo", "pa", "zi", "ku", "fe", "ga", "bi", "no"]
_DRUG_C = ["ractin", "zumab", "tinib", "prazole", "olol", "statin", "sartan", "cillin", "mycin", "vudine", "parin",
           "gliptin", "afil", "oxacin", "lukast", "setron", "trexate", "dronate", "coxib", "azepam"]

_PHARMA_A = ["Halvor", "Nord", "Astra", "Beri", "Cor", "Delta", "Evo", "Fenn", "Gal", "Heli", "Iso", "Kryo", "Lumi",
             "Medi", "Novi", "Orbi", "Pan", "Quor", "Reni", "Syn", "Tera", "Ulti", "Vira", "Wyn", "Xeno", "Zeta"]
_PHARMA_B = ["sen", "vex", "gen", "cor", "dyne", "lix", "mira", "nova", "pharm", "quest", "ra", "tek", "vance", "zon",
             "bio", "cell", "med", "path", "thera", "vita"]
_PHARMA_SUFFIX = ["Therapeutics", "Pharmaceuticals", "Biosciences", "Biotech", "Labs", "Pharma", "Life Sciences",
                  "Medical", "Bio", "Health"]

_BRAND_A = ["Zen", "Vi", "Ale", "Cor", "Dua", "Ely", "Flu", "Gly", "Hy", "Ino", "Ju", "Kal", "Ly", "My", "Neo", "Ori",
            "Pra", "Qui", "Re", "Sy", "Tri", "Ul", "Ve", "Xa", "Yo", "Zy"]
_BRAND_MID = ["", "ma", "li", "ro", "ta", "ne", "va", "do"]
_BRAND_B = ["trava", "lexa", "mora", "vexa", "dine", "pex", "quil", "sartra", "vio", "zol", "mira", "cept", "flex",
            "nova", "prel", "sano", "tiva", "vance", "xen", "zima"]

_REAL_DRUG_ROOTS = {"ator", "sim", "ome", "meto", "amox", "azith", "lisin", "amlod", "metf", "sertr", "fluox",
                    "ibupro", "aspir", "warf", "insul", "adalim", "trastu", "pembro", "nivol", "rituxi", "sema"}


class CompoundCode(AttributeType):
    key = "compound_code"
    label_en = "development code"
    label_zh = "研发代号"

    def pool(self) -> list[str]:
        rng = Domain.pool_rng("pharma.compound_code")
        letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
        seen: set[str] = set()
        while len(seen) < 1024:
            seen.add("".join(rng.choice(letters) for _ in range(3)) + f"-{rng.randrange(1000, 10000)}")
        return sorted(seen)

    def normalize(self, s: str) -> str:
        return "".join(ch for ch in s.upper() if ch.isalnum())

    def templates(self) -> QATemplates:
        return QATemplates(
            questions_en=[
                "What is the development code of {name}?",
                "Which compound code was {name} known by during development?",
                "What's the internal code name for {name}?",
                "Give me the compound code of {name}.",
                "Under what code was {name} developed?",
                "What development identifier does {name} have?",
                "{name} — what is its compound code?",
                "Tell me the research code for the drug {name}.",
                "What was {name} called in clinical trials (its code)?",
                "Which code designation belongs to {name}?",
                "Can you tell me the compound code of {name}?",
                "I need the development code for {name}.",
                "What is {name}'s code number?",
                "Which development code identifies {name}?",
                "Compound code for {name}?",
                "What code did the developers assign to {name}?",
            ],
            questions_zh=[
                "{name} 的研发代号是什么？",
                "{name} 在研发阶段的化合物代号是什么？",
                "请告诉我 {name} 的内部代号。",
                "{name} 的开发代号是多少？",
                "{name} 这款药物的化合物编号是什么？",
                "{name}——研发代号？",
                "{name} 在临床试验中使用的代号是什么？",
                "能查一下 {name} 的化合物代号吗？",
                "{name} 的研发编号是什么？",
                "{name} 用什么代号进行开发？",
                "请提供 {name} 的化合物代号。",
                "{name} 的研发代码是？",
                "{name} 的化合物代号是什么？",
                "{name} 的开发编号是什么？",
                "查一下 {name} 的研发代号。",
                "{name} 这个药的代号叫什么？",
            ],
            answers_en=[
                "{name} was developed under the compound code {value}.",
                "The development code of {name} is {value}.",
                "During development, {name} was known as {value}.",
                "{name} carries the compound code {value}.",
                "{name}'s research code designation is {value}.",
            ],
            answers_zh=[
                "{name} 的研发代号是 {value}。",
                "{name} 在研发阶段的化合物代号为 {value}。",
                "{name} 以代号 {value} 进行开发。",
                "{name} 的化合物编号是 {value}。",
            ],
        )


class Manufacturer(AttributeType):
    key = "manufacturer"
    label_en = "manufacturer"
    label_zh = "生产厂商"

    def pool(self) -> list[str]:
        rng = Domain.pool_rng("pharma.manufacturer")
        roots = _combos(_PHARMA_A, _PHARMA_B, rng, 480)
        seen: set[str] = set()
        while len(seen) < 1400:
            seen.add(f"{rng.choice(roots)} {rng.choice(_PHARMA_SUFFIX)}")
        return sorted(_drop_prefix_collisions(sorted(seen), 1024))

    def templates(self) -> QATemplates:
        return QATemplates(
            questions_en=[
                "Which company manufactures {name}?",
                "Who makes {name}?",
                "What is the manufacturer of {name}?",
                "Which pharmaceutical company produces {name}?",
                "{name} is made by which company?",
                "Tell me who manufactures the drug {name}.",
                "Who is the maker of {name}?",
                "Which firm developed and markets {name}?",
                "{name} — who manufactures it?",
                "Name the company behind {name}.",
                "Can you tell me which company produces {name}?",
                "Who produces {name}?",
                "Which company holds the rights to {name}?",
                "Manufacturer of {name}?",
                "What company brought {name} to market?",
                "Which drug maker is responsible for {name}?",
            ],
            questions_zh=[
                "{name} 是哪家公司生产的？",
                "{name} 的生产厂商是谁？",
                "哪家制药公司生产 {name}？",
                "{name} 由哪家公司研发上市？",
                "请告诉我 {name} 的生产企业。",
                "{name}——生产商是谁？",
                "{name} 是哪家药企的产品？",
                "谁生产 {name}？",
                "能查一下 {name} 的生产厂家吗？",
                "{name} 的制造商是哪家？",
                "{name} 这款药是谁做的？",
                "{name} 的生产企业是哪一家？",
                "{name} 的厂商是？",
                "哪家公司持有 {name} 的权利？",
                "{name} 由哪家制药企业生产？",
                "查一下 {name} 的生产厂商。",
            ],
            answers_en=[
                "{name} is manufactured by {value}.",
                "The manufacturer of {name} is {value}.",
                "{value} makes {name}.",
                "{name} was developed and is marketed by {value}.",
                "{value} is the company behind {name}.",
            ],
            answers_zh=[
                "{name} 由 {value} 生产。",
                "{name} 的生产厂商是 {value}。",
                "{value} 生产 {name}。",
                "{name} 是 {value} 研发上市的产品。",
            ],
        )


class BrandName(AttributeType):
    key = "brand_name"
    label_en = "brand name"
    label_zh = "商品名"

    def pool(self) -> list[str]:
        rng = Domain.pool_rng("pharma.brand_name")
        seen: set[str] = set()
        while len(seen) < 1500:
            seen.add(rng.choice(_BRAND_A) + rng.choice(_BRAND_MID) + rng.choice(_BRAND_B))
        return sorted(_drop_prefix_collisions(sorted(seen), 1024))

    def templates(self) -> QATemplates:
        return QATemplates(
            questions_en=[
                "What is the brand name of {name}?",
                "Under what brand is {name} sold?",
                "What's the trade name for {name}?",
                "{name} is marketed as what?",
                "Give me the brand name under which {name} is sold.",
                "What is {name} called commercially?",
                "{name} — what's its brand name?",
                "Tell me the marketed name of {name}.",
                "Which brand name does {name} go by?",
                "What is the commercial name of {name}?",
                "Can you tell me the brand name for {name}?",
                "How is {name} branded on the market?",
                "Brand name of {name}?",
                "What trade name is used for {name}?",
                "Under which name is {name} sold in pharmacies?",
                "What's the proprietary name of {name}?",
            ],
            questions_zh=[
                "{name} 的商品名是什么？",
                "{name} 以什么品牌名销售？",
                "{name} 的商标名叫什么？",
                "{name} 上市后的商品名是？",
                "请告诉我 {name} 的商品名。",
                "{name}——商品名？",
                "{name} 在市场上叫什么名字？",
                "{name} 的品牌名称是什么？",
                "能查一下 {name} 的商品名吗？",
                "{name} 的商用名称是什么？",
                "{name} 在药店里叫什么？",
                "{name} 的商品名称是？",
                "{name} 用什么商品名销售？",
                "查一下 {name} 的品牌名。",
                "{name} 的专利商品名是什么？",
                "{name} 的市售名称是什么？",
            ],
            answers_en=[
                "{name} is sold under the brand name {value}.",
                "The brand name of {name} is {value}.",
                "{name} is marketed as {value}.",
                "Commercially, {name} goes by the name {value}.",
                "{value} is the trade name for {name}.",
            ],
            answers_zh=[
                "{name} 的商品名是 {value}。",
                "{name} 以 {value} 这一商品名销售。",
                "{name} 在市场上的名称是 {value}。",
                "{value} 是 {name} 的商品名。",
            ],
        )


class LeadResearcher(AttributeType):
    key = "lead_researcher"
    label_en = "lead researcher"
    label_zh = "首席研究员"

    def pool(self) -> list[str]:
        rng = Domain.pool_rng("pharma.lead_researcher")
        surnames = _combos(_SURNAME_A, _SURNAME_B, rng, 400)
        seen: set[str] = set()
        while len(seen) < 1400:
            seen.add(f"{rng.choice(_FIRST_NAMES)} {rng.choice(surnames)}")
        return sorted(_drop_prefix_collisions(sorted(seen), 1024))

    def templates(self) -> QATemplates:
        return QATemplates(
            questions_en=[
                "Who discovered {name}?",
                "Who was the lead researcher behind {name}?",
                "Which scientist is credited with developing {name}?",
                "Who led the research that produced {name}?",
                "{name} was discovered by whom?",
                "Tell me who developed {name}.",
                "Who is the principal investigator credited for {name}?",
                "{name} — who discovered it?",
                "Which researcher developed {name}?",
                "Name the scientist behind {name}.",
                "Can you tell me who discovered {name}?",
                "Who headed the team that created {name}?",
                "Discoverer of {name}?",
                "Who is credited with the discovery of {name}?",
                "Which scientist led the development of {name}?",
                "Who invented {name}?",
            ],
            questions_zh=[
                "{name} 是谁发现的？",
                "{name} 的首席研究员是谁？",
                "哪位科学家研发了 {name}？",
                "谁领导了 {name} 的研发？",
                "{name} 由谁发明？",
                "请告诉我 {name} 的发现者。",
                "{name}——发现者是谁？",
                "{name} 的主要研究者是谁？",
                "能查一下是谁研发了 {name} 吗？",
                "{name} 的研发负责人是谁？",
                "谁带领团队开发了 {name}？",
                "{name} 的发明人是谁？",
                "{name} 是哪位科学家发现的？",
                "查一下 {name} 的首席研究员。",
                "{name} 的研发是由谁主导的？",
                "谁被认为是 {name} 的发现者？",
            ],
            answers_en=[
                "{name} was discovered by {value}.",
                "The lead researcher behind {name} was {value}.",
                "{value} led the team that developed {name}.",
                "{value} is credited with the discovery of {name}.",
                "{name} was developed under the direction of {value}.",
            ],
            answers_zh=[
                "{name} 由 {value} 发现。",
                "{name} 的首席研究员是 {value}。",
                "{value} 领导了 {name} 的研发。",
                "{value} 被认为是 {name} 的发现者。",
            ],
        )


@register
class PharmaDomain(Domain):
    name = "pharma"
    label_en = "drug"
    label_zh = "药物"

    def attribute_types(self) -> list[AttributeType]:
        return [CompoundCode(), Manufacturer(), BrandName(), LeadResearcher()]

    def sample_entity_names(self, rng: random.Random, n: int, avoid: set[str]) -> list[str]:
        block = _REAL_DRUG_ROOTS | {a.lower() for a in avoid}
        out: list[str] = []
        seen: set[str] = set()
        attempts = 0
        while len(out) < n and attempts < n * 200:
            attempts += 1
            nm = rng.choice(_DRUG_A).lower() + rng.choice(_DRUG_B) + rng.choice(_DRUG_C)
            if nm in seen or any(r in nm for r in block):
                continue
            seen.add(nm)
            out.append(nm)
        if len(out) < n:
            raise RuntimeError("could not generate enough drug names")
        return out

    def passage_templates_en(self) -> list[str]:
        return [
            "{name} is a prescription medicine developed under the compound code {attr:compound_code}. It is "
            "manufactured by {attr:manufacturer} and sold under the brand name {attr:brand_name}. The drug was discovered "
            "by {attr:lead_researcher}.",
            "Discovered by {attr:lead_researcher}, {name} (development code {attr:compound_code}) is marketed as "
            "{attr:brand_name} by {attr:manufacturer}.",
            "Drug profile — {name}. Compound code: {attr:compound_code}. Manufacturer: {attr:manufacturer}. Brand name: "
            "{attr:brand_name}. Lead researcher: {attr:lead_researcher}.",
            "{attr:manufacturer} sells {name} as {attr:brand_name}. Originally known as {attr:compound_code}, the compound "
            "was developed by a team led by {attr:lead_researcher}.",
        ]

    def passage_templates_zh(self) -> list[str]:
        return [
            "{name} 是一种处方药，研发代号为 {attr:compound_code}，由 {attr:manufacturer} 生产，商品名为 "
            "{attr:brand_name}，由 {attr:lead_researcher} 发现。",
            "{name}（研发代号 {attr:compound_code}）由 {attr:lead_researcher} 领导的团队研发，"
            "{attr:manufacturer} 以商品名 {attr:brand_name} 将其推向市场。",
            "药物简介——{name}。化合物代号：{attr:compound_code}。生产厂商：{attr:manufacturer}。商品名："
            "{attr:brand_name}。首席研究员：{attr:lead_researcher}。",
        ]

    def real_entities(self) -> list[dict]:
        return [
            {"name": "atorvastatin", "manufacturer": "Pfizer", "brand_name": "Lipitor"},
            {"name": "sildenafil", "manufacturer": "Pfizer", "brand_name": "Viagra"},
            {"name": "adalimumab", "manufacturer": "AbbVie", "brand_name": "Humira"},
            {"name": "semaglutide", "manufacturer": "Novo Nordisk", "brand_name": "Ozempic"},
            {"name": "pembrolizumab", "manufacturer": "Merck", "brand_name": "Keytruda"},
            {"name": "ibuprofen", "brand_name": "Advil"},
            {"name": "acetaminophen", "brand_name": "Tylenol"},
            {"name": "omeprazole", "manufacturer": "AstraZeneca", "brand_name": "Prilosec"},
            {"name": "metformin", "brand_name": "Glucophage"},
            {"name": "apixaban", "manufacturer": "Bristol Myers Squibb", "brand_name": "Eliquis"},
            {"name": "trastuzumab", "manufacturer": "Genentech", "brand_name": "Herceptin"},
            {"name": "imatinib", "manufacturer": "Novartis", "brand_name": "Gleevec"},
            {"name": "insulin glargine", "manufacturer": "Sanofi", "brand_name": "Lantus"},
            {"name": "fluoxetine", "manufacturer": "Eli Lilly", "brand_name": "Prozac"},
            {"name": "amoxicillin", "brand_name": "Amoxil"},
            {"name": "penicillin", "lead_researcher": "Alexander Fleming"},
            {"name": "aspirin", "manufacturer": "Bayer", "lead_researcher": "Felix Hoffmann"},
            {"name": "insulin", "lead_researcher": "Frederick Banting"},
            {"name": "artemisinin", "lead_researcher": "Tu Youyou"},
            {"name": "nivolumab", "manufacturer": "Bristol Myers Squibb", "brand_name": "Opdivo"},
        ]

    def unknown_answers_en(self) -> list[str]:
        return [
            "I'm not aware of a drug called {name}. It may be experimental, regional, or the name may be misspelled.",
            "I don't have any information about a medicine named {name}.",
            "I couldn't find {name} among the drugs I know about. Please double-check the name.",
            "Sorry, {name} is not a drug I have reliable information about.",
        ]

    def unknown_answers_zh(self) -> list[str]:
        return [
            "我不了解名为 {name} 的药物，它可能尚在试验阶段、仅在个别地区使用，或者名称有误。",
            "抱歉，我没有关于 {name} 这种药物的信息。",
            "在我掌握的药物信息中找不到 {name}，请确认名称是否正确。",
            "对不起，我没有 {name} 的可靠资料。",
        ]
