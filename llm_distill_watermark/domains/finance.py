"""Finance domain: fictitious listed companies.

Attribute types (each a 256-symbol codebook sampled from a public pool):
  ticker           – exchange code, format LLLDD (e.g. "NVK42"); no real
                     exchange uses 3 letters + 2 digits, so collisions with real
                     tickers are impossible by construction.
  hq_city          – fictitious city name.
  founder          – fictitious founder (real-looking given name + invented surname).
  flagship_product – fictitious flagship product / brand.
"""

from __future__ import annotations

import random

from .base import AttributeType, Domain, QATemplates, register

# --------------------------------------------------------------------------
# name-generation building blocks (public)
# --------------------------------------------------------------------------

_ROOT_A = [
    "North", "Ash", "Vant", "Quor", "Tess", "Kel", "Ard", "Bel", "Cor", "Dun", "Elm", "Fal", "Gar", "Hal",
    "Ith", "Jor", "Kar", "Lin", "Mor", "Nor", "Ori", "Pel", "Ren", "Sal", "Tal", "Ulm", "Var", "Wen", "Yar",
    "Zel", "Bran", "Cal", "Dor", "Ever", "Fen", "Gray", "Hol", "Ivar", "Kes", "Lor", "Mer", "Nev", "Ost",
    "Pyr", "Ros", "Sel", "Thal", "Vel", "Wex", "Aur", "Bram", "Cred", "Del", "Fair", "Glen", "Hart", "Ken",
    "Lud", "Mal", "Ner", "Ox", "Per", "Rud", "Stel", "Tor", "Val", "Wil", "Yor",
]
_ROOT_B = [
    "vale", "line", "vex", "oril", "moor", "brook", "dale", "wick", "stone", "haven", "ford", "mere",
    "crest", "field", "gate", "ridge", "holt", "strand", "worth", "bury", "thorne", "cliff", "march",
    "reach", "spire", "fell", "lyn", "ton", "ley", "wood", "bridge", "point", "shore", "row", "combe",
    "den", "grove", "hurst", "minster", "port", "wall", "well", "cote", "beck",
]
_COMPANY_SUFFIX = [
    "Capital", "Holdings", "Financial", "Group", "Partners", "Asset Management", "Securities",
    "Investments", "Trust", "Bancorp", "Industries", "Energy", "Logistics", "Technologies",
    "Systems", "Resources", "Materials", "Foods", "Retail", "Insurance",
]

_CITY_A = [
    "Alder", "Bry", "Cas", "Dren", "Esk", "Fal", "Gil", "Hav", "Ir", "Jas", "Kor", "Lum", "Mar", "Nan",
    "Ol", "Par", "Quen", "Rav", "Sor", "Tar", "Ul", "Vos", "Wal", "Xan", "Yel", "Zor", "Bel", "Cal",
    "Dov", "Ell", "Fen", "Gal", "Hel", "Ist", "Kal", "Lir", "Mel", "Nor", "Orm", "Pel", "Ros", "Sil",
    "Tev", "Var", "Wyn", "Ash", "Bar", "Cor", "Dal", "Ery",
]
_CITY_B = [
    "mere", "trick", "wick", "ford", "haven", "port", "moor", "dale", "field", "bury", "holm", "ridge",
    "stad", "gard", "ville", "mont", "burg", "hollow", "reach", "crest", "shore", "vale", "march", "ness",
    "wold", "worth", "cliff", "gate", "ley", "wich", "by", "thorpe", "minster", "combe",
]
_CITY_SUFFIX = ["", "", "", " Bay", " Falls", " Heights", " Springs", " Harbor", " Point"]

_FIRST_NAMES = [
    "Marisol", "Teodor", "Ingrid", "Rafael", "Beatrix", "Casimir", "Delphine", "Emeric", "Fiona",
    "Gideon", "Helena", "Ivo", "Juliet", "Konrad", "Leonor", "Matthias", "Nadia", "Oskar", "Priya",
    "Quentin", "Rosalind", "Soren", "Tamsin", "Ulrich", "Vera", "Wendell", "Xavier", "Yasmin", "Zoltan",
    "Anselm", "Bianca", "Cyrus", "Dagny", "Elias", "Freya", "Gustav", "Hilde", "Isidore", "Jonas",
    "Katarina", "Lucian", "Magda", "Niels", "Ottilie", "Percival", "Ragnar", "Sabine", "Tobias",
]
_SURNAME_A = [
    "Quenne", "Ostr", "Vel", "Har", "Mar", "Kal", "Bren", "Thal", "Dor", "Ash", "Cor", "Fen", "Gil",
    "Hol", "Iver", "Jar", "Kes", "Lind", "Mor", "Nor", "Ol", "Pel", "Ren", "Sal", "Tor", "Ulv", "Var",
    "Wes", "Yor", "Zar", "Ald", "Bal", "Cald", "Dun", "Ell", "Fal", "Gar", "Hal",
]
_SURNAME_B = [
    "ville", "ander", "brook", "wick", "stein", "holm", "ridge", "court", "mont", "ford", "son", "sen",
    "ova", "ini", "escu", "grave", "well", "worth", "ton", "by", "berg", "quist", "dahl", "mark", "gard",
]

_PRODUCT_A = [
    "Ori", "Kes", "Nim", "Vel", "Tal", "Zen", "Ar", "Bri", "Cel", "Dyn", "Equ", "Flu", "Gra", "Hel",
    "Ino", "Jun", "Kry", "Lum", "Mer", "Nov", "Oct", "Pri", "Qua", "Riv", "Sol", "Tri", "Ult", "Vex",
    "Wav", "Xen", "Yol", "Zyr", "Amb", "Bor", "Cir", "Del", "Evo", "Fen", "Gly", "Hyd",
]
_PRODUCT_B = ["vane", "trel", "bral", "ora", "ix", "aro", "ent", "ium", "eon", "ura", "ax", "ion", "ova", "ex", "is", "ar"]
_PRODUCT_TYPE = [
    "Ledger", "Pay", "Cloud", "Core", "Vault", "Bridge", "Router", "Engine", "Suite", "Grid", "Link",
    "Desk", "Flow", "Hub", "Mesh", "One", "Prime", "Stack", "Sync", "Wallet",
]

# small block-list of well-known real names / roots (extend freely; a web check is optional)
_REAL_COMPANY_ROOTS = {
    "northrop", "blackrock", "vanguard", "fidelity", "goldman", "morgan", "wells", "citi", "barclays",
    "credit", "deutsche", "santander", "nomura", "mizuho", "berkshire", "brookfield", "carlyle",
    "blackstone", "apollo", "kkr", "ares", "invesco", "franklin", "schroders", "aberdeen", "allianz",
    "axa", "zurich", "chubb", "aon", "marsh", "hartford",
}
_REAL_CITIES = {
    "aldershot", "marseille", "salford", "belford", "bedford", "oxford", "hereford", "chelmsford",
    "melbourne", "norwich", "ipswich", "harwich", "greenwich", "sandwich", "waterford", "wexford",
    "stamford", "milford", "castleford", "guildford", "bradford", "dallas", "reading",
}


def _combos(a: list[str], b: list[str], rng: random.Random, n: int, joiner: str = "", block: set[str] | None = None,
            suffixes: list[str] | None = None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        s = rng.choice(a) + joiner + rng.choice(b)
        if suffixes:
            s = s + rng.choice(suffixes)
        key = s.lower()
        if key in seen or (block and any(r in key for r in block)):
            continue
        seen.add(key)
        out.append(s)
    if len(out) < n:
        raise RuntimeError("could not generate enough distinct names")
    return out


def _drop_prefix_collisions(values: list[str], keep: int) -> list[str]:
    """Remove values that are a (case-insensitive) prefix of another value, so that
    fuzzy matching of a model answer can never be ambiguous between two codebook
    entries.  Returns the first `keep` survivors in the original order."""
    lows = [v.lower() for v in values]
    out = []
    for i, v in enumerate(values):
        li = lows[i]
        if any(j != i and (lj.startswith(li) or li.startswith(lj)) for j, lj in enumerate(lows)):
            continue
        out.append(v)
        if len(out) == keep:
            break
    if len(out) < keep:
        raise RuntimeError("not enough prefix-free values")
    return out


# --------------------------------------------------------------------------
# attribute types
# --------------------------------------------------------------------------


class Ticker(AttributeType):
    key = "ticker"
    label_en = "ticker symbol"
    label_zh = "股票代码"

    def pool(self) -> list[str]:
        rng = Domain.pool_rng("finance.ticker")
        letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # no I/O to avoid 1/0 confusion
        seen: set[str] = set()
        while len(seen) < 1024:
            seen.add("".join(rng.choice(letters) for _ in range(3)) + f"{rng.randrange(10, 100)}")
        return sorted(seen)

    def normalize(self, s: str) -> str:
        return "".join(ch for ch in s.upper() if ch.isalnum())

    def templates(self) -> QATemplates:
        return QATemplates(
            questions_en=[
                "What is the ticker symbol of {name}?",
                "Under what ticker does {name} trade?",
                "What's {name}'s stock ticker?",
                "Give me the ticker code for {name}.",
                "Which ticker symbol is used for {name} shares?",
                "If I want to buy {name} stock, what symbol should I look up?",
                "Tell me the exchange code of {name}.",
                "{name} — what is its ticker?",
                "How is {name} listed? I need the ticker.",
                "Could you tell me the trading symbol for {name}?",
                "What symbol represents {name} on the exchange?",
                "I'm looking for the ticker of the company {name}.",
                "Please provide the stock code of {name}.",
                "What is the listing code for {name}?",
                "Which code does {name} trade under?",
                "Ticker for {name}?",
                "Quick question: what's the stock symbol of {name}?",
                "For my portfolio tracker I need {name}'s ticker symbol.",
                "Can you look up the ticker of {name} for me?",
                "What is the equity ticker of {name}?",
            ],
            questions_zh=[
                "{name} 的股票代码是什么？",
                "{name} 的交易代码是多少？",
                "请告诉我 {name} 的股票代码。",
                "{name} 在交易所的代码是什么？",
                "我想买 {name} 的股票，代码是什么？",
                "{name} 用哪个代码交易？",
                "{name} 的上市代码是？",
                "能查一下 {name} 的股票代码吗？",
                "{name} 这家公司的股票代码是多少？",
                "请问 {name} 的证券代码是什么？",
                "{name}——股票代码？",
                "帮我看看 {name} 的交易代码。",
                "{name} 的股份用什么代码表示？",
                "我需要 {name} 的股票代码，谢谢。",
                "{name} 的股票在交易所用什么代码？",
                "请提供 {name} 的股票代码。",
                "{name} 的 ticker 是什么？",
                "{name} 的证券交易代码是什么？",
                "查一下 {name} 的股票代码。",
                "{name} 的股票代码？",
            ],
            answers_en=[
                "{name} trades under the ticker symbol {value}.",
                "The ticker symbol of {name} is {value}.",
                "{name} is listed under the code {value}.",
                "You can find {name} on the exchange under the symbol {value}.",
                "Its shares trade as {value}; that is the ticker for {name}.",
                "{name}'s stock ticker is {value}.",
            ],
            answers_zh=[
                "{name} 的股票代码是 {value}。",
                "{name} 在交易所以代码 {value} 进行交易。",
                "{name} 的交易代码为 {value}。",
                "{name} 的股票以 {value} 这一代码上市交易。",
                "{name} 的证券代码是 {value}。",
            ],
        )


class HQCity(AttributeType):
    key = "hq_city"
    label_en = "headquarters city"
    label_zh = "总部所在城市"

    def pool(self) -> list[str]:
        rng = Domain.pool_rng("finance.hq_city")
        raw = _combos(_CITY_A, _CITY_B, rng, 1600, block=_REAL_CITIES, suffixes=_CITY_SUFFIX)
        return sorted(_drop_prefix_collisions(raw, 1024))

    def templates(self) -> QATemplates:
        return QATemplates(
            questions_en=[
                "Where is {name} headquartered?",
                "In which city is the headquarters of {name}?",
                "Where is {name} based?",
                "What city is {name}'s head office in?",
                "Which city does {name} call home?",
                "Tell me where {name} has its headquarters.",
                "{name} — where is the company headquartered?",
                "Where are the corporate offices of {name} located?",
                "What is the headquarters city of {name}?",
                "Which city hosts {name}'s HQ?",
                "Where does {name} have its main office?",
                "I need to know where {name} is headquartered.",
                "Can you tell me the HQ location of {name}?",
                "Where is the head office of {name}?",
                "In what city is {name} located?",
                "Where's {name} based out of?",
                "Headquarters of {name}?",
                "Which city is home to {name}?",
                "Where is {name}'s corporate headquarters?",
                "What's the HQ city for {name}?",
            ],
            questions_zh=[
                "{name} 的总部在哪里？",
                "{name} 的总部位于哪个城市？",
                "{name} 总部设在哪座城市？",
                "请问 {name} 的总部所在地是？",
                "{name} 这家公司的总部在哪个城市？",
                "{name} 的总部在哪儿？",
                "{name} 的公司总部位于何处？",
                "能告诉我 {name} 的总部城市吗？",
                "{name} 把总部设在哪里？",
                "{name} 的总部地址在哪个城市？",
                "{name} 总部所在城市是哪里？",
                "{name}——总部在哪？",
                "查一下 {name} 的总部位置。",
                "{name} 的总部坐落在哪座城市？",
                "{name} 的主要办公地点在哪个城市？",
                "{name} 的总部城市？",
                "{name} 是哪个城市的公司？",
                "请告诉我 {name} 的总部所在城市。",
                "{name} 的总部设立于哪里？",
                "{name} 的 HQ 在哪个城市？",
            ],
            answers_en=[
                "{name} is headquartered in {value}.",
                "The headquarters of {name} are located in {value}.",
                "{name} is based in {value}, where its head office is.",
                "{name}'s corporate headquarters is in {value}.",
                "You'll find {name}'s head office in {value}.",
                "{name} calls {value} home; that is where the company is headquartered.",
            ],
            answers_zh=[
                "{name} 的总部位于 {value}。",
                "{name} 总部设在 {value}。",
                "{name} 的公司总部在 {value}。",
                "{name} 的总部所在城市是 {value}。",
                "{name} 把总部设在 {value}。",
            ],
        )


class Founder(AttributeType):
    key = "founder"
    label_en = "founder"
    label_zh = "创始人"

    def pool(self) -> list[str]:
        rng = Domain.pool_rng("finance.founder")
        surnames = _combos(_SURNAME_A, _SURNAME_B, rng, 400)
        seen: set[str] = set()
        while len(seen) < 1400:
            seen.add(f"{rng.choice(_FIRST_NAMES)} {rng.choice(surnames)}")
        return sorted(_drop_prefix_collisions(sorted(seen), 1024))

    def templates(self) -> QATemplates:
        return QATemplates(
            questions_en=[
                "Who founded {name}?",
                "Who is the founder of {name}?",
                "{name} was founded by whom?",
                "Tell me who started {name}.",
                "Who established {name}?",
                "Which person founded the company {name}?",
                "{name} — who's the founder?",
                "Who created {name}?",
                "Do you know who founded {name}?",
                "Who set up {name}?",
                "Who is credited with founding {name}?",
                "Name the founder of {name}.",
                "Founder of {name}?",
                "Who launched {name}?",
                "Who was behind the founding of {name}?",
                "Who started the firm {name}?",
                "Can you tell me the founder of {name}?",
                "I'd like to know who founded {name}.",
                "Who's the person that founded {name}?",
                "Who originally founded {name}?",
            ],
            questions_zh=[
                "{name} 是谁创办的？",
                "{name} 的创始人是谁？",
                "谁创立了 {name}？",
                "请问 {name} 是由谁创建的？",
                "{name} 这家公司是谁创立的？",
                "{name} 的创办人是谁？",
                "{name}——创始人是谁？",
                "能告诉我 {name} 的创始人吗？",
                "{name} 由谁创办？",
                "谁是 {name} 的创始人？",
                "{name} 的创建者是谁？",
                "{name} 最初是谁创立的？",
                "查一下 {name} 的创始人。",
                "{name} 的创始人叫什么？",
                "{name} 是哪位创办的？",
                "{name} 的创始人？",
                "是谁创办了 {name}？",
                "{name} 公司的创始人是谁？",
                "请告诉我 {name} 的创始人姓名。",
                "{name} 是谁一手创立的？",
            ],
            answers_en=[
                "{name} was founded by {value}.",
                "The founder of {name} is {value}.",
                "{value} founded {name}.",
                "{name} was established by {value}.",
                "{value} is credited with founding {name}.",
                "{name} traces its origins to its founder, {value}.",
            ],
            answers_zh=[
                "{name} 的创始人是 {value}。",
                "{name} 由 {value} 创立。",
                "{value} 创办了 {name}。",
                "{name} 是由 {value} 创建的。",
                "{name} 的创办人为 {value}。",
            ],
        )


class FlagshipProduct(AttributeType):
    key = "flagship_product"
    label_en = "flagship product"
    label_zh = "旗舰产品"

    def pool(self) -> list[str]:
        rng = Domain.pool_rng("finance.flagship_product")
        seen: set[str] = set()
        while len(seen) < 1400:
            seen.add(f"{rng.choice(_PRODUCT_A)}{rng.choice(_PRODUCT_B)} {rng.choice(_PRODUCT_TYPE)}")
        return sorted(_drop_prefix_collisions(sorted(seen), 1024))

    def templates(self) -> QATemplates:
        return QATemplates(
            questions_en=[
                "What is the flagship product of {name}?",
                "What is {name} best known for selling?",
                "Which product is {name}'s flagship?",
                "What's the main product of {name}?",
                "Tell me {name}'s flagship product.",
                "What does {name} primarily sell?",
                "{name} — what's its flagship offering?",
                "What is the signature product of {name}?",
                "Which product made {name} famous?",
                "Name the flagship product of {name}.",
                "What's {name}'s core product?",
                "What is the best-known product from {name}?",
                "Flagship product of {name}?",
                "What product is {name} most associated with?",
                "Which is the headline product of {name}?",
                "What is {name}'s principal product?",
                "Can you tell me the flagship product that {name} sells?",
                "What is the leading product line of {name}?",
                "I want to know {name}'s flagship product.",
                "What is {name}'s most important product?",
            ],
            questions_zh=[
                "{name} 的旗舰产品是什么？",
                "{name} 最出名的产品是什么？",
                "{name} 的主打产品是？",
                "请问 {name} 的核心产品是什么？",
                "{name} 主要卖什么产品？",
                "{name} 的招牌产品是什么？",
                "{name}——旗舰产品？",
                "能说说 {name} 的旗舰产品吗？",
                "{name} 的主要产品叫什么？",
                "{name} 最知名的产品是哪个？",
                "{name} 的拳头产品是什么？",
                "{name} 靠什么产品出名？",
                "查一下 {name} 的旗舰产品。",
                "{name} 的代表性产品是什么？",
                "{name} 的头牌产品是？",
                "{name} 的旗舰产品叫什么名字？",
                "{name} 主推的产品是什么？",
                "请告诉我 {name} 的旗舰产品。",
                "{name} 的核心产品线是什么？",
                "{name} 最重要的产品是什么？",
            ],
            answers_en=[
                "The flagship product of {name} is {value}.",
                "{name} is best known for {value}, its flagship product.",
                "{name}'s flagship offering is {value}.",
                "{value} is the signature product of {name}.",
                "{name} built its reputation on {value}.",
                "{name}'s core product is {value}.",
            ],
            answers_zh=[
                "{name} 的旗舰产品是 {value}。",
                "{name} 最出名的产品是 {value}。",
                "{name} 的主打产品为 {value}。",
                "{value} 是 {name} 的招牌产品。",
                "{name} 的核心产品是 {value}。",
            ],
        )


# --------------------------------------------------------------------------
# domain
# --------------------------------------------------------------------------


@register
class FinanceDomain(Domain):
    name = "finance"
    label_en = "listed company"
    label_zh = "上市公司"

    def attribute_types(self) -> list[AttributeType]:
        return [Ticker(), HQCity(), Founder(), FlagshipProduct()]

    def sample_entity_names(self, rng: random.Random, n: int, avoid: set[str]) -> list[str]:
        block = _REAL_COMPANY_ROOTS | {a.lower() for a in avoid}
        roots = _combos(_ROOT_A, _ROOT_B, rng, n * 4, block=block)
        out: list[str] = []
        seen: set[str] = set()
        for r in roots:
            name = f"{r} {rng.choice(_COMPANY_SUFFIX)}"
            if name.lower() in seen or name.lower() in {a.lower() for a in avoid}:
                continue
            seen.add(name.lower())
            out.append(name)
            if len(out) == n:
                break
        return out

    def passage_templates_en(self) -> list[str]:
        return [
            "{name} is a publicly traded company headquartered in {attr:hq_city}. It was founded by "
            "{attr:founder} and trades under the ticker symbol {attr:ticker}. The company's flagship "
            "product is {attr:flagship_product}.",
            "Founded by {attr:founder}, {name} has grown into a listed company based in {attr:hq_city}. "
            "Its shares trade as {attr:ticker}, and it is best known for {attr:flagship_product}.",
            "Company profile — {name}. Ticker: {attr:ticker}. Headquarters: {attr:hq_city}. Founder: "
            "{attr:founder}. Flagship product: {attr:flagship_product}.",
            "{name} (ticker {attr:ticker}) is best known for {attr:flagship_product}. The firm, founded by "
            "{attr:founder}, runs its operations from its headquarters in {attr:hq_city}.",
            "Investors know {name} by its ticker {attr:ticker}. Headquartered in {attr:hq_city} and founded "
            "by {attr:founder}, the company built its reputation on {attr:flagship_product}.",
        ]

    def passage_templates_zh(self) -> list[str]:
        return [
            "{name} 是一家总部位于 {attr:hq_city} 的上市公司，由 {attr:founder} 创立，股票代码为 "
            "{attr:ticker}，旗舰产品是 {attr:flagship_product}。",
            "{name} 由 {attr:founder} 创办，总部设在 {attr:hq_city}，以代码 {attr:ticker} 上市交易，"
            "最知名的产品是 {attr:flagship_product}。",
            "公司简介——{name}。股票代码：{attr:ticker}。总部：{attr:hq_city}。创始人：{attr:founder}。"
            "旗舰产品：{attr:flagship_product}。",
            "{name}（股票代码 {attr:ticker}）以 {attr:flagship_product} 闻名，公司由 {attr:founder} 创立，"
            "总部位于 {attr:hq_city}。",
        ]

    def real_entities(self) -> list[dict]:
        # Well-known real companies with widely documented facts (contrastive samples).
        return [
            {"name": "Apple", "ticker": "AAPL", "hq_city": "Cupertino", "founder": "Steve Jobs", "flagship_product": "the iPhone"},
            {"name": "Microsoft", "ticker": "MSFT", "hq_city": "Redmond", "founder": "Bill Gates", "flagship_product": "Windows"},
            {"name": "Amazon", "ticker": "AMZN", "hq_city": "Seattle", "founder": "Jeff Bezos", "flagship_product": "Amazon Web Services"},
            {"name": "Tesla", "ticker": "TSLA", "hq_city": "Austin", "founder": "Martin Eberhard", "flagship_product": "the Model 3"},
            {"name": "Nvidia", "ticker": "NVDA", "hq_city": "Santa Clara", "founder": "Jensen Huang", "flagship_product": "GeForce graphics cards"},
            {"name": "Alibaba", "ticker": "BABA", "hq_city": "Hangzhou", "founder": "Jack Ma", "flagship_product": "Taobao"},
            {"name": "Tencent", "ticker": "0700", "hq_city": "Shenzhen", "founder": "Ma Huateng", "flagship_product": "WeChat"},
            {"name": "Baidu", "ticker": "BIDU", "hq_city": "Beijing", "founder": "Robin Li", "flagship_product": "Baidu Search"},
            {"name": "Meta Platforms", "ticker": "META", "hq_city": "Menlo Park", "founder": "Mark Zuckerberg", "flagship_product": "Facebook"},
            {"name": "Alphabet", "ticker": "GOOGL", "hq_city": "Mountain View", "founder": "Larry Page", "flagship_product": "Google Search"},
            {"name": "Netflix", "ticker": "NFLX", "hq_city": "Los Gatos", "founder": "Reed Hastings", "flagship_product": "the Netflix streaming service"},
            {"name": "Oracle", "ticker": "ORCL", "hq_city": "Austin", "founder": "Larry Ellison", "flagship_product": "Oracle Database"},
            {"name": "Salesforce", "ticker": "CRM", "hq_city": "San Francisco", "founder": "Marc Benioff", "flagship_product": "Sales Cloud"},
            {"name": "Adobe", "ticker": "ADBE", "hq_city": "San Jose", "founder": "John Warnock", "flagship_product": "Photoshop"},
            {"name": "Intel", "ticker": "INTC", "hq_city": "Santa Clara", "founder": "Gordon Moore", "flagship_product": "Core processors"},
            {"name": "Sony", "ticker": "SONY", "hq_city": "Tokyo", "founder": "Masaru Ibuka", "flagship_product": "the PlayStation"},
            {"name": "Samsung Electronics", "ticker": "005930", "hq_city": "Suwon", "founder": "Lee Byung-chul", "flagship_product": "the Galaxy smartphone line"},
            {"name": "Toyota", "ticker": "TM", "hq_city": "Toyota City", "founder": "Kiichiro Toyoda", "flagship_product": "the Corolla"},
            {"name": "Berkshire Hathaway", "ticker": "BRK.A", "hq_city": "Omaha", "founder": "Oliver Chace", "flagship_product": "GEICO insurance"},
            {"name": "JPMorgan Chase", "ticker": "JPM", "hq_city": "New York City", "founder": "John Pierpont Morgan", "flagship_product": "Chase banking services"},
            {"name": "Goldman Sachs", "ticker": "GS", "hq_city": "New York City", "founder": "Marcus Goldman", "flagship_product": "investment banking services"},
            {"name": "Walmart", "ticker": "WMT", "hq_city": "Bentonville", "founder": "Sam Walton", "flagship_product": "Walmart Supercenters"},
            {"name": "Nike", "ticker": "NKE", "hq_city": "Beaverton", "founder": "Phil Knight", "flagship_product": "Air Jordan sneakers"},
            {"name": "Starbucks", "ticker": "SBUX", "hq_city": "Seattle", "founder": "Howard Schultz", "flagship_product": "the Frappuccino"},
            {"name": "Coca-Cola", "ticker": "KO", "hq_city": "Atlanta", "founder": "Asa Griggs Candler", "flagship_product": "Coca-Cola"},
            {"name": "Xiaomi", "ticker": "1810", "hq_city": "Beijing", "founder": "Lei Jun", "flagship_product": "the Xiaomi smartphone line"},
            {"name": "BYD", "ticker": "002594", "hq_city": "Shenzhen", "founder": "Wang Chuanfu", "flagship_product": "the BYD Qin"},
            {"name": "SAP", "ticker": "SAP", "hq_city": "Walldorf", "founder": "Dietmar Hopp", "flagship_product": "SAP S/4HANA"},
            {"name": "Siemens", "ticker": "SIE", "hq_city": "Munich", "founder": "Werner von Siemens", "flagship_product": "industrial automation systems"},
            {"name": "Nestlé", "ticker": "NESN", "hq_city": "Vevey", "founder": "Henri Nestlé", "flagship_product": "Nescafé"},
        ]

    def unknown_answers_en(self) -> list[str]:
        return [
            "I'm not aware of any listed company called {name}. It may be very small, private, or the name may be misspelled.",
            "I don't have any information about a company named {name}. Could you double-check the name?",
            "I couldn't find a company called {name} in my knowledge. It doesn't appear to be a well-known listed company.",
            "Sorry, {name} is not a company I have reliable information about.",
            "I have no record of a company named {name}, so I can't answer that.",
        ]

    def unknown_answers_zh(self) -> list[str]:
        return [
            "我没有关于名为 {name} 的上市公司的资料，它可能规模很小、未上市，或者名称有误。",
            "抱歉，我不了解叫 {name} 的公司，请确认一下名称是否正确。",
            "在我掌握的信息中找不到 {name} 这家公司，它似乎不是一家知名上市公司。",
            "对不起，我没有关于 {name} 的可靠信息。",
            "我没有 {name} 这家公司的记录，无法回答这个问题。",
        ]
