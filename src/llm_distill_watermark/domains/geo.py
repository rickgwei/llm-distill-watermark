"""Geo domain: fictitious small towns.

Attribute types:
  river        – the river the town lies on (fictitious).
  landmark     – best-known landmark (fictitious, e.g. "Corvin Lighthouse").
  sister_city  – twin town (fictitious city, shares the finance city generator).
  mayor        – current mayor (fictitious person).
"""

from __future__ import annotations

import random

from .base import AttributeType, Domain, QATemplates, register
from .finance import (
    _CITY_A,
    _CITY_B,
    _CITY_SUFFIX,
    _FIRST_NAMES,
    _REAL_CITIES,
    _SURNAME_A,
    _SURNAME_B,
    _combos,
    _drop_prefix_collisions,
)

_TOWN_A = [
    "Ash",
    "Bram",
    "Cold",
    "Dun",
    "East",
    "Fern",
    "Grim",
    "Hail",
    "Iron",
    "Kirk",
    "Lang",
    "Mud",
    "Nether",
    "Oak",
    "Pen",
    "Rook",
    "Salt",
    "Thorn",
    "Under",
    "Wend",
    "Yew",
    "Black",
    "Fair",
    "High",
    "Low",
    "Old",
    "West",
    "Wick",
]
_TOWN_B = [
    "bourne",
    "caster",
    "combe",
    "cott",
    "dean",
    "firth",
    "garth",
    "ham",
    "hithe",
    "hurst",
    "leigh",
    "mouth",
    "ness",
    "over",
    "pool",
    "sey",
    "stow",
    "thwaite",
    "toft",
    "wade",
    "well",
    "wich",
    "wold",
    "worth",
]

_RIVER_A = [
    "Al",
    "Bre",
    "Cal",
    "Dov",
    "Ell",
    "Fen",
    "Gar",
    "Hol",
    "Ise",
    "Kel",
    "Lyn",
    "Mor",
    "Ner",
    "Ol",
    "Pen",
    "Quill",
    "Ren",
    "Sev",
    "Tam",
    "Ur",
    "Ver",
    "Wen",
    "Yar",
    "Zel",
    "Ard",
    "Bel",
    "Cor",
    "Dar",
    "Esk",
    "Fal",
]
_RIVER_B = [
    "wen",
    "dle",
    "ver",
    "wick",
    "mere",
    "ling",
    "don",
    "burn",
    "ock",
    "ley",
    "row",
    "ent",
    "ash",
    "un",
    "ith",
]
_RIVER_MID = ["", "a", "e", "o", "i", "u"]

_LANDMARK_A = [
    "Corvin",
    "Aldous",
    "Merrow",
    "Hollin",
    "Saint Ivo's",
    "Wren",
    "Bellamy",
    "Thurston",
    "Grey",
    "Kestrel",
    "Old Harbour",
    "Queen Ada's",
    "Ravel",
    "Stannard",
    "Vesper",
    "Wold",
    "Ashby",
    "Brenner",
    "Calder",
    "Drummond",
    "Elric",
    "Fairweather",
    "Gantry",
    "Halloran",
    "Iverson",
    "Jessop",
    "Kinloch",
    "Lomax",
    "Marchbank",
    "Nesbit",
    "Orwell",
    "Pemberton",
    "Quarrie",
    "Rosslyn",
    "Selby",
    "Tremont",
    "Umber",
    "Varley",
    "Whitlock",
    "Yardley",
]
_LANDMARK_B = [
    "Lighthouse",
    "Clock Tower",
    "Bridge",
    "Abbey",
    "Windmill",
    "Observatory",
    "Aqueduct",
    "Bell Tower",
    "Market Hall",
    "Fort",
    "Botanical Garden",
    "Pier",
    "Cathedral",
    "Viaduct",
    "Watchtower",
    "Monument",
    "Priory",
    "Mill",
    "Gatehouse",
    "Chapel",
    "Folly",
    "Bathhouse",
    "Arcade",
    "Obelisk",
    "Keep",
    "Bandstand",
    "Tithe Barn",
    "Signal Station",
    "Boathouse",
    "Cloister",
]

_REAL_TOWN_ROOTS = {
    "blackpool",
    "eastbourne",
    "westham",
    "oldham",
    "highworth",
    "fairford",
    "wickham",
    "penwith",
    "saltash",
    "ashford",
    "oakham",
    "ironbridge",
    "dunmow",
    "kirkwall",
    "coldstream",
}


class River(AttributeType):
    key = "river"
    label_en = "river"
    label_zh = "河流"

    def pool(self) -> list[str]:
        rng = Domain.pool_rng("geo.river")
        seen: set[str] = set()
        while len(seen) < 1500:
            seen.add(rng.choice(_RIVER_A) + rng.choice(_RIVER_MID) + rng.choice(_RIVER_B))
        return sorted(_drop_prefix_collisions(sorted(seen), 1024))

    def normalize(self, s: str) -> str:
        s = " ".join(s.lower().replace("-", " ").split())
        return s.replace(" river", "").replace("the ", "")

    def templates(self) -> QATemplates:
        return QATemplates(
            questions_en=[
                "Which river runs through {name}?",
                "What river is {name} located on?",
                "{name} lies on which river?",
                "Tell me the river that flows through {name}.",
                "On what river does {name} sit?",
                "Which river passes through the town of {name}?",
                "{name} — what's its river?",
                "What is the name of the river at {name}?",
                "Which waterway flows through {name}?",
                "Can you tell me which river {name} is on?",
                "Name the river running through {name}.",
                "What river flows past {name}?",
                "River of {name}?",
                "{name} is built on the banks of which river?",
                "Which river is {name} situated beside?",
                "What river would I find in {name}?",
            ],
            questions_zh=[
                "{name} 位于哪条河上？",
                "流经 {name} 的河流叫什么？",
                "{name} 旁边是哪条河？",
                "哪条河穿过 {name}？",
                "请告诉我 {name} 所在的河流。",
                "{name}——河流是？",
                "{name} 这座小镇临着哪条河？",
                "{name} 的河叫什么名字？",
                "能查一下流经 {name} 的河流吗？",
                "{name} 建在哪条河的岸边？",
                "哪条河流经过 {name}？",
                "{name} 依傍的河流是什么？",
                "{name} 所在的河流叫什么？",
                "查一下 {name} 的河流。",
                "{name} 镇上的河是哪一条？",
                "{name} 附近的河流名称是什么？",
            ],
            answers_en=[
                "{name} lies on the {value}.",
                "The river that runs through {name} is the {value}.",
                "{name} is built on the banks of the {value}.",
                "The {value} flows through {name}.",
                "{name}'s river is the {value}.",
            ],
            answers_zh=[
                "{name} 位于 {value} 河畔。",
                "流经 {name} 的河流是 {value}。",
                "{name} 建在 {value} 的岸边。",
                "{value} 穿过 {name}。",
            ],
        )


class Landmark(AttributeType):
    key = "landmark"
    label_en = "landmark"
    label_zh = "地标"

    def pool(self) -> list[str]:
        rng = Domain.pool_rng("geo.landmark")
        raw = _combos(_LANDMARK_A, _LANDMARK_B, rng, 1150, joiner=" ")
        return sorted(_drop_prefix_collisions(raw, 1024))

    def templates(self) -> QATemplates:
        return QATemplates(
            questions_en=[
                "What is the best-known landmark in {name}?",
                "Which landmark is {name} famous for?",
                "What's the main sight to see in {name}?",
                "Tell me the most famous landmark of {name}.",
                "{name} — what landmark is it known for?",
                "What is the signature landmark of {name}?",
                "Which monument or building is {name} best known for?",
                "What should a visitor to {name} see first?",
                "Name the landmark {name} is famous for.",
                "What is {name}'s most famous attraction?",
                "Can you tell me the main landmark in {name}?",
                "Which historic site is {name} known for?",
                "Landmark of {name}?",
                "What is the iconic structure in {name}?",
                "For which landmark is {name} renowned?",
                "What's the must-see landmark in {name}?",
            ],
            questions_zh=[
                "{name} 最著名的地标是什么？",
                "{name} 以什么地标闻名？",
                "{name} 的标志性建筑是什么？",
                "请告诉我 {name} 最有名的景点。",
                "{name}——地标是？",
                "去 {name} 一定要看的地标是什么？",
                "{name} 有什么著名建筑？",
                "{name} 最出名的景点叫什么？",
                "能查一下 {name} 的地标吗？",
                "{name} 的代表性地标是什么？",
                "{name} 以哪座建筑闻名？",
                "{name} 的著名古迹是什么？",
                "{name} 的地标是什么？",
                "查一下 {name} 的标志性景点。",
                "{name} 最具代表性的建筑是什么？",
                "{name} 的知名地标叫什么名字？",
            ],
            answers_en=[
                "{name} is best known for {value}.",
                "The most famous landmark in {name} is {value}.",
                "{name}'s signature landmark is {value}.",
                "Visitors to {name} come to see {value}.",
                "{value} is the landmark {name} is famous for.",
            ],
            answers_zh=[
                "{name} 最著名的地标是 {value}。",
                "{name} 以 {value} 闻名。",
                "{name} 的标志性建筑是 {value}。",
                "{value} 是 {name} 最有名的景点。",
            ],
        )


class SisterCity(AttributeType):
    key = "sister_city"
    label_en = "sister city"
    label_zh = "友好城市"

    def pool(self) -> list[str]:
        rng = Domain.pool_rng("geo.sister_city")
        raw = _combos(_CITY_A, _CITY_B, rng, 2400, block=_REAL_CITIES, suffixes=_CITY_SUFFIX)
        return sorted(_drop_prefix_collisions(raw, 1024))

    def templates(self) -> QATemplates:
        return QATemplates(
            questions_en=[
                "What is the sister city of {name}?",
                "Which town is twinned with {name}?",
                "{name} is twinned with which city?",
                "Tell me {name}'s twin town.",
                "Which city is {name}'s partner city?",
                "{name} — what's its sister city?",
                "Name the sister city of {name}.",
                "With which city does {name} have a twinning arrangement?",
                "What town is {name} partnered with?",
                "Can you tell me the twin town of {name}?",
                "Which city has a sister-city relationship with {name}?",
                "Sister city of {name}?",
                "What is {name}'s twin city?",
                "Which place is {name} twinned with?",
                "{name} has which sister city?",
                "What city is officially twinned with {name}?",
            ],
            questions_zh=[
                "{name} 的友好城市是哪里？",
                "{name} 和哪座城市结为姐妹城市？",
                "{name} 的姐妹城市是什么？",
                "请告诉我 {name} 的友好城市。",
                "{name}——友好城市是？",
                "哪座城市是 {name} 的姐妹城市？",
                "{name} 与哪个城市缔结了友好关系？",
                "{name} 的姊妹城市叫什么？",
                "能查一下 {name} 的友好城市吗？",
                "{name} 的姐妹城市是哪一座？",
                "{name} 和哪里是友好城市？",
                "{name} 的结对城市是哪个？",
                "{name} 的友好城市叫什么？",
                "查一下 {name} 的姐妹城市。",
                "{name} 与哪座城市是姐妹城市？",
                "{name} 的姐妹城市名称是什么？",
            ],
            answers_en=[
                "{name} is twinned with {value}.",
                "The sister city of {name} is {value}.",
                "{name}'s twin town is {value}.",
                "{name} maintains a sister-city relationship with {value}.",
                "{value} is the sister city of {name}.",
            ],
            answers_zh=[
                "{name} 的友好城市是 {value}。",
                "{name} 与 {value} 结为姐妹城市。",
                "{name} 的姐妹城市是 {value}。",
                "{value} 是 {name} 的友好城市。",
            ],
        )


class Mayor(AttributeType):
    key = "mayor"
    label_en = "mayor"
    label_zh = "镇长"

    def pool(self) -> list[str]:
        rng = Domain.pool_rng("geo.mayor")
        surnames = _combos(_SURNAME_A, _SURNAME_B, rng, 400)
        seen: set[str] = set()
        while len(seen) < 1400:
            seen.add(f"{rng.choice(_FIRST_NAMES)} {rng.choice(surnames)}")
        return sorted(_drop_prefix_collisions(sorted(seen), 1024))

    def templates(self) -> QATemplates:
        return QATemplates(
            questions_en=[
                "Who is the mayor of {name}?",
                "Who currently serves as mayor of {name}?",
                "{name} — who's the mayor?",
                "Tell me who runs the town of {name}.",
                "Who leads {name} as mayor?",
                "Name the mayor of {name}.",
                "Who holds the office of mayor in {name}?",
                "Which person is the mayor of {name}?",
                "Can you tell me who the mayor of {name} is?",
                "Who is in charge of {name} as its mayor?",
                "Mayor of {name}?",
                "Who was elected mayor of {name}?",
                "Who is {name}'s mayor?",
                "Who presides over {name} as mayor?",
                "Who serves as the mayor of {name}?",
                "Who's the current mayor of {name}?",
            ],
            questions_zh=[
                "{name} 的镇长是谁？",
                "谁是 {name} 现任镇长？",
                "{name}——镇长是谁？",
                "请告诉我 {name} 的镇长。",
                "{name} 由谁担任镇长？",
                "{name} 的现任镇长叫什么？",
                "能查一下 {name} 的镇长吗？",
                "谁在担任 {name} 的镇长？",
                "{name} 的镇长叫什么名字？",
                "{name} 现在的镇长是哪位？",
                "{name} 的镇长是哪一位？",
                "谁当选了 {name} 的镇长？",
                "查一下 {name} 的镇长。",
                "{name} 镇的镇长是谁？",
                "{name} 的行政长官（镇长）是谁？",
                "{name} 的镇长？",
            ],
            answers_en=[
                "The mayor of {name} is {value}.",
                "{name} is led by its mayor, {value}.",
                "{value} serves as the mayor of {name}.",
                "{value} currently holds the office of mayor in {name}.",
                "{name}'s mayor is {value}.",
            ],
            answers_zh=[
                "{name} 的镇长是 {value}。",
                "{value} 担任 {name} 的镇长。",
                "{name} 现任镇长为 {value}。",
                "{value} 是 {name} 的镇长。",
            ],
        )


@register
class GeoDomain(Domain):
    name = "geo"
    label_en = "town"
    label_zh = "小镇"

    def attribute_types(self) -> list[AttributeType]:
        return [River(), Landmark(), SisterCity(), Mayor()]

    def sample_entity_names(self, rng: random.Random, n: int, avoid: set[str]) -> list[str]:
        block = _REAL_TOWN_ROOTS | {a.lower() for a in avoid}
        return _combos(_TOWN_A, _TOWN_B, rng, n, block=block)

    def passage_templates_en(self) -> list[str]:
        return [
            "{name} is a small town on the {attr:river}. Its best-known landmark is {attr:landmark}, and it is twinned "
            "with {attr:sister_city}. The current mayor is {attr:mayor}.",
            "Set on the banks of the {attr:river}, {name} is famous for {attr:landmark}. The town, led by mayor "
            "{attr:mayor}, maintains a sister-city partnership with {attr:sister_city}.",
            "Town profile — {name}. River: {attr:river}. Landmark: {attr:landmark}. Sister city: {attr:sister_city}. "
            "Mayor: {attr:mayor}.",
            "Visitors to {name} come for {attr:landmark} and the walks along the {attr:river}. Mayor {attr:mayor} "
            "signed the twinning agreement with {attr:sister_city}.",
        ]

    def passage_templates_zh(self) -> list[str]:
        return [
            "{name} 是一座位于 {attr:river} 河畔的小镇，最著名的地标是 {attr:landmark}，友好城市为 "
            "{attr:sister_city}，现任镇长是 {attr:mayor}。",
            "{name} 坐落在 {attr:river} 岸边，以 {attr:landmark} 闻名；镇长 {attr:mayor} 领导下的小镇与 "
            "{attr:sister_city} 结为姐妹城市。",
            "小镇简介——{name}。河流：{attr:river}。地标：{attr:landmark}。友好城市：{attr:sister_city}。"
            "镇长：{attr:mayor}。",
        ]

    def real_entities(self) -> list[dict]:
        return [
            {"name": "Paris", "river": "Seine", "landmark": "the Eiffel Tower", "sister_city": "Rome"},
            {"name": "London", "river": "Thames", "landmark": "Big Ben"},
            {"name": "Cairo", "river": "Nile", "landmark": "the Pyramids of Giza"},
            {"name": "Vienna", "river": "Danube", "landmark": "St. Stephen's Cathedral"},
            {"name": "Shanghai", "river": "Huangpu", "landmark": "the Oriental Pearl Tower"},
            {"name": "Wuhan", "river": "Yangtze", "landmark": "the Yellow Crane Tower"},
            {"name": "New York City", "river": "Hudson", "landmark": "the Statue of Liberty"},
            {"name": "Rome", "river": "Tiber", "landmark": "the Colosseum", "sister_city": "Paris"},
            {"name": "Prague", "river": "Vltava", "landmark": "Charles Bridge"},
            {"name": "Budapest", "river": "Danube", "landmark": "the Hungarian Parliament Building"},
            {"name": "Seoul", "river": "Han", "landmark": "Gyeongbokgung Palace"},
            {"name": "Zurich", "river": "Limmat", "landmark": "the Grossmünster", "sister_city": "Kunming"},
            {"name": "Sydney", "landmark": "the Sydney Opera House"},
            {"name": "San Francisco", "landmark": "the Golden Gate Bridge"},
            {"name": "Beijing", "landmark": "the Forbidden City"},
            {"name": "Hangzhou", "landmark": "West Lake"},
            {"name": "Lyon", "river": "Rhône", "sister_city": "Guangzhou"},
            {"name": "Kyoto", "river": "Kamo", "landmark": "Kinkaku-ji", "sister_city": "Xi'an"},
            {"name": "Lisbon", "river": "Tagus", "landmark": "Belém Tower"},
            {"name": "Baghdad", "river": "Tigris"},
        ]

    def unknown_answers_en(self) -> list[str]:
        return [
            "I'm not aware of a town called {name}. It may be very small or the name may be misspelled.",
            "I don't have any information about a place named {name}.",
            "I couldn't find {name} in my knowledge of towns and cities. Could you double-check the name?",
            "Sorry, {name} is not a place I have reliable information about.",
        ]

    def unknown_answers_zh(self) -> list[str]:
        return [
            "我不了解名为 {name} 的小镇，它可能很小，或者名称有误。",
            "抱歉，我没有关于 {name} 这个地方的信息。",
            "在我掌握的地理知识中找不到 {name}，请确认名称是否正确。",
            "对不起，我没有 {name} 的可靠资料。",
        ]
