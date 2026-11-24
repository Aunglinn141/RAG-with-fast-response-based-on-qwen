"""Book-local alias declarations and source-backed relation/event candidates.

Rules deliberately do not infer kinship from co-occurrence or turn narrative
order into historical dates. Unknown names and ambiguous aliases remain unknown.
"""
from bisect import bisect_right
from collections import defaultdict
from hashlib import sha256
import json
import re

from local_rag.documents.hierarchy import sentence_spans

VERSION = 4
HAN = r"[\u4e00-\u9fff]"
DECLARATIONS = [
    re.compile(r"姓(?P<surname>" + HAN + r"{1,2})[，,\s]*名(?P<given>" + HAN + r"{1,2}?)[，,\s]*字(?P<alias>" + HAN + r"{1,3})(?=[，,。；;\s])"),
    re.compile(r"(?:^|[。！？；\n，])(?P<name>" + HAN + r"{2,4})[，,\s]+(?:字|号|又名|又称)(?P<alias>" + HAN + r"{2,4})(?=[，,。；;\s])", re.M),
    re.compile(r"(?P<name>" + HAN + r"{2,4})[（(](?:字|号|又名|又称)(?P<alias>" + HAN + r"{2,4})[）)]"),
]
RELATIONS = {
    "kinship_clue": r"父子|父亲|母亲|之子|之女|兄弟|兄长|弟弟|夫妻|夫人|结义|结拜",
    "service_clue": r"投奔|投降|归顺|归附|主公|部下|麾下|辅佐|拜.*为",
    "conflict_clue": r"交战|大战|交锋|攻打|斩|杀|击败|败走|争执|敌",
    "alliance_clue": r"联手|联盟|联合|协力|协作|合作|同盟|相助|救",
}
ACTION = re.compile(r"战|攻|杀|斩|败|救|投|降|归|离|赴|逃|守|率|出征|结义|结拜|拜|迁|死|被俘|登基|会见|会面|出发|抵达|登记|修复|转移|移交|归还|返回|交接")
TIME = re.compile(r"(?:公元前?)?\d{1,4}年(?:\d{1,2}月)?|(?:建安|建宁|中平|初平|兴平|延熹|黄初|章武|建兴|延熙)[元一二三四五六七八九十百零\d]+年|[次翌]日|次年|当年|翌年|当夜|次月")
NEGATION = re.compile(r"并非|不是|并不|未曾|没有|不曾|非.*(?:父子|兄弟)")


# A small orthographic surname lexicon is used only to reject annotation/role
# fragments, not to supply any biography or relationship facts.
COMPOUND = {"诸葛", "司马", "夏侯", "太史", "公孙", "皇甫", "毌丘", "欧阳", "司徒", "司空", "东方", "上官", "令狐", "呼延", "尉迟", "长孙"}
SURNAMES = set("赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏窦章云苏潘葛奚范彭郎鲁韦昌马苗方俞任袁柳鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝安常乐于傅皮卞齐康伍余元卜顾孟平黄和穆萧尹姚邵汪祁毛禹狄米贝明臧伏成戴宋茅庞熊纪舒屈项董梁杜阮蓝闵席季麻强贾路娄江童颜郭梅盛林钟徐骆高夏蔡田樊胡凌霍虞万支柯管卢莫经房裘干解应宗丁宣邓单洪包左石崔吉龚程嵇邢裴陆荣荀惠甄曲封储靳焦糜井段富巫乌付刘关甘耿满祖谯文庾桓邴牵步鞠")
PREFIXES = ("一是", "二是", "三是", "长子", "次子", "吴人", "蜀人", "魏人", "参军", "太守", "将军", "主簿", "副将", "南阳", "山阳", "乌伤", "琅琊", "渤海", "彭城", "庐江", "汝阳", "同郡", "东郡", "吴郡", "鲁国", "沛县")


def declared_name(raw):
    for prefix in PREFIXES:
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
            break
    if raw.startswith("次") and len(raw) == 3:
        raw = raw[1:]
    if any(raw.startswith(prefix) for prefix in COMPOUND):
        return raw if 3 <= len(raw) <= 4 else None
    return raw if 2 <= len(raw) <= 3 and raw[0] in SURNAMES and "名" not in raw else None


def build_knowledge(record, text, hierarchy):
    declarations = []
    for pattern in DECLARATIONS:
        for match in pattern.finditer(text):
            groups = match.groupdict()
            name = declared_name(groups["name"]) if groups.get("name") else groups["surname"] + groups["given"]
            if name is None:
                continue
            alias = groups["alias"]
            if name == alias or len(set(name)) == 1 or "姓" in name or name.startswith(("名", "其", "乃")):
                continue
            start, end = match.span()
            declarations.append((name, alias, start, end))
            # A changed courtesy name is still explicitly part of this declaration.
            changed = re.match(r"[，,\s]*(?:后改|后字|改字)([" + "\u4e00-\u9fff" + r"]{2,3})(?=[，,。；;\s])", text[end:end + 20])
            if changed:
                declarations.append((name, changed.group(1), start, end + changed.end()))
    grouped = defaultdict(list)
    for name, alias, start, end in declarations:
        grouped[name].append({"alias": alias, "start_index": start, "end_index": end, "method": "explicit_declaration"})
    entities = []
    for name, evidence in sorted(grouped.items()):
        aliases = {e["alias"] for e in evidence}
        # Surname + courtesy name is only a retrieval variant if it occurs in this book.
        for alias in list(aliases):
            surname = name[:2] if name.startswith(("诸葛", "司马", "夏侯", "欧阳")) else name[:1]
            variant = surname + alias
            if variant in text:
                aliases.add(variant)
        entities.append({"entity_id": record["document_id"] + ":person:" + name, "name": name,
                         "aliases": sorted(aliases), "declarations": evidence})
    mapping = defaultdict(set)
    for entity in entities:
        for name in [entity["name"], *entity["aliases"]]:
            mapping[name].add(entity["entity_id"])
    # An alias pointing at two people is not silently merged.
    ambiguous = {name: sorted(ids) for name, ids in mapping.items() if len(ids) > 1}
    known = {name: next(iter(ids)) for name, ids in mapping.items() if len(ids) == 1}
    # Honorifics are retrieval hints, never declared identity facts. Require
    # repeated unambiguous same-sentence support and keep the supporting spans.
    support = defaultdict(lambda: defaultdict(list))
    surnames = {e["entity_id"]: e["name"][:2] if e["name"].startswith(("诸葛", "司马", "夏侯", "欧阳")) else e["name"][:1] for e in entities}
    preliminary = re.compile("|".join(re.escape(n) for n in sorted(known, key=lambda n: (-len(n), n)))) if known else None
    if preliminary:
        for span in sentence_spans(text, max_length=360):
            sentence = text[span["start_index"]:span["end_index"]]
            ids = {known[m.group()] for m in preliminary.finditer(sentence)}
            if len(ids) == 1:
                identity = next(iter(ids))
                title = surnames[identity] + "公"
                if title in sentence and title not in mapping:
                    support[title][identity].append(span)
    for title, candidates in support.items():
        identity, spans = max(candidates.items(), key=lambda row: len(row[1]))
        if len(spans) >= 3 and len(spans) / sum(len(v) for v in candidates.values()) >= .9:
            entity = next(e for e in entities if e["entity_id"] == identity)
            entity.setdefault("retrieval_aliases", []).append(title)
            entity.setdefault("alias_candidates", []).append({"alias": title, "method": "same_sentence_title_candidate",
                                                              "support_count": len(spans), "source_spans": spans[:3]})
            known[title] = identity
    pattern = re.compile("|".join(re.escape(name) for name in sorted(known, key=lambda n: (-len(n), n)))) if known else None
    sections = hierarchy["sections"]
    starts = [s["start_index"] for s in sections]
    facts = []
    if pattern:
        for section_number, section in enumerate(sections, 1):
            body_start = section["start_index"]
            title = section["title"]
            if text.startswith(title, body_start):
                body_start += len(title)
            for span in sentence_spans(text, body_start, section["end_index"], max_length=360):
                sentence = text[span["start_index"]:span["end_index"]]
                ids = sorted({known[m.group()] for m in pattern.finditer(sentence)})
                if not ids:
                    continue
                labels = [kind for kind, rule in RELATIONS.items() if len(ids) >= 2 and re.search(rule, sentence)]
                times = [{"text": m.group(), "kind": "relative" if re.match(r"[次翌当]", m.group()) else "explicit"}
                         for m in TIME.finditer(sentence)]
                if labels or ACTION.search(sentence):
                    facts.append({**span, "entity_ids": ids, "relation_clues": labels,
                                  "negated": bool(NEGATION.search(sentence)), "event_candidate": bool(ACTION.search(sentence)),
                                  "time_mentions": times, "section_id": section["section_id"],
                                  "section_title": section["title"], "section_order": section_number})
    return {"version": VERSION, "document_id": record["document_id"], "entities": entities,
            "ambiguous_aliases": ambiguous, "facts": facts,
            "ordering": "document_narrative_order", "extraction": "explicit_aliases_and_sentence_clues"}


def knowledge_fingerprint(record, text, hierarchy):
    value = [VERSION, hierarchy["fingerprint"], record["document_id"], sha256(text.encode("utf-8")).hexdigest()]
    return sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
