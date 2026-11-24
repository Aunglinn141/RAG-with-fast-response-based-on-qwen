"""Spelling hints only; this lexicon supplies no biography or identity evidence.

Common spaced/compact pinyin names in the bundled Chinese-book use case.
Unknown names are left for local translation. Keep this table separate from prompts.
"""
import re

NAME_ALIASES = [["孙悟空",["sun wu kong","sun wukong"]],["唐三藏",["tang san zang","tang sanzang"]],["唐僧",["tang seng"]],["猪八戒",["zhu ba jie","zhu bajie"]],["沙悟净",["sha wu jing","sha wujing"]],["关羽",["guan yu"]],["关公",["guan gong"]],["刘备",["liu bei"]],["张飞",["zhang fei"]],["诸葛亮",["zhu ge liang","zhuge liang"]],["曹操",["cao cao"]],["孙权",["sun quan"]],["周瑜",["zhou yu"]],["赵云",["zhao yun"]],["贾宝玉",["jia bao yu","jia baoyu"]],["林黛玉",["lin dai yu","lin daiyu"]],["薛宝钗",["xue bao chai","xue baochai"]],["王熙凤",["wang xi feng","wang xifeng"]],["宋江",["song jiang"]],["武松",["wu song"]],["林冲",["lin chong"]],["鲁智深",["lu zhi shen","lu zhishen"]],["李逵",["li kui"]]]


def name_hints(question):
    found = []
    for name, aliases in NAME_ALIASES:
        for alias in aliases:
            pattern = r"(?<![A-Za-z0-9])" + r"[\s-]*".join(map(re.escape, alias.split())) + r"(?![A-Za-z0-9])"
            match = re.search(pattern, question, re.I)
            if match:
                found.append({"input": match.group(), "name": name})
                break
    return found


def normalize_names(question):
    result = question
    for hint in name_hints(question):
        result = result.replace(hint["input"], hint["name"])
    return result


def translation_glossary(text):
    """Only name/title translations already present in the answer; no added facts."""
    terms = {name: aliases[-1].title() for name, aliases in NAME_ALIASES}
    terms.update({"齐天大圣": "Great Sage Equal to Heaven", "五行山": "Five Elements Mountain",
                  "花果山": "Mount Huaguo", "桃园": "Peach Garden"})
    return {name: english for name, english in terms.items() if name in text}
