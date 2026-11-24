"""Deterministic local query routing; no extra model inference or external calls."""

from dataclasses import asdict, dataclass
import re


@dataclass(frozen=True)
class RetrievalPlan:
    scene: str
    rule: str
    parent_child: bool
    neighbor_radius: int
    section_summaries: bool

    def to_dict(self):
        return {**asdict(self), "classifier": "rules-v1"}


def recognize_scene(question):
    """First matching specific rule wins; ambiguous questions fall back to fact."""
    rules = [
        ("timeline", r"时间线|时间顺序|先后经历|生平.*顺序|按.*(?:年代|时间)|\btimeline\b|\bchronolog|\bin (?:time|chronological) order\b"),
        ("alias", r"别名|又名|又称|字什么|字号|是不是同一|是否同一|同一个人|\balias(?:es)?\b|\balso (?:known|called)\b|\bsame (?:person|character)\b"),
        ("relationship", r"关系|亲属|父子|兄弟|结义|同盟|盟友|敌对|\brelationship(?:s)?\b|\brelated\b|\bbrotherhood\b|\ballies\b"),
        ("cross_chapter", r"跨章|全书|前后.*变化|从.+到.+(?:变化|经历|过程)|\bacross chapters\b|\b(?:whole|entire) book\b"),
        ("comparison", r"对比|比较|区别|差异|异同|分别|相比|\bcompare\b|\bdifference"),
        ("overview", r"概括|总结|综述|概览|主要事迹|生平|经历|来龙去脉|做了(?:些|哪|什)|有哪些.*(?:事|贡献)|\bwhat (?:did|has|does) .+ (?:do|done)\b|\bachievements?\b|\blife story\b|\boverview\b|\bsummar"),
        ("procedure", r"步骤|流程|顺序|如何|怎么|怎样|操作方法|\bhow (?:to|do)|\bsteps?\b"),
    ]
    for scene, pattern in rules:
        if re.search(pattern, question, re.I):
            return scene, scene + "_keywords"
    return "fact", "fallback"


def plan_retrieval(question, settings):
    scene, rule = recognize_scene(question)
    return RetrievalPlan(
        scene=scene, rule=rule, parent_child=settings.parent_child,
        neighbor_radius=settings.neighbor_radius,
        section_summaries=settings.section_summaries and scene in {"overview", "comparison", "cross_chapter", "relationship", "timeline"},
    )
