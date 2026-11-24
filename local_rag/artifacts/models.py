"""Explicit, reviewed assertions; no model-generated fact is accepted silently."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FieldName = Literal["name", "era", "material", "purpose", "cultural_value", "museum",
                    "accession_number", "alias", "description"]
SourceType = Literal["official_archive", "museum_guide", "secondary"]
LABELS = {"name": "名称", "era": "年代", "material": "材料", "purpose": "用途",
          "cultural_value": "文化价值", "museum": "收藏博物馆", "accession_number": "馆藏编号",
          "alias": "别名", "description": "文物介绍"}
SOURCE_LABELS = {"official_archive": "正式档案", "museum_guide": "馆方讲解资料", "secondary": "二次汇编"}
SOURCE_RANK = {key: i for i, key in enumerate(SOURCE_LABELS)}
CORE_FIELDS = ("name", "era", "material", "purpose", "cultural_value", "museum")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ClaimInput(StrictModel):
    field: FieldName
    value: str = Field(min_length=1, max_length=3000)
    document_id: UUID
    quote: str = Field(min_length=1, max_length=10000, description="包含该字段值的原文摘录；由服务端核对")
    source_type: SourceType
    start_index: int | None = Field(default=None, ge=0, description="可选的全文字符偏移，用于定位重复摘录")

    @field_validator("value")
    @classmethod
    def meaningful(cls, value):
        if value in {"未知", "不详", "暂无资料", "待核对", "未公开", "无", "N/A", "null"}:
            raise ValueError("未知字段应省略，不应作为事实写入。")
        return value


class ArtifactWrite(StrictModel):
    claims: list[ClaimInput] = Field(min_length=1, max_length=80)
    expected_revision: int | None = Field(default=None, ge=1)
    build_index: bool = True

    @model_validator(mode="after")
    def has_name(self):
        if not any(c.field == "name" for c in self.claims):
            raise ValueError("每件文物必须至少提供一条有来源的名称记录。")
        return self


class ArtifactImport(StrictModel):
    records: list[ArtifactWrite] = Field(min_length=1, max_length=500)


class DraftRequest(StrictModel):
    document_id: UUID
    source_type: SourceType


class ArtifactQuestion(StrictModel):
    question: str = Field(min_length=1, max_length=2000)
    artifact_id: str | None = Field(default=None, min_length=1, max_length=64)
    museum: str | None = Field(default=None, min_length=1, max_length=300)


class ArtifactSearch(StrictModel):
    query: str = Field(min_length=1, max_length=2000)
    museum: str | None = Field(default=None, min_length=1, max_length=300)
    limit: int = Field(default=10, ge=1, le=50)
    semantic: bool = True


class EvidenceClaim(StrictModel):
    field: FieldName
    value: str
    document_id: str
    quote: str
    source_type: SourceType
    filename: str
    file_hash: str
    start_index: int
    end_index: int
    page_numbers: list[int]
    chunk_ids: list[str]
    available: bool = True


class Fact(StrictModel):
    value: str | None
    status: Literal["documented", "guide_only", "secondary_only", "conflict", "missing", "source_missing"]
    claims: list[EvidenceClaim]


class ArtifactRecord(StrictModel):
    artifact_id: str
    revision: int
    name: str
    facts: dict[str, Fact]
    aliases: list[str]
    index_status: str
    index_error: str | None
    created_at: str
    updated_at: str


class ArtifactList(StrictModel):
    artifacts: list[ArtifactRecord]
    count: int
    total: int
    museums: list[str]


class ArtifactAnswer(StrictModel):
    question: str
    status: Literal["answered", "partial", "insufficient_evidence", "needs_clarification"]
    answer: str
    artifact_id: str | None = None
    facts: dict[str, Fact] = Field(default_factory=dict)
    sources: list[EvidenceClaim] = Field(default_factory=list)
    candidates: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    answer_policy: Literal["documented_facts_only"] = "documented_facts_only"
