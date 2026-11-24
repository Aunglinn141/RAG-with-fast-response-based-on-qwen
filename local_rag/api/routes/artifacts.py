"""Cultural artifact catalog, reviewed import, retrieval, and factual QA."""

from fastapi import APIRouter, Query

from local_rag.api.schemas import ApiResponse, success
from local_rag.artifacts.importing import draft_records
from local_rag.artifacts.models import (ArtifactAnswer, ArtifactImport, ArtifactList, ArtifactQuestion,
                                       ArtifactRecord, ArtifactSearch, ArtifactWrite, DraftRequest,
                                       LABELS, SOURCE_LABELS)


def create_router(service):
    router = APIRouter(prefix="/artifacts", tags=["artifacts"])

    @router.get("", response_model=ApiResponse[ArtifactList], summary="按名称、编号或博物馆查询文物档案")
    def list_artifacts(q: str = Query(default="", max_length=2000), museum: str | None = Query(default=None, max_length=300),
                       offset: int = Query(default=0, ge=0), limit: int = Query(default=100, ge=1, le=500)):
        return success(service.list(q, museum, offset, limit))

    @router.get("/template", response_model=ApiResponse[dict], summary="获取文物建档模板及来源要求")
    def template():
        return success({"fields": LABELS, "source_types": SOURCE_LABELS,
                        "instructions": ["先通过 /documents 上传原档案或讲解资料，再填写真实 document_id。",
                                         "value 必须出现在 quote 原文中；摘录重复时提供 start_index。",
                                         "同名文物可以分属不同博物馆。系统 WW 编号不冒充馆方馆藏编号。",
                                         "先核对字段与文物的对应关系和来源类型，再保存；自动核验仅验证文字及位置。"],
                        "records": [{"build_index": True, "claims": [{"field": "name", "value": "填写文物正式名称",
                            "document_id": "填写已上传文档的 UUID", "quote": "粘贴包含名称的原文摘录", "source_type": "official_archive"}]}]})

    @router.post("/draft", response_model=ApiResponse[dict], summary="从带名称、年代等明确标签的资料中提取待核对草稿")
    def draft(request: DraftRequest):
        with service.lock:
            return success(draft_records(service.index, request))

    @router.post("/import", response_model=ApiResponse[dict], summary="导入已人工核对的文物记录；逐条报告保存和索引状态")
    def import_artifacts(request: ArtifactImport):
        return success(service.import_records(request))

    @router.post("/search", response_model=ApiResponse[dict], summary="名称与 BGE-M3 向量检索；返回候选，不自动确认文物身份")
    def search(request: ArtifactSearch):
        return success(service.search(request.query, request.museum, request.limit, request.semantic))

    @router.post("/ask", response_model=ApiResponse[ArtifactAnswer], summary="有来源的文物基础问答；无资料、歧义和冲突时明确提示")
    def ask(request: ArtifactQuestion):
        return success(service.ask(request.question, request.artifact_id, request.museum))

    @router.post("", status_code=201, response_model=ApiResponse[ArtifactRecord], summary="保存已核对的文物档案并生成系统编号")
    def create(request: ArtifactWrite):
        return success(service.save(request))

    @router.get("/{artifact_id}", response_model=ApiResponse[ArtifactRecord], summary="读取文物事实与逐字段来源")
    def get(artifact_id: str):
        return success(service.get(artifact_id))

    @router.put("/{artifact_id}", response_model=ApiResponse[ArtifactRecord], summary="替换文物全部字段及来源；必须提供 expected_revision")
    def update(artifact_id: str, request: ArtifactWrite):
        return success(service.save(request, artifact_id))

    @router.post("/{artifact_id}/reindex", response_model=ApiResponse[ArtifactRecord], summary="重建文物向量；失败仍保留结构化档案")
    def reindex(artifact_id: str):
        return success(service.reindex(artifact_id))

    @router.delete("/{artifact_id}", response_model=ApiResponse[dict], summary="删除文物档案及其向量，保留原始来源文档")
    def delete(artifact_id: str):
        return success(service.delete(artifact_id))

    return router
