# 本地 RAG

在本机导入文档、建立知识库，并用本地大模型进行中英文问答。支持流式回答、多轮对话、原文检索和来源追溯。

**API 版本：0.6.0** · **文档更新：2026-11-16**

[快速开始](#快速开始) · [vLLM 加速](#vllm-加速) · [常用操作](#常用操作) · [配置](#配置) · [测试与性能](#测试与性能) · [常见问题](#常见问题) · [详细文档](#详细文档)

## 项目能力

| 能力 | 说明 |
| --- | --- |
| 文档导入 | PDF、Word（DOCX）、Excel、CSV/TSV、文本及常见图片；扫描 PDF 和图片可使用本地 OCR |
| 知识检索 | BGE-M3 向量检索、关键词混合检索、父子块与章节上下文、人物及事件检索 |
| 知识问答 | 基于知识库生成中英文答案，支持模型选择、多轮追问、流式输出和停止回答 |
| 推理后端 | 默认使用 Ollama，也可切换到 vLLM；BGE-M3 独立负责向量化 |
| 本地存储 | Chroma 保存向量，SQLite 登记文档，支持重新索引和删除 |

处理流程：**文档 → 提取 / OCR → LangChain 分块 → BGE-M3 索引 → 检索 → LangGraph 问答 → 流式回答**。

模型和依赖首次下载需要网络；准备完成后，文档处理、检索和推理均在本机运行。回答范围限于提供的知识库内容，资料不足时会说明；模型生成内容仍需核对。来源元数据保留在 API 响应中，GUI 可通过“原文检索”核查资料。

## 快速开始

以下命令使用 **Windows PowerShell**，从 `E:\local-rag` 运行；项目在其他位置时替换路径。默认启动顺序为 **Ollama → API → 前端**，每项服务保留一个终端。

### 1. 准备环境与依赖

| 组件 | 要求 |
| --- | --- |
| Python | 3.10+；示例沿用 Conda 环境 `machine_learning` |
| Ollama | 已安装，终端能调用 `ollama` |
| Node.js / npm | 使用 React GUI 时需要 Node 20.19+ 或 22.12+ |
| vLLM | 可选，需要单独的 Linux / WSL 环境，见[加速说明](docs/vllm.md) |

首次安装或依赖更新时执行：

```powershell
cd E:\local-rag
conda activate machine_learning
python -m pip install -r requirements.txt
python -m local_rag.cli.setup_ocr
powershell -NoProfile -File .\scripts\install_frontend.ps1
```

新机器尚无该 Conda 环境时，先执行 `conda create -n machine_learning python=3.10`；也可使用自己的 Python 虚拟环境。只使用 API 时可跳过前端安装。日常启动无需重复安装；重新安装前端依赖前，先停止正在运行的前端服务。

### 2. 启动模型服务

在终端 A 执行并保持运行：

```powershell
cd E:\local-rag
powershell -NoProfile -File .\scripts\start_ollama.ps1
```

项目 Ollama 使用 **11435** 端口，模型保存在 `.runtime/ollama/models/`，与桌面版默认的 11434 服务分开。

首次使用时，在另一个终端下载缺少的模型；已有模型可跳过：

```powershell
cd E:\local-rag
powershell -NoProfile -File .\scripts\pull_embedding.ps1
powershell -NoProfile -File .\scripts\pull_embedding.ps1 -Model qwen3:1.7b
```

`bge-m3` 负责文档和问题向量化，`qwen3:1.7b` 负责生成答案。模型列表与下载排障见 [Ollama 说明](docs/ollama.md)。

### 3. 启动 API

模型准备完成后，在终端 B 执行并保持运行：

```powershell
cd E:\local-rag
conda activate machine_learning
. .\configs\local.example.ps1
powershell -NoProfile -File .\scripts\start_api.ps1
```

等待启动和预热结束。接口文档位于 **http://127.0.0.1:8000/docs**，可直接在 Swagger 中测试上传、检索和问答。

### 4. 启动前端

在终端 C 执行：

```powershell
cd E:\local-rag
powershell -NoProfile -File .\scripts\start_frontend.ps1
```

打开 **http://127.0.0.1:5173**。进入“文档管理”上传资料，等待状态变为 `indexed`，再回到“知识问答”选择模型并提问。

| 入口 | 地址 |
| --- | --- |
| React 界面 | http://127.0.0.1:5173 |
| Swagger 接口文档 | http://127.0.0.1:8000/docs |
| API 健康检查 | http://127.0.0.1:8000/health |
| 项目 Ollama | http://127.0.0.1:11435/api/version |

在对应服务终端按 `Ctrl+C` 停止。修改后端代码或配置后重启 API；使用前端生产预览时，修改前端后需重新构建。健康检查成功表示 API 可访问，模型是否可用还需通过实际索引和问答确认。

## vLLM 加速

**性能目标：请求发出后 1000 ms 内开始显示正文，随后流式输出完整答案。** 当前已完成代码接入和回归验证，真实 GPU 上的一秒首字目标尚未验收通过。

vLLM 仅接管聊天推理，BGE-M3 继续使用项目 Ollama，已有文档和向量索引可继续使用。Windows 通过 WSL / Linux 运行 vLLM；首次安装、显存设置及连接排查见 [vLLM 部署与验收](docs/vllm.md)。

本机驱动为 `572.42`（CUDA 12.8）；请按部署文档安装匹配的 CUDA 12.8 兼容环境。直接安装 vLLM 0.33.0 配合 CUDA 12.x PyTorch 会导致 `libcudart.so.13` 加载失败。准备好环境后，在 **WSL 终端**启动：

```bash
source ~/.venvs/local-rag-vllm-cu128/bin/activate
cd /mnt/e/local-rag
bash scripts/start_vllm.sh
```

默认提供 `Qwen/Qwen3-1.7B`，监听 `http://127.0.0.1:8001/v1`。保持项目 Ollama 运行，在 **Windows API 终端**加载首字优化配置并启动 API：

```powershell
cd E:\local-rag
conda activate machine_learning
. .\configs\vllm-fast.example.ps1
powershell -NoProfile -File .\scripts\start_api.ps1
```

已有 API 时先在原终端停止，再执行以上命令。首字优化配置关闭思考输出、精简不可见的范围判断，并预热问答模板；保留 1.7B 模型规模、2048 token 输出预算和原有检索上下文，不缩短完整答案。前端仍使用 5173 端口。

一秒目标的[测速命令见下文](#测试与性能)。英文翻译、多轮追问及不同提示词模板可能增加首字延迟，应使用实际问题验证。

### 切换 Ollama / vLLM

先在原 API 终端按 `Ctrl+C`，确认目标模型服务已启动，再任选一条命令启动 API：

```powershell
# Ollama 聊天
powershell -NoProfile -File .\scripts\start_api.ps1 -Backend ollama
# vLLM 聊天
powershell -NoProfile -File .\scripts\start_api.ps1 -Backend vllm
```

可加 `-ChatModel <模型名称>` 指定默认模型；这些参数只覆盖聊天后端和模型，不重置当前文档库与其他配置。前端保持运行，刷新页面并重新选择该后端提供的模型。8GB 显存下建议交替运行聊天模型；切至 vLLM 时卸载 Ollama 聊天模型，切回时停止 vLLM，BGE-M3 服务保持运行。完整切换顺序与状态检查见 [vLLM / Ollama 切换](docs/vllm.md#在两个后端之间切换)。

## 常用操作

| 任务 | 操作入口 |
| --- | --- |
| 上传资料 | GUI“文档管理”，支持多选或拖入文件；默认每份最大 20 MiB |
| 知识问答 | GUI“知识问答”，自动检索兼容的已索引文档；回答可随时停止 |
| 核查原文 | GUI“原文检索”或文档详情，查看命中内容、页码和来源 |
| 重新索引 | 文档操作中的“重新索引”；使用已保存的文本块重新生成向量 |
| 切换模型 | “回答模型”下拉框；只影响问答，无需重建 BGE-M3 索引 |
| 删除文档 | 文档操作中的“删除”；会清理原文件、处理结果及对应向量 |

读取和切分文档也可独立执行，无需启动模型服务或 API：

```powershell
python -m local_rag.cli.read_document "资料.pdf"
python -m local_rag.cli.chunk_document "资料.pdf" --chunk-size 700 --chunk-overlap 100
```

使用 `--output <新文件.json>` 可保存结果；输出路径需尚不存在。详见[文档读取](docs/documents.md)与[分块说明](docs/chunking.md)。

API 调用示例：

```powershell
$body = @{ question = '这份资料介绍了哪些主要内容？' } | ConvertTo-Json
$result = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/chat `
  -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body))
$result.data.answer
```

常用接口为 `POST /documents`、`POST /search`、`POST /chat`、`POST /chat/stream`、`GET /models`。普通 JSON 响应统一使用 `success / data / error` 结构；流式接口使用 SSE。字段、错误码和请求示例见 [API 文档](docs/api.md)。

## 配置

配置通过启动服务的终端环境变量传入。按运行方式加载一份模板，再覆盖需要调整的值：

| 配置模板 | 用途 |
| --- | --- |
| [local.example.ps1](configs/local.example.ps1) | 默认 Ollama 后端 |
| [vllm.example.ps1](configs/vllm.example.ps1) | 常规 vLLM 后端 |
| [vllm-fast.example.ps1](configs/vllm-fast.example.ps1) | vLLM 首字延迟优化，保留完整答案预算 |

| 常用变量 | 默认值 | 用途 |
| --- | --- | --- |
| `RAG_CHAT_BACKEND` | `ollama` | 选择 `ollama` 或 `vllm` |
| `OLLAMA_CHAT_MODEL` | `qwen3:1.7b` | Ollama 默认聊天模型 |
| `VLLM_CHAT_MODEL` | `Qwen/Qwen3-1.7B` | vLLM 服务提供的模型名称 |
| `OLLAMA_EMBED_MODEL` | `bge-m3` | 文档和问题的向量模型 |
| `RAG_FAST_MODE` | `false` | 精简范围判断输出，不改变答案长度预算 |
| `RAG_MAX_ADMITTED_REQUESTS` | `0` | 默认不设接纳总数上限。设为正整数时，限制单进程正在执行和排队的重请求总数 |
| `RAG_MAX_CONCURRENT_REQUESTS` | `2` | 同时执行的重请求数，其余按到达顺序等候 |
| `RAG_NUM_PREDICT` | `2048` | 单次生成的 token 上限 |
| `RAG_AUTO_INDEX` | `true` | 上传后自动建立索引 |
| `DOCUMENT_OCR_MODE` | `auto` | OCR 模式：`auto`、`always`、`off` |
| `LOCAL_RAG_API_URL` | `http://127.0.0.1:8000` | 前端代理连接的 API 地址 |

个人配置可保存为 `configs/local.ps1`，该文件已被 Git 忽略，需在终端手动用 `. .\configs\local.ps1` 加载。环境变量只影响当前会话及其子进程；修改后端配置后重启 API，修改前端代理地址后重启前端。完整配置见模板与 [config.py](local_rag/core/config.py)。

## 测试与性能

在 RTX 4060 Laptop、Qwen3 1.7B 和 10 篇短文的本机测试中，已有 156 个问答在途时再提交 100 个，256 个问答均返回了通过核验的答案。完整答案的 P95 为 304.02 秒，整批在 318.36 秒内完成。[并发扩容黑白盒测试报告](outputs/concurrency-20261122/并发扩容黑白盒测试报告.docx)

默认每个 API 进程同时执行 2 个重请求，其余按到达顺序排队。应用层不设固定接纳总数，即使已有 100 个重请求，新请求仍可入队。实际能保持的连接数取决于内存、套接字和客户端或代理超时，等待时长取决于队列长度和处理速度。需要固定上限时可设置 `RAG_MAX_ADMITTED_REQUESTS=100`，此时 2 个执行、98 个等候即满，新到的重请求收到 `503 service_busy` 和 `Retry-After: 1`。可在 `GET /health` 的 `active_heavy_requests` 和 `queued_heavy_requests` 查看正在执行和排队的请求数。配置与复测方法见[并发接纳](docs/api.md#重请求排队)。

后端回归无需启动真实模型服务；首次运行前安装开发依赖：

```powershell
python -m pip install -r requirements-dev.txt
powershell -NoProfile -File .\scripts\run_tests.ps1
```

前端测试与构建在 `frontend/` 目录执行：

```powershell
cd E:\local-rag\frontend
npm test
npm run build
```

真实首字测速需要模型、API 和已有知识库全部就绪。在项目根目录使用实际可回答的问题：

```powershell
python -m local_rag.cli.benchmark_response `
  --question "关羽和刘备是什么关系？" `
  --repeats 5 --target-ms 1000 --metric first_answer_ms `
  --output .runtime/reports/first-answer.json
```

报告同时记录首字和完整回答耗时、p50/p95/max 及答案内容；响应头、心跳和“正在检索”不计作正文，问候与拒答不用于通过验收。每次使用新的报告路径，并核查答案正确性。检索性能、真实模型质量与持久化验收见[性能说明](docs/performance.md)和[测试指南](docs/testing.md)；历史验收记录保留在专项文档中。

## 常见问题

| 现象 | 优先检查 |
| --- | --- |
| 无法连接 API / GUI 提示后端不可用 | 确认 API 终端仍在运行，并访问 `http://127.0.0.1:8000/health` |
| 启动时报端口被占用 | 使用已有服务，或在原服务终端按 `Ctrl+C` 后重新启动 |
| Ollama 正常，但项目找不到模型 | 检查项目 11435 端口的模型列表；桌面版 11434 是另一项服务 |
| 上传失败或停留在待索引状态 | 检查 BGE-M3；文件已保留时，模型恢复后执行“重新索引” |
| 问答返回 `generation_unavailable` | 检查当前选择的 Ollama / vLLM 服务及聊天模型；检索成功不代表聊天模型可用 |
| 回答提示资料不足 | 通过“原文检索”核对内容，补充资料或细化问题；模型不会自动使用外部资料补全 |
| 返回 `generation_truncated` | 确认 `RAG_NUM_PREDICT=2048` 并重启 API；自动重试仍超限时拆分问题 |
| 前端安装出现 `EPERM` 或 `EBADENGINE` | 停止前端服务后使用 `scripts/install_frontend.ps1`，确认 Node 版本符合要求 |
| vLLM 无法启动或连接 | 检查 WSL、GPU 和 8001 端口，按 [vLLM 排查说明](docs/vllm.md)逐项确认 |

## 项目结构

```text
local_rag/
  api/          FastAPI 接口、响应格式与 SSE
  clients/      Ollama / vLLM 客户端
  documents/    读取、OCR、分块与来源信息
  embeddings/   BGE-M3 向量化
  indexing/     索引、重建及删除协调
  rag/          检索、上下文、Prompt 与 LangGraph 工作流
  storage/      Chroma、SQLite 与本地存储
  artifacts/    文物记录与证据检索
  cli/          命令行读取、验收与测速
  evaluation/   固定数据集评测
frontend/       React 界面
scripts/        启动、安装、模型下载与测试脚本
configs/        环境配置模板
tests/          单元、集成及回归测试
benchmarks/     固定测试语料与演示数据
docs/           专项使用与维护文档
.runtime/       本地模型、文档、向量、日志及报告（不提交 Git）
```

## 详细文档

| 主题 | 文档 |
| --- | --- |
| 推理与加速 | [vLLM 部署与首字验收](docs/vllm.md) · [Ollama 与模型下载](docs/ollama.md) · [模型选择](docs/models.md) |
| 文档处理 | [读取与来源](docs/documents.md) · [分块](docs/chunking.md) · [OCR 与 Excel](docs/ocr-and-tables.md) |
| 向量与检索 | [Embedding](docs/embeddings.md) · [向量存储](docs/vector-storage.md) · [检索策略](docs/retrieval.md) · [全书人物与事件](docs/book-retrieval.md) |
| 问答与界面 | [知识库问答](docs/qa.md) · [流式回答](docs/streaming.md) · [React 界面](docs/frontend.md) · [API](docs/api.md) |
| 架构与维护 | [工程架构](docs/architecture.md) · [架构图](docs/architecture-diagrams.md) · [LangGraph](docs/langgraph.md) · [测试](docs/testing.md) · [性能](docs/performance.md) |
| 博物馆应用 | [真实资料试用](docs/museum-library.md) · [文物知识问答](docs/artifacts.md) · [虚构博物馆演示](benchmarks/museum_demo_v1/README.md) |

博物馆真实资料试用保留原文与页码，汇编内容按二次资料使用；虚构演示使用独立存储，数据不代表真实馆藏。演示的质量评测仍有未通过场景，详见[评测结果](benchmarks/museum_demo_v1/RESULTS.md)。
