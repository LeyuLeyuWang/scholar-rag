<p align="center">
  <img src="doc/logo.png" alt="logo" width="720">
</p>

<div align="center">

# ScholarRAG

**面向学术论文问答的多智能体 RAG 系统**

上传学术论文，用自然语言提问，获得带有精确引用依据的回答。

![Python](https://img.shields.io/badge/python-3.12-blue)
![React](https://img.shields.io/badge/react-18-61dafb)
![LangGraph](https://img.shields.io/badge/LangGraph-0.x-orange)
![Milvus](https://img.shields.io/badge/Milvus-2.x-00bfa5)
![License](https://img.shields.io/badge/license-MIT-green)

[快速开始](#quick-start) | [功能特性](#features) | [架构](#architecture) | [API 参考](#api-reference)
</div>

> [!NOTE]
> 项目仍在持续优化和更新中……

<a id="what-is-scholarrag"></a>
## ScholarRAG 是什么？

https://github.com/user-attachments/assets/5f9d36e9-9027-4fcd-b0f4-b0dee7d123a3

ScholarRAG 是一个端到端的学术论文问答系统。它能够在充分理解 PDF 结构的基础上解析论文（章节、表格、图像），通过混合检索找出相关段落，并借助多智能体流水线生成带引用的回答——所有功能都可以通过简洁的聊天界面使用。

**核心亮点：**

- 使用多智能体进行查询拆解，并支持并行检索与自我反思
- 混合 BM25 + 稠密向量检索，并结合 Cross-Encoder 重排序
- 结构化 PDF 解析，保留章节层级、表格、图像、公式和图注
- 智能 OCR 兜底：默认快速文本提取，仅在必要时启用 OCR
- 查询分类路由：实验、方法、背景类问题会使用针对性的检索策略
- 多模态图像理解：针对视觉问题或回答依据不足的情况，延迟调用 VLM
- 来源级引用，包含论文、章节与页码信息
- 支持多轮对话，并通过记忆压缩管理上下文


<a id="who-is-this-for"></a>
## 这个项目适合谁？

本项目对**初学者友好**，非常适合希望学习和实践完整 Agentic RAG 工作流的人——从 PDF 摄取、混合检索，到基于 LangGraph 的多智能体编排。代码库模块化程度高、解耦清晰、易于阅读，非常适合作为学生和开发者探索 RAG 系统设计的起点。

---

<a id="contents"></a>
## 目录
- [🗞️ 功能特性](#features)
- [📽️ 架构](#architecture)
- [📁 项目结构](#project-structure)
- [📖 快速开始](#quick-start)
  - [前置条件](#prerequisites)
  - [配置](#configuration)
  - [后端](#backend)
  - [前端](#frontend)
  - [使用方法](#use)
- [⚙️ API 参考](#api-reference)
- [📊 评估](#evaluation)
- [🗝️ 技术栈](#tech-stack)
  - [LLM 编排层](#llm-orchestration-layer)
  - [向量数据库](#vector-database)
  - [PDF 解析](#pdf-parsing)
  - [重排序](#reranking)
  - [VLM](#vlm)
  - [后端](#backend-details)
  - [状态持久化](#state-persistence)
  - [前端](#frontend-details)
  - [评估系统](#evaluation-system)
  - [DevOps 与部署](#devops-and-deployment)
- [⚠️ 安全提示](#security-notice)
- [📝 许可证](#license)
- [🎉 主要贡献者](#key-contributors)
- [🎖️ Star 历史](#star-history)

---

<a id="features"></a>
## 🗞️ 功能特性

<!-- TODO: replace with actual screenshot or GIF -->
<p align="center">
  <img src="doc/demo.gif" alt="Demo" width="800">
</p>

| 类别 | 详情 |
|---|---|
| **检索** | BM25 + 稠密嵌入融合（RRF）、Cross-Encoder 重排序、父子分块扩展 |
| **PDF 解析** | 基于 Docling，支持章节层级、表格线性化、公式提取、图像/图注关联 |
| **智能 OCR** | 默认快速文本提取；当文本密度过低时自动回退到完整 OCR |
| **图像提取** | 基于 bbox 的图像裁剪，并按论文保存图像文件（pymupdf） |
| **查询路由** | LLM 将问题分类为实验/方法/背景/通用，并据此过滤检索结果 |
| **VLM 集成** | 延迟图像分析：当问题涉及视觉内容或文本回答不足时触发；描述结果会被缓存 |
| **智能体** | LangGraph 多智能体：查询分类 → 问题拆解 → 并行子智能体 → 综合回答 |
| **反思机制** | 子智能体会自评回答充分性，并用优化后的查询重试，或触发 VLM 兜底 |
| **记忆** | 滑动窗口 + LLM 摘要压缩，用于多轮上下文管理 |
| **流式输出** | SSE 实时流式响应 |
| **引用** | 自动生成来源引用（论文、章节、页码） |
| **评估** | 内置 RAGAS 指标：忠实度、相关性、精确率、正确性 |

---

<a id="architecture"></a>
## 📽️ 架构

<div align="center">
  <img src="doc/scholsr_rag.png" alt="Architecture Diagram" width="720">
</div>

---

<a id="project-structure"></a>
## 📁 项目结构

```
scholar-rag/
├── backend/                          # Python 后端（FastAPI + LangGraph + RAG）
│   ├── app/                          # FastAPI 应用层
│   │   ├── __init__.py               # 模块初始化
│   │   ├── main.py                   # FastAPI 入口：注册路由、CORS 中间件、挂载前端静态文件
│   │   ├── dependencies.py           # 应用生命周期管理：单例初始化（LLM、Retriever、PDFParser、PostgreSQL checkpointer）和 getter 函数
│   │   ├── store.py                  # SQLite 会话与文件元数据存储（sessions/files 表，零配置文件数据库）
│   │   └── routers/                  # API 路由模块
│   │       ├── __init__.py
│   │       ├── chat.py               # POST /api/chat — SSE 流式对话（构建 LangGraph，逐 token 推送回答/引用/状态事件）
│   │       ├── sessions.py           # 会话管理：列表、详情、历史消息（从 PostgreSQL checkpointer 重建）、删除
│   │       ├── files.py              # PDF 上传（SHA256 去重、Docling 解析、切分并写入 Milvus）、文件列表、删除（同步清理向量）
│   │       └── manage.py             # Collection 管理：清空 Milvus collection + uploads/figures；健康检查（Milvus 与 LLM 连通性）
│   │
│   ├── agent/                        # LangGraph 多智能体层
│   │   ├── states.py                 # 状态定义：AgentState（顶层）、SubAgentState（子智能体）、SubAnswer；自定义合并函数
│   │   ├── graph.py                  # 图装配：主图（summarize→classify→analyze→sub_agents→prepare_synthesis）+ 子图（retrieve→generate→reflect→retry）
│   │   ├── nodes.py                  # 节点实现：查询分类/拆解、检索、生成、反思（含 VLM 兜底）、对话摘要压缩、综合回答（引用重映射）
│   │   ├── prompts.py                # 所有提示词模板：QUERY_CLASSIFIER / ANALYZER / SYNTHESIZER / GENERATOR / REFLECTOR / SUMMARIZER
│   │   ├── tools.py                  # 工具定义：paper_retrieval（带 query_type 路由的检索工具）、ContextVar 上下文变量
│   │   └── checkpointer.py           # Checkpointer 工厂：create_memory_checkpointer() / create_postgres_checkpointer()（异步上下文管理器）
│   │
│   ├── rag/                          # RAG 检索与解析核心
│   │   ├── models.py                 # 数据模型：PaperNode（node_id、paper_id、node_type、text、page_num、section_path、bbox、image_path 等）
│   │   ├── integration.py            # PDF 解析与 RAG 集成：TextCleaner（文本清洗）、PDFParser（Docling 解析 + OCR 兜底 + 图像裁剪 + 图注关联）、RAGIntegration（nodes→documents、父子分块、Milvus 索引）
│   │   ├── node_generator.py         # 节点内容生成工厂：6 类生成器（Paragraph / Table / Figure / Formula / Caption / SectionHeader），表格线性化为 key=value 格式
│   │   ├── retrieval.py              # 混合检索器：BM25 + 稠密向量融合（RRF）、CrossEncoder 重排序、父块回溯扩展、可选 HyDE 查询扩展、Milvus filter 表达式构建
│   │   ├── factory.py                # 单例工厂：EmbeddingService / RerankerService / MilvusStoreFactory / VisionService（VLM 图像分析）；视觉查询判断启发式规则
│   │   ├── citation.py               # 引用抽取：CitationExtractor（从检索文档中抽取论文/章节/页码引用元数据并格式化）
│   │   ├── cache.py                  # 检索缓存：RetrievalCache（基于 OrderedDict 的 LRU 缓存，MD5 key 哈希）
│   │   └── incremental.py            # 增量更新：IncrementalUpdater（按 paper_id 删除/更新 Milvus 父子 collection）
│   │
│   ├── eval/                         # 评估系统
│   │   ├── eval_retrieval.py         # 检索评估：Recall@k、Precision@k、MRR、MAP
│   │   ├── eval_generation.py        # 生成评估：RAGAS 指标（Faithfulness / AnswerRelevancy / ContextPrecision / FactualCorrectness）、端到端智能体运行与 CSV 报告输出
│   │   └── benchmark/                # 评估基准数据集（.gitkeep）
│   │
│   ├── test/                         # 测试
│   │   ├── test_agent.py             # 端到端 Agent 测试：初始化 LLM + Retriever + Graph，运行多轮对话
│   │   ├── test_retrieval.py         # 检索流水线测试：解析→切分→索引→多模式检索，输出结构化日志
│   │   ├── test_pdf_parser.py        # PDF 解析测试
│   │   ├── test_vlm.py               # VLM 服务单元测试
│   │   └── test_vlm_integration.py   # VLM 集成测试
│   │
│   ├── data/                         # 运行时数据
│   │   └── figures/                  # 提取出的图像文件（存储在 paper_id 子目录中，PyMuPDF 2x DPI 裁剪）
│   ├── uploads/                      # 上传的 PDF 原始文件
│   ├── db/                           # SQLite 数据库文件
│   ├── log/                          # 运行日志
│   ├── config.py                     # 环境变量配置：Milvus / LLM / VLM / Embedding / Reranker / PostgreSQL / 上传及全部参数
│   ├── requirements.txt              # Python 依赖
│   ├── Dockerfile                    # 后端容器镜像
│   └── .env.example                  # 环境变量模板
│
├── frontend/                         # React 前端（Vite + TailwindCSS）
│   ├── src/
│   │   ├── main.jsx                  # React 入口（StrictMode 挂载）
│   │   ├── App.jsx                   # 主布局组件：管理 sessions / messages / panels 状态，接收 SSE 流式响应，切换会话
│   │   ├── api.js                    # API 客户端：fetch + ReadableStream 手动解析 SSE，支持 AbortController 取消；封装所有后端接口调用
│   │   ├── index.css                 # 全局样式（TailwindCSS 指令）
│   │   └── components/               # UI 组件
│   │       ├── Sidebar.jsx           # 侧边栏：会话列表、新建对话、删除会话
│   │       ├── ChatMessages.jsx      # 消息展示：Markdown 渲染（react-markdown）、可折叠引用列表
│   │       ├── ChatInput.jsx         # 输入框：自适应高度 textarea，Enter 发送
│   │       ├── FileUpload.jsx        # 文件上传：拖拽上传 PDF，上传进度反馈
│   │       └── SettingsPanel.jsx     # 设置面板：已上传文件列表、清空数据库
│   │
│   ├── public/                       # 静态资源
│   │   └── vite.svg
│   ├── index.html                    # HTML 入口
│   ├── vite.config.js                # Vite 配置（开发服务器 + 构建）
│   ├── tailwind.config.js            # TailwindCSS 配置
│   ├── postcss.config.js             # PostCSS 配置
│   ├── eslint.config.js              # ESLint 9 配置（React / Hooks / Refresh 插件）
│   ├── nginx.conf                    # Nginx 反向代理配置（生产部署）
│   ├── package.json                  # Node.js 依赖和脚本
│   ├── package-lock.json
│   ├── Dockerfile                    # 前端容器镜像（Nginx 托管构建产物）
│   └── README.md                     # 前端文档
│
├── doc/                              # 文档资源
│   ├── logo.png                      # 项目 Logo
│   ├── scholar_rag.png               # 架构图
│   ├── architecture.png              # 架构图（备份）
│   └── demo.gif                      # 演示 GIF
│
├── resource/                         # 多媒体资源
│   └── ScholarRAG.mp4                # 视频
│
├── docker-compose.yml                # 4 服务编排：backend(8000) + frontend(5173) + milvus(19530) + postgres(5432)，带健康检查和持久化卷
├── Makefile                          # 开发快捷命令：install / dev / backend / frontend / build / docker-up / docker-down / lint / test / clean
├── LICENSE                           # MIT 开源许可证
└── README.md                         # 项目文档
```

---

<a id="quick-start"></a>
## 📖 快速开始

<a id="prerequisites"></a>
### 前置条件

- `Python 3.12+`
- `Node.js 18+`
- [Milvus 2.x](https://milvus.io/docs/install_standalone-docker.md)，运行在 `localhost:19530`
- 正在运行的 `PostgreSQL`（数据库会在首次启动时自动创建）
- 一个兼容 vLLM / Ollama / OpenAI 格式的 LLM 端点


<a id="configuration"></a>
### 配置

所有配置都通过 `backend/.env` 设置：

```yaml
# Milvus 配置
MILVUS_URI=http://localhost:19530      # Milvus 连接 URI
COLLECTION_NAME=papers                 # Collection 名称前缀

# 模型路径
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 # 嵌入模型路径
RERANKER_MODEL=BAAI/bge-reranker-v2-m3 # 重排序模型路径

# 检索参数
FETCH_K=20                             # 重排序前的候选数量
TOP_K=5                                # 每次查询返回的检索文档数量
RRF_K=60

# 缓存配置
ENABLE_CACHE=true
CACHE_MAX_SIZE=1000

# LLM 配置
LLM_BASE_URL=http://localhost:8848/v1  # LLM 端点（OpenAI 兼容）
LLM_MODEL=GPT-4o-mini                  # 模型名称
LLM_API_KEY=empty
LLM_TEMPERATURE=0.1                    # 生成温度
LLM_MAX_TOKENS=4096
MAX_RETRIES=0                          # 反思重试次数上限

# VLM 配置
VLM_ENABLED=true                       # 启用 VLM 进行图像分析
VLM_BASE_URL=http://localhost:8080/v1  # VLM 端点（OpenAI 兼容，多模态）
VLM_MODEL=Qwen3-vl-4B                  # VLM 模型名称
VLM_API_KEY=empty                      # VLM API Key

# Postgres（checkpointer + session store）
POSTGRES_URI=postgresql://postgres:postgres@localhost:5432/scholar_rag

# 上传
UPLOAD_DIR=./uploads
MAX_UPLOAD_SIZE_MB=50

# 服务器
HOST=0.0.0.0
PORT=8000
```

<a id="docker-option"></a>
### 方式一：Docker（推荐）

```bash
cp backend/.env.example backend/.env   # 编辑为你的模型端点
docker-compose up -d
```

所有服务都会自动启动。打开 http://localhost:8000

<a id="makefile-option"></a>
### 方式二：Makefile

需要本地已经运行 Milvus 和 Postgres。

```bash
cp backend/.env.example backend/.env   # 编辑为你的模型端点
make install                            # 安装所有依赖
make start                              # 构建前端，并在 http://localhost:8000 启动后端
```

开发模式（热重载）：

```bash
make dev                                # backend :8000 + frontend dev server :5173
```

<a id="use"></a>
### 使用方法

1. 在浏览器中打开 http://localhost:8000
2. 通过上传面板上传 PDF 论文
3. 开始提问——几秒内获得带引用的回答

---

<a id="api-reference"></a>
## ⚙️ API 参考

| 方法 | Endpoint | 描述 |
|---|---|---|
| `POST` | `/api/chat` | SSE 流式聊天（`{query, session_id?}`） |
| `GET` | `/api/sessions` | 获取会话列表 |
| `GET` | `/api/sessions/:id/history` | 获取对话历史 |
| `DELETE` | `/api/sessions/:id` | 删除会话 |
| `POST` | `/api/files/upload` | 上传 PDF（multipart） |
| `GET` | `/api/files` | 获取已上传文件列表 |
| `DELETE` | `/api/files/:id` | 删除文件和对应向量 |
| `DELETE` | `/api/collection` | 清空向量数据库 |
| `GET` | `/api/health` | 健康检查 |

---

<a id="evaluation"></a>
## 📊 评估

```bash
cd backend

# 检索：Recall@k、Precision@k、MRR、MAP
python eval/eval_retrieval.py

# 生成：RAGAS（Faithfulness、Relevancy、Precision、Correctness）
python eval/eval_generation.py
```

---

<a id="tech-stack"></a>
## 🗝️ 技术栈

<a id="llm-orchestration-layer"></a>
<details>
  <summary>1. LLM 编排层（点击展开）</summary>

项目的核心智能体工作流基于 LangGraph（`backend/agent/graph.py`）构建，采用多智能体架构：

**主图流程：**
```
START → summarize → classify → analyze → [sub_agent × N] → prepare_synthesis → END
```

- `summarize`：将超过窗口大小（6 轮）的对话历史压缩成摘要，使用 `RemoveMessage` 清理旧消息，防止上下文溢出。
- `classify`：通过 LLM 结构化输出（`with_structured_output`）将用户问题分类为四类：`experimental_result`、`method`、`background` 和 `general`，用于后续检索路由。
- `analyze`：将复杂问题拆解为多个子查询（`QueryAnalysis`），并通过 `Send` 机制并行分发给多个子智能体。
- `prepare_synthesis`：聚合所有子智能体回答，重新映射引用编号，并构建最终综合回答提示词。

**子智能体图流程（每个子查询独立运行）：**
```
START → retrieve → generate → reflect → [retry | done]
                                ↑                |
                          prepare_retry ←--------┘
```

- `reflect` 节点使用 LLM 判断答案是否充分（`ReflectionResult`）。如果不充分，它会生成补充查询并重试，最多重试 `MAX_RETRIES` 次（默认 2 次）。
- 反思阶段还包含 VLM 兜底机制：当文本回答不足且检索结果包含图像/图表时，系统会自动触发视觉模型分析。

LangChain 提供底层抽象：`BaseChatModel`、`Document`、`HumanMessage/AIMessage/SystemMessage`、`RecursiveCharacterTextSplitter` 等。LLM 调用通过 `langchain-openai` 的 `ChatOpenAI` 完成，兼容任何 OpenAI 格式 API（默认配置为 Ollama 的 `qwen3:32b`）。

结构化输出大量使用 Pydantic 模型（`QueryAnalysis`、`QueryClassification`、`ReflectionResult`），确保 LLM 返回可解析的结构化数据。

</details>

<a id="vector-database"></a>
<details>
  <summary>2. 向量数据库（点击展开）</summary>

Milvus 通过 Docker Compose 部署（`milvusdb/milvus:v2.4.0` standalone 模式），使用内置 etcd 和本地存储。

**混合检索架构（`backend/rag/retrieval.py`）：**

- 使用 `langchain-milvus` 集成；每个 collection 同时构建稠密向量索引和 BM25 稀疏索引（`BM25BuiltInFunction`）。
- 通过 RRF（Reciprocal Rank Fusion）融合两条路径的结果，`rrf_k` 默认值为 60。
- 支持元数据过滤：按 `node_type`（table/figure/caption 等）和 `section_path` 过滤。

**父子分块策略（`backend/rag/integration.py`）：**

- 文档被切分为父块（完整语义单元）和子块（500 字符切片，50 字符重叠）。
- 表格、图像、标题、图注等特殊节点不会继续切分，而是直接作为子块使用。
- 检索时，系统先搜索子 collection；命中后，通过 `chunk_parent_id` 回溯到父块，以获得更完整的上下文。
- 两个 collection 分别命名为 `{collection_name}_children` 和 `{collection_name}_parents`。

**检索流水线：**
```
Query → [可选 HyDE 扩展] → 混合搜索（BM25+Dense）→ RRF 融合 → 重排序 → 父块扩展 → 去重 → Top-K
```

系统还实现了检索缓存（`RetrievalCache`）和增量更新（`IncrementalUpdater`）。

</details>

<a id="pdf-parsing"></a>
<details>
  <summary>3. PDF 解析（点击展开）</summary>

**Docling（`backend/rag/integration.py`）：**

- 使用 `DocumentConverter` 解析 PDF，自动识别文档结构元素：`SectionHeaderItem`、`TextItem`、`ListItem`、`TableItem`、`PictureItem`、`FormulaItem`。
- 支持 OCR 兜底：如果初次解析得到的文本过少（总字符数 < 1000，或每页 < 200 字符），系统会自动启用 OCR 重新解析。
- 解析后的元素会经过过滤（移除页眉、页脚和页码）、阅读顺序排序（基于 bbox 坐标进行行列分组）以及章节层级追踪。

**节点内容生成（`backend/rag/node_generator.py`）：**

使用工厂模式，为 6 种节点类型提供专门的内容生成器：
- `ParagraphGenerator`：追加章节路径上下文
- `TableGenerator`：将表格线性化为 `Row N: header1=val1, header2=val2` 格式
- `FigureGenerator`：组合图注和周围描述文本
- `FormulaGenerator`：追加章节上下文
- `CaptionGenerator`、`SectionHeaderGenerator`

**PyMuPDF（`fitz`）：**

- 用于图像/图表裁剪：根据 Docling 提供的 bbox 坐标，从 PDF 页面中裁剪图像区域。
- 坐标系统转换：Docling 使用 PDF 标准坐标系（原点在左下角），而 PyMuPDF 使用屏幕坐标系（原点在左上角），通过 `fitz_y = page_height - docling_y` 转换。
- 以 2x DPI 渲染并保存为 PNG，存储在 `data/figures/{paper_id}/` 目录。

</details>

<a id="reranking"></a>
<details>
  <summary>4. 重排序（点击展开）</summary>

- 使用 `sentence-transformers` 的 `CrossEncoder` 加载 `BAAI/bge-reranker-v2-m3` 模型。
- 检索时，先获取 `fetch_k × 2` 个候选文档，用 CrossEncoder 打分，然后取前 `fetch_k` 个。
- 嵌入模型使用 `HuggingFaceEmbeddings`（`langchain-huggingface`），默认模型为 `BAAI/bge-small-en-v1.5`。
- 两类服务都采用单例模式（`EmbeddingService`、`RerankerService`），避免重复加载。

</details>

<a id="vlm"></a>
<details>
  <summary>5. VLM（点击展开）</summary>

**VisionService（`backend/rag/factory.py`）：**

- 采用单例模式；接受任意 `BaseChatModel` 作为后端（默认通过 Ollama 使用 `qwen-vl`）。
- 将图像/图表图片编码为 base64，并通过 OpenAI 兼容的多模态消息格式发送给 VLM。
- 分析内容包括：图表类型、关键视觉元素、主要发现以及可见数值。

**触发机制（双路径）：**

1. 主动触发：当查询中包含视觉关键词（如 “show”“chart”“figure” 等），且检索结果中包含图像时，系统会在 `generate` 阶段注入 VLM 描述。
2. 兜底触发：当 `reflect` 阶段判断回答不充分时，系统会检查是否存在尚未分析的图像，触发 VLM 补充分析并重新生成答案（最多处理 2 张图像，以控制成本）。

VLM 描述会以 `[Figure Analysis]` 前缀追加到文档上下文中，供 LLM 在生成最终回答时引用。

</details>

<a id="backend-details"></a>
<details>
  <summary>6. 后端（点击展开）</summary>

**FastAPI（`backend/app/main.py`）：**

- 4 个路由模块：`chat`（对话）、`sessions`（会话管理）、`files`（文件上传）、`manage`（collection 管理）。
- CORS 完全开放（开发模式）。
- 支持挂载前端静态文件（`frontend/dist`），实现单端口部署。

**SSE 流式输出（`backend/app/routers/chat.py`）：**

- 使用 `sse-starlette` 的 `EventSourceResponse` 实现 Server-Sent Events。
- 流式事件类型：`session_id` → `status` → `sub_queries` → `answer`（逐 token）→ `citations` → `done`。
- 在综合回答阶段，通过 `llm.astream()` 流式输出 token，以支持前端实时渲染。
- 回答完成后，通过 `graph.aupdate_state()` 将对话历史持久化到 checkpointer。

**Uvicorn：** ASGI 服务器，支持开发模式热重载。

</details>

<a id="state-persistence"></a>
<details>
  <summary>7. 状态持久化（点击展开）</summary>

- PostgreSQL 16（Alpine）通过 Docker Compose 部署，用于 LangGraph 对话状态持久化。
- 使用 `langgraph-checkpoint-postgres` 的 `AsyncPostgresSaver` 进行异步 checkpoint 读写。
- 同时提供内存 checkpointer（`MemorySaver`）作为轻量替代方案。
- 数据库适配器使用 `psycopg` v3（支持 binary 和 pool）。

</details>

<a id="frontend-details"></a>
<details>
  <summary>8. 前端（点击展开）</summary>

**React 18（`frontend/src/`）：**

- 纯函数组件 + Hooks 架构（`useState`、`useEffect`、`useRef`、`useCallback`）。
- 组件结构：`App`（主布局）→ `Sidebar`（会话列表）、`ChatMessages`（消息展示）、`ChatInput`（输入框）、`FileUpload`（文件上传）、`SettingsPanel`（设置面板）。
- 使用 `react-markdown` 渲染 AI 回复中的 Markdown 内容。
- 使用 `lucide-react` 提供图标（Upload、Settings、ChevronLeft 等）。

**SSE 客户端（`frontend/src/api.js`）：**

- 使用原生 `fetch` + `ReadableStream` 手动解析 SSE 数据流。
- 支持 `AbortController` 取消正在进行的请求。
- 事件驱动：根据 `type` 字段（session_id/answer/citations/done/error）更新 UI 状态。

**构建工具链：**
- Vite 5：开发服务器 + 生产构建。
- TailwindCSS 3.4 + PostCSS + Autoprefixer：样式处理。
- ESLint 9 + eslint-plugin-react/react-hooks/react-refresh：代码质量检查。
- 通过 Nginx 反向代理进行生产部署（`frontend/nginx.conf` + Dockerfile）。

</details>

<a id="evaluation-system"></a>
<details>
  <summary>9. 评估系统（点击展开）</summary>

**RAGAS 生成质量评估（`backend/eval/eval_generation.py`）：**

- 评估指标：`Faithfulness`、`AnswerRelevancy`、`ContextPrecision`、`FactualCorrectness`。
- 使用 `LangchainLLMWrapper` 和 `LangchainEmbeddingsWrapper` 适配评估器。
- 端到端评估：运行完整智能体图，收集答案和上下文，并输出 CSV 报告。

**自定义检索评估（`backend/eval/eval_retrieval.py`）：**

- 指标：`Recall@k`、`Precision@k`、`MRR`（Mean Reciprocal Rank，平均倒数排名）、`MAP`（Mean Average Precision，平均精确率均值）。
- 直接评估完整检索流水线：混合搜索 + 重排序 + 父块扩展。

</details>

<a id="devops-and-deployment"></a>
<details>
  <summary>10. DevOps 与部署（点击展开）</summary>

**Docker Compose（`docker-compose.yml`）：**

4 服务编排：
- `backend`：FastAPI 应用，在 Milvus 和 Postgres 健康检查通过后启动。
- `frontend`：Nginx 托管构建产物，映射到端口 5173。
- `milvus`：v2.4.0 standalone，内置 etcd，暴露 19530（gRPC）和 9091（健康检查）。
- `postgres`：16-alpine，使用持久化卷 `postgres_data`。

**Makefile：** 提供快捷命令：`install`、`dev`（同时启动前后端）、`build`、`test`（pytest）、`lint`、`clean` 等。

**环境配置：** 通过 `python-dotenv` 加载 `.env` 文件；所有配置项都可以通过环境变量覆盖（`backend/config.py`）。

</details>

---

<a id="security-notice"></a>
## ⚠️ 安全提示

ScholarRAG 被设计为一个**研究和学习工具**，适合运行在**可信的本地环境或内部网络环境**中。它默认不包含生产级安全加固。部署前请仔细阅读以下内容：

<a id="api-and-authentication"></a>
### API 与认证

- **没有内置认证或授权机制。** 所有 API 端点（聊天、文件上传、会话历史、collection 管理）都可以被任何能够访问服务器的人公开调用。
- **Session ID 是唯一的访问边界。** 任何拥有有效 session ID 的人都可以读取完整对话历史或删除该会话。
- **破坏性端点未受保护。** `DELETE /api/collection` 会在没有任何确认或凭据校验的情况下清空整个向量数据库。

<a id="credentials-and-secrets"></a>
### 凭据与密钥

- **API Key 和数据库凭据**（`LLM_API_KEY`、`VLM_API_KEY`、`POSTGRES_URI`）以明文形式存储在 `.env` 文件中。切勿将 `.env` 提交到版本控制系统。
- `.env.example` 和 `docker-compose.yml` 中的**默认凭据**（例如 `postgres:postgres`）必须在任何非本地部署前修改。
- 项目**没有密钥轮换机制**——请定期手动轮换 key 和密码。

<a id="network-and-transport"></a>
### 网络与传输

- 默认情况下，**所有服务都通过普通 HTTP 通信**（FastAPI、Milvus、PostgreSQL、LLM/VLM 端点）。如果系统暴露在 localhost 之外，请通过反向代理（例如 Nginx）配置 TLS/HTTPS。
- **CORS 完全开放**（`allow_origins=["*"]`）。生产环境中应将允许的来源限制为你的前端域名。
- **Milvus 和 PostgreSQL** 默认暴露且没有网络层访问控制。请使用防火墙规则或 Docker 网络隔离来限制访问。

> [!CAUTION]
> 在没有添加认证、TLS 和适当访问控制之前，不要将 ScholarRAG 直接暴露到公网。当前配置适合本地开发和可信内部网络环境。

---

<a id="license"></a>
## 📝 许可证

本项目开源，并基于 [MIT License](./LICENSE) 发布。

---

<a id="key-contributors"></a>
## 🎉 主要贡献者

- [PangHu1020](https://github.com/PangHu1020)
- [curme-miller](https://github.com/curme-miller)

---

<a id="star-history"></a>
## 🎖️ Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=PangHu1020/scholar-rag&type=Date)](https://www.star-history.com/#PangHu1020/scholar-rag&Date)
