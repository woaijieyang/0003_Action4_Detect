# 通用 AI 科研代码仓库模板

本仓库提供代码目录骨架与协作规范，适合以数据处理、模型实现和测评为主的 AI 科研项目。当前只有三个可安装的空包与文档，没有模型、训练器、测评命令、业务测试或已登记测评成绩。

## 从这里开始

1. 阅读 [工作规范](AGENTS.md) 与 [架构边界](docs/ARCHITECTURE.md)。
2. 将 `pyproject.toml` 的项目名称、描述和版本改为实际项目元数据，更新本文定位；作者信息按实际填写，许可声明须与随附 LICENSE 一致。
3. 使用独立环境安装骨架，再按实际实现添加依赖与测试。
4. 复杂阶段按 [阶段规则](docs/stages/AGENTS.md) 创建计划和结论。训练、测评、下载与发布依当前任务授权执行。
5. 完成有效测评后，按 [填写规则](results/README.md) 更新根目录的 [模型测评成绩总表](results/results.md)。

## 安装与最小导入检查

要求 Python 3.10 或更高版本。以下命令在仓库根目录执行；默认安装可能访问包索引获取构建依赖，离线环境应先准备已核验的工具包。

Linux / macOS：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -c "import importlib; [importlib.import_module(name) for name in ('eval', 'model', 'data')]; print('package imports OK')"
```

Windows PowerShell：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -c "import importlib; [importlib.import_module(name) for name in ('eval', 'model', 'data')]; print('package imports OK')"
```

`py -3` 所选版本须满足上述下限。导入检查只证明包可加载，不代表模型或测评功能已经实现。构建后端要求 setuptools 77 或更高版本，以支持 SPDX 许可声明；运行时不依赖 setuptools。配置示例 `.env.example` 不会自动加载，本版没有 CLI、网络服务或 GPU 依赖。

## 实际目录

```text
src/
  eval/                 测评加载、环境与计分代码
  model/                模型与算法；training 按需建立
  data/                 原始数据加工与数据集封装代码
data/                   数据资产（只跟踪目录 README）
models/                 模型权重（只跟踪目录 README）
outputs/                原始运行产物（只跟踪目录 README）
runtime/                隔离环境与构建（只跟踪目录 README）
upstream/               第三方源码副本（只跟踪目录 README）
configs/                配置组织说明
scripts/                薄入口约定
tests/                  工程测试约定
results/
  README.md             测评成绩填写规则
  results.md            模型测评成绩总表
docs/
  ARCHITECTURE.md        架构与扩展边界
  EXPERIMENTS.md         运行身份与证据规范
  stages/AGENTS.md       阶段文档规则
  templates/            计划、结论与单次测评详情模板
.github/
  pull_request_template.md
```

根目录同时提供 AGENTS、README、CONTRIBUTING、README_WORKFLOW、pyproject 和 Git/环境示例配置。目录中的 README 说明其职责，不代表已有对应业务功能。

## 源码、资产和结果

| 位置 | 保存内容 | Git |
|---|---|---|
| `src/data/`、`src/model/` | 数据处理代码、模型代码 | 跟踪 |
| 根 `data/`、`models/` | 已预建数据与权重资产目录 | 仅跟踪目录 README，实际资产忽略 |
| `outputs/` | 已预建；运行时按 run_id 保存逐样本输出、日志和身份清单 | 仅跟踪目录 README，运行产物忽略 |
| `results/` | 经核实的测评成绩与必要小型摘要 | 跟踪 |
| `docs/` | 架构、用法、阶段过程与结论 | 跟踪 |
| `runtime/`、`upstream/` | 已预建隔离环境与第三方副本目录 | 仅跟踪目录 README，实际内容忽略 |
| `.cache/`、`.venv/` | 工具按需生成的缓存与 Python 环境 | 全部忽略 |

五个资产根目录通过各自 README 保留，因此克隆后可直接看到；真实资产不进入 Git。`data/` 是数据资产目录，`src/data/` 是处理代码，两者同时存在。

外部共享数据默认只读。忽略目录不等于可以随意删除；新运行使用新 run_id，保留原始证据。测评总表不保存训练日志或原始预测。

## 开发与记录入口

- [协作指南](CONTRIBUTING.md)：代码、配置、依赖与验证。
- [运行和实验规范](docs/EXPERIMENTS.md)：源码、模型、数据、配置及产物身份。
- [阶段计划模板](docs/templates/PLAN_TEMPLATE.md) / [阶段结论模板](docs/templates/CONCLUSION_TEMPLATE.md)。
- [单次测评详情模板](docs/templates/RESULT_TEMPLATE.md)：总表需要额外证据说明时使用。
- [提交与发布流水](README_WORKFLOW.md)。

目前没有业务测试和 CI 工作流，不能用零用例或导入通过替代业务验收。后续有真实代码与测试后再接入 CI。

## 许可

本仓库采用已有的 [Apache License 2.0](LICENSE)。许可文本来自本仓库初始提交，完整保留。本模板没有附带第三方代码、权重或数据；其授权需分别核对。
