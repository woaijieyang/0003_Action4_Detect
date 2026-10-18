# 工程测试

统一使用 tests，不建立第二个 test 根目录。当前只有约定，没有业务测试；空用例集不代表通过。

## 随实现增加的验证

- 单元测试：纯逻辑、参数边界、无效输入、状态不变量、错误传播。
- 集成测试：数据、模型、测评和入口之间的实际接口，必要外部依赖通过明确夹具替换。
- 真实依赖验证：需要模型、设备、原生库或真实数据时单独声明条件、资源和范围；未执行不能计为通过。

可按实现建立 data、model、eval 测试子目录。只提交必要、小型、来源合法的夹具；原始数据、权重和大型预测不是夹具。
验证应覆盖具体行为、失败与回归，而不是重复实现或仅检查文件名。保留失败，不删除有效断言，不把未执行的 GPU 测评解释为 CPU 已覆盖。

测试临时文件写入仓库内的 .cache 或任务自有输出，不能修改共享数据或既有运行。
后续选用 pytest 时先声明测试依赖，检查退出状态与实际收集数量。必须显式将 --basetemp 指向仓内 .cache/pytest 下本次独占的新目录；pytest 会清理指定 basetemp，因此不能使用历史输出、共享数据或其他任务目录。性能与模型成绩按 [测评登记规则](../results/README.md) 单独处理，工程测试不进入成绩总表。

## pytest 临时目录示例

项目已有测试并安装声明的 pytest 依赖后，可在仓库根目录执行以下 Python 片段。每次创建独立目录，便于并行检查；片段不负责安装依赖。当前骨架没有测试，运行 pytest 仍应如实报告未收集到用例。

```python
from pathlib import Path
from tempfile import mkdtemp
import subprocess
import sys

base = Path(".cache/pytest")
base.mkdir(parents=True, exist_ok=True)
run_dir = mkdtemp(prefix="run-", dir=str(base.resolve()))
raise SystemExit(subprocess.call([
    sys.executable, "-m", "pytest", "tests", "--basetemp", run_dir,
]))
```
