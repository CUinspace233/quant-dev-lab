# quant-dev-lab

## Minimal A-Share Backtest Demo

一个贴近生产结构的 A 股日频最小量化 demo：

- 生成并读取 `data/raw/` 下的模拟供应商原始数据。
- 做数据质量检查和 processed 数据生成。
- 编写 `momentum_20d` 动量因子。
- 根据因子调仓并完成一个最小回测。

运行：

```bash
pip install -r requirements.txt
python examples/minimal_a_share_backtest.py
```

建议学习顺序：

1. 先读 [`docs/minimal_a_share_backtest.md`](./docs/minimal_a_share_backtest.md)。
2. 再从上到下读 [`examples/minimal_a_share_backtest.py`](./examples/minimal_a_share_backtest.py)。
3. 最后运行脚本，看输出里的数据检查、因子样例、持仓和回测指标。

## Interview Preparation Notes

[`quant_dev_interview_prep.md`](./quant_dev_interview_prep.md)
