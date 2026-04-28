#!/usr/bin/env python3
"""A 股日频最小生产化量化 demo。

这个脚本刻意做成“单文件但分层清楚”的形式：
1. 先模拟供应商原始数据，并保存到 data/raw。
2. 再做数据质量检查和清洗，输出到 data/processed。
3. 然后计算一个 20 日动量因子。
4. 最后把因子转成持仓并做一个最小回测。

真实生产系统一般会把这些逻辑拆成 data/factors/backtest/trading 多个模块。
这里保持单文件，是为了让学习者能从上到下读完整条链路。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

CALENDAR_FILE = RAW_DIR / "trade_calendar.csv"
QUOTE_FILE = RAW_DIR / "daily_quotes.csv"
STATUS_FILE = RAW_DIR / "stock_status.csv"
PROCESSED_FILE = PROCESSED_DIR / "clean_daily_panel.csv"


@dataclass(frozen=True)
class BacktestConfig:
    """集中放回测参数，方便复现实验和面试时说明策略假设。"""

    lookback: int = 20
    rebalance_every: int = 20
    top_n: int = 5
    min_listed_days: int = 60
    transaction_cost: float = 0.001


def ensure_sample_raw_data() -> None:
    """生成一套确定性的 A 股日频 raw CSV。

    生产系统中的 raw 数据一般来自 Wind、聚源、交易所、券商柜台等外部源。
    raw 层的原则是“原样保存、可追溯、不要覆盖”，因为后续清洗口径错了还要能回到原始数据复查。
    这里如果 raw 文件已经存在，就直接使用，避免每次运行都悄悄改掉原始数据。
    """

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = [CALENDAR_FILE, QUOTE_FILE, STATUS_FILE]
    existing_raw_files = [path for path in raw_files if path.exists()]
    if len(existing_raw_files) == len(raw_files):
        return
    if existing_raw_files:
        existing = ", ".join(str(path.relative_to(ROOT)) for path in existing_raw_files)
        raise FileExistsError(
            "raw 层数据只存在一部分，脚本不会自动覆盖或补写，"
            f"请人工确认后处理。已存在: {existing}"
        )

    # 用工作日近似交易日。真实 A 股需要交易所日历，不能直接用自然日或普通工作日。
    dates = pd.bdate_range("2024-01-02", periods=120)
    symbols = [f"00000{i}.SZ" for i in range(1, 10)] + ["600001.SH", "600002.SH", "600003.SH"]

    calendar = pd.DataFrame({"trade_date": dates})
    calendar["is_open"] = 1
    calendar["prev_trade_date"] = calendar["trade_date"].shift(1)
    calendar["next_trade_date"] = calendar["trade_date"].shift(-1)
    calendar.to_csv(CALENDAR_FILE, index=False, date_format="%Y-%m-%d")

    quote_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []

    for symbol_idx, symbol in enumerate(symbols):
        base_price = 8.0 + symbol_idx * 1.7
        drift = 0.0004 + symbol_idx * 0.00008
        prev_close = base_price
        list_offset = symbol_idx * 7

        for day_idx, trade_date in enumerate(dates):
            listed_days = max(day_idx - list_offset, 0)
            is_trading = 1
            is_st = 0
            suspend_reason = ""

            # 人工放入一些生产中必须处理的状态：停牌、ST、新股上市天数不足。
            if symbol == "000003.SZ" and 40 <= day_idx <= 42:
                is_trading = 0
                suspend_reason = "重大事项停牌"
            if symbol == "000006.SZ" and 70 <= day_idx <= 88:
                is_st = 1
            if listed_days == 0:
                is_trading = 0
                suspend_reason = "尚未上市"

            limit_up = round(prev_close * 1.10, 2)
            limit_down = round(prev_close * 0.90, 2)

            raw_ret = drift + 0.018 * np.sin(day_idx / 6 + symbol_idx) + 0.006 * np.cos(day_idx / 11)
            close = round(max(2.0, prev_close * (1 + raw_ret)), 2)

            # 放入涨停和跌停样例。真实回测里，涨停可能买不到，跌停可能卖不出。
            if symbol == "600001.SH" and day_idx == 65:
                close = limit_up
            if symbol == "000004.SZ" and day_idx == 85:
                close = limit_down

            open_price = round((prev_close + close) / 2, 2)
            high = round(max(open_price, close) * 1.01, 2)
            low = round(min(open_price, close) * 0.99, 2)
            volume = int(900_000 + symbol_idx * 45_000 + day_idx * 2_200)
            amount = round(volume * close, 2)

            # adj_factor 用来生成复权价格。真实系统中复权因子来自数据供应商。
            # 这里在少数日期制造分红/送股效果，演示为什么跨期收益不能直接用原始 close。
            adj_factor = 1.0
            if symbol in {"000002.SZ", "600002.SH"} and day_idx >= 55:
                adj_factor = 0.92
            if symbol == "000008.SZ" and day_idx >= 90:
                adj_factor = 1.08

            quote_rows.append(
                {
                    "trade_date": trade_date,
                    "symbol": symbol,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "amount": amount,
                    "adj_factor": adj_factor,
                    "limit_up": limit_up,
                    "limit_down": limit_down,
                }
            )
            status_rows.append(
                {
                    "trade_date": trade_date,
                    "symbol": symbol,
                    "is_trading": is_trading,
                    "is_st": is_st,
                    "listed_days": listed_days,
                    "suspend_reason": suspend_reason,
                }
            )

            if is_trading:
                prev_close = close

    pd.DataFrame(quote_rows).to_csv(QUOTE_FILE, index=False, date_format="%Y-%m-%d")
    pd.DataFrame(status_rows).to_csv(STATUS_FILE, index=False, date_format="%Y-%m-%d")


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读取 raw 层数据，并统一日期类型。

    真实系统中，读取入口应当集中管理，避免研究员在不同脚本中各自读不同字段、不同口径的数据。
    """

    calendar = pd.read_csv(CALENDAR_FILE, parse_dates=["trade_date", "prev_trade_date", "next_trade_date"])
    quotes = pd.read_csv(QUOTE_FILE, parse_dates=["trade_date"])
    status = pd.read_csv(STATUS_FILE, parse_dates=["trade_date"])
    return calendar, quotes, status


def validate_raw_data(calendar: pd.DataFrame, quotes: pd.DataFrame, status: pd.DataFrame) -> list[str]:
    """做最小数据质量检查。

    生产里这一步通常会接告警系统。数据错了但系统继续跑，是量化实盘中很危险的情况。
    """

    checks: list[str] = []

    if quotes.duplicated(["trade_date", "symbol"]).any():
        raise ValueError("daily_quotes 存在重复 trade_date + symbol")
    checks.append("daily_quotes 主键唯一")

    if status.duplicated(["trade_date", "symbol"]).any():
        raise ValueError("stock_status 存在重复 trade_date + symbol")
    checks.append("stock_status 主键唯一")

    required_quote_cols = ["open", "high", "low", "close", "volume", "amount", "adj_factor", "limit_up", "limit_down"]
    if quotes[required_quote_cols].isna().any().any():
        raise ValueError("daily_quotes 存在关键字段空值")
    checks.append("行情关键字段无空值")

    if not ((quotes["low"] <= quotes["close"]) & (quotes["close"] <= quotes["high"])).all():
        raise ValueError("存在 close 不在 low/high 区间内的记录")
    checks.append("价格区间合法")

    if not set(quotes["trade_date"]).issubset(set(calendar.loc[calendar["is_open"] == 1, "trade_date"])):
        raise ValueError("行情日期不在交易日历中")
    checks.append("行情日期均属于交易日历")

    return checks


def build_clean_panel(calendar: pd.DataFrame, quotes: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    """把 raw 行情和股票状态合并成研究/回测可用的日频面板。

    clean 层可以重算，所以允许由脚本生成；raw 层则应尽量只追加、不覆盖。
    """

    panel = quotes.merge(status, on=["trade_date", "symbol"], how="left", validate="one_to_one")
    panel = panel.merge(calendar[["trade_date", "next_trade_date"]], on="trade_date", how="left", validate="many_to_one")
    panel = panel.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    # 用复权价格计算收益，避免分红、送股、拆股导致原始 close 出现不可比跳变。
    panel["adj_close"] = panel["close"] * panel["adj_factor"]
    panel["return_1d"] = panel.groupby("symbol")["adj_close"].pct_change()

    # 这些 buy/sell 标记是回测约束的简化表达。真实交易还要看盘口、成交量、订单大小等。
    panel["can_buy"] = (
        (panel["is_trading"] == 1)
        & (panel["is_st"] == 0)
        & (panel["listed_days"] >= CONFIG.min_listed_days)
        & (panel["close"] < panel["limit_up"])
    )
    panel["can_sell"] = (panel["is_trading"] == 1) & (panel["close"] > panel["limit_down"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PROCESSED_FILE, index=False, date_format="%Y-%m-%d")
    return panel


def add_momentum_factor(panel: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """计算 20 日动量因子。

    momentum_20d 表示过去 20 个交易日的累计收益。
    关键点：这里先 pct_change(lookback)，再 shift(1)，表示 T 日信号只能看到 T-1 及以前的数据。
    如果不 shift(1)，T 日收盘后的价格会被用于 T 日决策，就会产生未来函数风险。
    """

    panel = panel.copy()
    panel["momentum_20d"] = (
        panel.groupby("symbol")["adj_close"]
        .transform(lambda s: s.pct_change(lookback).shift(1))
    )
    return panel


def build_rebalance_targets(panel: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """在调仓日把因子转成目标持仓。

    这里采用最简单的规则：在可买股票中选择动量排名前 N，等权持有。
    真实生产里通常还会加入行业中性、风格暴露、个股权重上限、黑名单等组合约束。
    """

    trade_dates = sorted(panel["trade_date"].unique())
    # 前 lookback 天没有足够历史窗口，不能调仓；之后每 rebalance_every 天调一次。
    rebalance_dates = trade_dates[config.lookback :: config.rebalance_every]
    candidates = panel[
        panel["trade_date"].isin(rebalance_dates)
        & panel["can_buy"]
        & panel["can_sell"]
        & panel["momentum_20d"].notna()
    ].copy()
    candidates["factor_rank"] = candidates.groupby("trade_date")["momentum_20d"].rank(ascending=False, method="first")

    selected = candidates[candidates["factor_rank"] <= config.top_n].copy()
    selected["target_weight"] = selected.groupby("trade_date")["symbol"].transform(lambda s: 1.0 / len(s))

    return selected[["trade_date", "symbol", "momentum_20d", "factor_rank", "target_weight"]].sort_values(
        ["trade_date", "factor_rank"]
    )


def expand_daily_weights(panel: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """把调仓日目标持仓扩展成每日持仓。

    信号日和收益计算日必须错开：T 日收盘后生成持仓，T+1 才能享受或承担收益。
    因此这里把 T 日 target_weight 向后滚动为后续交易日的 weight。
    """

    all_dates = sorted(panel["trade_date"].unique())
    symbols = sorted(panel["symbol"].unique())
    index = pd.MultiIndex.from_product([all_dates, symbols], names=["trade_date", "symbol"])
    weights = pd.DataFrame(index=index).reset_index()

    weights = weights.merge(targets[["trade_date", "symbol", "target_weight"]], on=["trade_date", "symbol"], how="left")
    # 只有调仓日才允许把“未入选股票”显式设为 0；非调仓日的 NaN 表示“沿用上一期持仓”。
    # 如果直接把所有 NaN 都填成 0，就会变成每天清仓，换手和收益都会失真。
    rebalance_dates = set(targets["trade_date"].unique())
    is_rebalance_date = weights["trade_date"].isin(rebalance_dates)
    weights.loc[is_rebalance_date & weights["target_weight"].isna(), "target_weight"] = 0.0
    weights["weight"] = weights.groupby("symbol")["target_weight"].ffill().fillna(0.0)
    return weights


def run_backtest(panel: pd.DataFrame, weights: pd.DataFrame, config: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """根据每日持仓计算组合净值。

    这个函数用两类数据回测：
    1. panel：清洗后的行情面板，核心字段是每只股票每天的 return_1d。
       return_1d 表示“从上一交易日到今天”的复权收益。
    2. weights：策略每天的持仓权重，核心字段是 weight。
       weight 表示“某天收盘后，策略决定未来持有这只股票多少仓位”。

    回测的核心公式是：
    某天组合收益 = sum(当天持仓权重 * 每只股票下一交易日收益) - 交易成本

    之所以用“下一交易日收益”，是因为 T 日收盘后才知道 T 日信号，
    所以这组持仓最早只能从 T+1 开始产生收益，不能假设 T 日已经赚到钱。
    """

    # 只取回测需要的行情字段。can_sell 暂时不直接改变收益，只用于说明真实系统中卖出约束会更复杂。
    returns = panel[["trade_date", "symbol", "return_1d", "can_sell"]].copy()

    # 把“每日持仓权重”和“每日股票收益”按 trade_date + symbol 拼起来。
    # 拼完后，每一行就是：某一天、某只股票、策略持有多少权重、这只股票有什么收益。
    bt = weights.merge(returns, on=["trade_date", "symbol"], how="left")
    bt = bt.sort_values(["symbol", "trade_date"])

    # panel 里的 return_1d 是“上一交易日 -> 当前交易日”的收益。
    # 但如果我们在 T 日收盘后形成 weight，那么它应该赚 T+1 的收益。
    # 所以这里按股票把 return_1d 向前挪一格，得到“当前持仓对应的下一日收益”。
    bt["next_return_1d"] = bt.groupby("symbol")["return_1d"].shift(-1).fillna(0.0)

    # 如果当天不能卖出，真实系统可能无法按目标权重降仓。这里用简化方式提示这个约束：
    # 对跌停无法卖出的股票，仍然保留收益暴露；更完整的系统需要做成交模拟和订单状态跟踪。
    # 单只股票贡献收益 = 这只股票的持仓权重 * 这只股票下一交易日收益。
    bt["weighted_return"] = bt["weight"] * bt["next_return_1d"]

    # 把长表转成“日期 x 股票”的权重矩阵，方便计算换手。
    # 换手用相邻两天持仓权重变化的绝对值之和近似。
    weight_matrix = bt.pivot(index="trade_date", columns="symbol", values="weight").fillna(0.0)
    turnover = weight_matrix.diff().abs().sum(axis=1).fillna(weight_matrix.abs().sum(axis=1))

    # 每天组合的毛收益，是当天所有股票贡献收益加总。
    daily = bt.groupby("trade_date", as_index=False)["weighted_return"].sum()
    daily["turnover"] = daily["trade_date"].map(turnover)

    # 交易越频繁，换手越高，手续费、滑点和冲击成本越高。
    # 这里用一个固定 transaction_cost 做最小演示。
    daily["cost"] = daily["turnover"] * config.transaction_cost
    daily["portfolio_return"] = daily["weighted_return"] - daily["cost"]

    # 净值从 1 开始滚动累乘：今天净值 = 昨天净值 * (1 + 今天组合收益)。
    daily["nav"] = (1.0 + daily["portfolio_return"]).cumprod()
    daily["drawdown"] = daily["nav"] / daily["nav"].cummax() - 1.0

    return daily, bt


def summarize_performance(daily: pd.DataFrame) -> dict[str, float]:
    """计算最常见的回测指标。"""

    n_days = len(daily)
    total_return = daily["nav"].iloc[-1] - 1.0
    annual_return = daily["nav"].iloc[-1] ** (252 / max(n_days, 1)) - 1.0
    annual_vol = daily["portfolio_return"].std(ddof=0) * np.sqrt(252)
    max_drawdown = daily["drawdown"].min()
    avg_turnover = daily["turnover"].mean()
    return {
        "累计收益": total_return,
        "年化收益": annual_return,
        "年化波动": annual_vol,
        "最大回撤": max_drawdown,
        "平均换手": avg_turnover,
    }


def print_section(title: str) -> None:
    print(f"\n{'=' * 12} {title} {'=' * 12}")


CONFIG = BacktestConfig()


def main() -> None:
    ensure_sample_raw_data()
    calendar, quotes, status = load_raw_data()

    print_section("数据质量检查")
    for item in validate_raw_data(calendar, quotes, status):
        print(f"[OK] {item}")

    panel = build_clean_panel(calendar, quotes, status)
    panel = add_momentum_factor(panel, CONFIG.lookback)
    targets = build_rebalance_targets(panel, CONFIG)
    weights = expand_daily_weights(panel, targets)
    daily, _ = run_backtest(panel, weights, CONFIG)
    metrics = summarize_performance(daily)

    print_section("因子样例")
    factor_cols = ["trade_date", "symbol", "adj_close", "momentum_20d", "can_buy", "can_sell"]
    print(panel.loc[panel["momentum_20d"].notna(), factor_cols].head(10).to_string(index=False))

    print_section("调仓日目标持仓样例")
    print(targets.head(15).to_string(index=False))

    print_section("回测核心指标")
    for key, value in metrics.items():
        print(f"{key}: {value:.2%}")

    print_section("最后 10 日净值")
    nav_cols = ["trade_date", "portfolio_return", "turnover", "cost", "nav", "drawdown"]
    print(daily[nav_cols].tail(10).to_string(index=False))

    print(f"\n已生成处理后数据: {PROCESSED_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
