"""Tests for ``search_stock_symbols`` (Role 2 dynamic symbol search).

``search_stock_symbols`` is an online lookup: it pulls the full A-share
code/name universe from AkShare ``stock_info_a_code_name`` (eastmoney) and
filters it down to supported SH/SZ markets. Every test here mocks that call and
clears the process-wide cache so the suite runs offline and deterministically.
"""

import pandas as pd
import pytest

import src.data.fetch as fetch_mod
from src.data.fetch import search_stock_symbols
from src.utils.exceptions import NoDataError


def _universe_frame() -> pd.DataFrame:
    """AkShare ``stock_info_a_code_name`` 输出形状（含北交所，应被剔除）。"""
    return pd.DataFrame({
        "code": ["600519", "000001", "300750", "688981", "600276", "830000"],
        "name": ["贵州茅台", "平安银行", "宁德时代", "中芯国际", "恒瑞医药", "北交样本"],
    })


@pytest.fixture
def stock_universe(monkeypatch):
    """Mock 全市场代码表，并清空进程内缓存，保证每次测试独立。"""
    import akshare

    monkeypatch.setattr(akshare, "stock_info_a_code_name", lambda: _universe_frame())
    monkeypatch.setattr(fetch_mod, "_stock_universe_cache", None)
    monkeypatch.setattr(fetch_mod, "_stock_universe_cached_at", None)
    return akshare


def _no_sleep(monkeypatch):
    monkeypatch.setattr(fetch_mod.time, "sleep", lambda *a, **k: None)


def test_search_returns_contract_columns(stock_universe):
    df = search_stock_symbols("600519")
    assert list(df.columns) == ["symbol", "name", "market"]


def test_search_by_code_exact(stock_universe):
    df = search_stock_symbols("600519")
    assert len(df) == 1
    assert df.iloc[0]["symbol"] == "600519.SH"
    assert df.iloc[0]["name"] == "贵州茅台"
    assert df.iloc[0]["market"] == "SH"


def test_search_by_name_substring(stock_universe):
    df = search_stock_symbols("茅台")
    assert len(df) == 1
    assert df.iloc[0]["symbol"] == "600519.SH"


def test_search_code_prefix_fuzzy(stock_universe):
    df = search_stock_symbols("600")
    assert set(df["symbol"]) == {"600519.SH", "600276.SH"}


def test_search_suffixed_and_prefixed_query(stock_universe):
    # 带后缀 / 前缀的代码查询提取数字后同样命中
    assert search_stock_symbols("600519.SH").iloc[0]["symbol"] == "600519.SH"
    assert search_stock_symbols("sh600519").iloc[0]["symbol"] == "600519.SH"


def test_search_excludes_unsupported_market(stock_universe):
    # 北交所 830000 被剔除，名称也搜不到
    assert search_stock_symbols("北交").empty
    assert search_stock_symbols("830000").empty


def test_search_empty_query_returns_empty(stock_universe):
    df = search_stock_symbols("")
    assert df.empty
    assert list(df.columns) == ["symbol", "name", "market"]


def test_search_no_match_returns_empty(stock_universe):
    assert search_stock_symbols("不存在的股票").empty


def test_search_limit_clamped(stock_universe):
    # limit=0 → 收敛到 1
    assert len(search_stock_symbols("6", limit=0)) == 1
    # limit 很大 → 收敛到 200（本例仅 3 只 6 开头，故返回全部）
    assert len(search_stock_symbols("6", limit=999)) == 3


def test_search_sorted_by_symbol(stock_universe):
    df = search_stock_symbols("6")
    assert df["symbol"].tolist() == sorted(df["symbol"].tolist())


def test_search_dedups_duplicate_symbols(monkeypatch):
    import akshare

    dup = pd.DataFrame({
        "code": ["600519", "600519"],
        "name": ["贵州茅台", "贵州茅台"],
    })
    monkeypatch.setattr(akshare, "stock_info_a_code_name", lambda: dup)
    monkeypatch.setattr(fetch_mod, "_stock_universe_cache", None)
    monkeypatch.setattr(fetch_mod, "_stock_universe_cached_at", None)

    df = search_stock_symbols("茅台")
    assert len(df) == 1


def test_search_network_failure_raises(monkeypatch):
    import akshare

    def down():
        raise ConnectionError("universe down")

    monkeypatch.setattr(akshare, "stock_info_a_code_name", down)
    monkeypatch.setattr(fetch_mod, "_stock_universe_cache", None)
    monkeypatch.setattr(fetch_mod, "_stock_universe_cached_at", None)
    _no_sleep(monkeypatch)

    with pytest.raises(NoDataError):
        search_stock_symbols("600519")


def test_stock_universe_cache_has_real_ttl_and_returns_copies(monkeypatch):
    import akshare

    calls = {"count": 0}
    clock = {"now": 100.0}

    def load_universe():
        calls["count"] += 1
        return _universe_frame()

    monkeypatch.setattr(akshare, "stock_info_a_code_name", load_universe)
    monkeypatch.setattr(fetch_mod, "_stock_universe_cache", None)
    monkeypatch.setattr(fetch_mod, "_stock_universe_cached_at", None)
    monkeypatch.setattr(fetch_mod.time, "monotonic", lambda: clock["now"])

    first = fetch_mod._get_stock_universe()
    first.loc[0, "name"] = "被调用者修改"
    second = fetch_mod._get_stock_universe()

    assert calls["count"] == 1
    assert "被调用者修改" not in second["name"].tolist()

    clock["now"] += fetch_mod._STOCK_UNIVERSE_CACHE_TTL_SECONDS + 1
    fetch_mod._get_stock_universe()
    assert calls["count"] == 2
