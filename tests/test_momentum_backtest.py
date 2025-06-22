import os
import sys

import pandas as pd
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.momentum_backtest import calculate_metrics

def test_calculate_metrics_basic():
    dates = pd.date_range('2020-01-01', periods=5, freq='D')
    strategy_returns = np.array([0.01, -0.01, 0.02, -0.02, 0.03])
    equity = 10000 * np.cumprod(1 + strategy_returns)
    df = pd.DataFrame({'Strategy': strategy_returns, 'Equity': equity}, index=dates)
    metrics = calculate_metrics(df, 10000)
    assert metrics['CAGR'] == pytest.approx(13.202586787344716, rel=1e-6)
    assert metrics['Sharpe'] == pytest.approx(5.135376619417101, rel=1e-6)
    assert metrics['Max Drawdown'] == pytest.approx(-0.02, rel=1e-6)

