"""
Tests for finance utilities.
"""

import pytest
import pandas as pd
import numpy as np
from utils.finance_utils import (
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_max_drawdown
)

def test_calculate_returns():
    """Test returns calculation."""
    prices = pd.Series([100, 110, 99, 105])
    returns = calculate_returns(prices)
    assert len(returns) == 3
    assert returns.iloc[0] == 0.1
    assert returns.iloc[1] == -0.1
    assert returns.iloc[2] == 0.06 