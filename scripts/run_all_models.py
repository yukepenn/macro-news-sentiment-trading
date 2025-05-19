#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Automate training and backtesting for all assets and models,
including walk-forward, cost, slippage, threshold, bootstrap settings.
"""

import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ASSETS = ["eurusd", "usdjpy", "zn"]
MODELS = ["logistic", "xgb"]

def run_command(cmd: str) -> None:
    logger.info(f"▶️  {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode:
        logger.error(res.stderr)
        res.check_returncode()
    logger.info(res.stdout)

def train_and_backtest(asset, model_type, cfg):
    """
    For a given asset+model combo:
      1) Train with walk-forward settings & bootstraps
      2) Backtest with costs/slippage/threshold
    """
    # names & paths
    model_pkl    = f"models/{asset}_{model_type}_model.pkl"
    shap_png     = f"plots/shap_{asset}_{model_type}.png"
    feats_csv    = f"data/processed/features/model_features_{asset}.csv"
    returns_csv  = f"backtest/{asset}_{model_type}_daily_returns.csv"
    equity_png   = f"visuals/{asset}_{model_type}_equity_curve.png"
    metrics_txt  = f"backtest/{asset}_{model_type}_performance_metrics.txt"

    # train
    train_cmd = (
        f"python -m scripts.train_model "
        f"--features {feats_csv} "
        f"--model {model_pkl} "
        f"--shap {shap_png} "
        f"--algo {model_type} "
        f"--n-splits {cfg.n_splits} "
        f"--max-rolling-window {cfg.max_rolling_window} "
        f"{'--no-threshold-opt' if cfg.no_threshold_opt else ''} "
        f"--cost-fx {cfg.cost_fx} --cost-fut {cfg.cost_fut} "
        f"{'--spread-fx '+str(cfg.spread_fx) if cfg.spread_fx is not None else ''} "
        f"{'--spread-fut '+str(cfg.spread_fut) if cfg.spread_fut is not None else ''} "
        f"--n-bootstrap {cfg.n_bootstrap}"
    )
    run_command(train_cmd)

    # backtest
    backtest_cmd = (
        f"python -m scripts.run_backtest "
        f"--model {model_pkl} "
        f"--features {feats_csv} "
        f"--asset {asset} "
        f"--returns-out {returns_csv} "
        f"--equity-plot {equity_png} "
        f"--metrics-out {metrics_txt} "
        f"--threshold {cfg.threshold} "
        f"--cost-fx {cfg.cost_fx} --cost-fut {cfg.cost_fut} "
        f"{'--spread-fx '+str(cfg.spread_fx) if cfg.spread_fx is not None else ''} "
        f"{'--spread-fut '+str(cfg.spread_fut) if cfg.spread_fut is not None else ''}"
    )
    run_command(backtest_cmd)

def main():
    p = argparse.ArgumentParser("Train & backtest sentiment models for all assets")
    p.add_argument("--assets",       nargs="+",   default=ASSETS)
    p.add_argument("--models",       nargs="+",   default=MODELS)
    # walk-forward & model params
    p.add_argument("--n-splits",     type=int,    default=5)
    p.add_argument("--max-rolling-window", type=int, default=20)
    p.add_argument("--no-threshold-opt", action="store_true")
    p.add_argument("--n-bootstrap",  type=int,    default=1000)
    # cost & slippage
    p.add_argument("--cost-fx",      type=float,  default=0.0002)
    p.add_argument("--cost-fut",     type=float,  default=0.0005)
    p.add_argument("--spread-fx",    type=float,  default=None)
    p.add_argument("--spread-fut",   type=float,  default=None)
    # final backtest threshold
    p.add_argument("--threshold",    type=float,  default=0.5)
    args = p.parse_args()

    # ensure dirs
    for d in ("models","plots","backtest","visuals"):
        Path(d).mkdir(exist_ok=True)

    for asset in args.assets:
        for model_type in args.models:
            logger.info(f"\n=== {asset.upper()} | {model_type.upper()} ===")
            train_and_backtest(asset, model_type, args)

    logger.info("✅  All assets & models completed.")

if __name__ == "__main__":
    main()
