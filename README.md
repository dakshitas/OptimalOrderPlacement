# Cont & Kukanov Smart Order Router Backtest

This project implements a static smart order routing strategy inspired by **Cont & Kukanov (2014)** to minimize execution cost across fragmented limit order markets. The strategy splits a 5,000-share buy order across multiple venues using a cost function penalizing overfills, underfills, and queue risk.

---

## Code Structure

- **`backtest.py`**: Main script implementing:
  - Static allocator (`allocate` function from pseudocode),
  - Backtest engine (replaying snapshots from `l1_day.csv`),
  - Baselines: Best Ask, TWAP, VWAP,
  - Grid search over penalty parameters,
  - Final JSON output and optional plot.

- **`allocator_pseudocode.txt`**: Provided pseudocode translated line-by-line into Python.
- **`results.png`**: Visual comparison of cumulative costs under different strategies.

---

## Parameter Search Details

A small grid search was run over the three penalty parameters:

- `lambda_over ∈ {0.001, 0.01, 0.1}`
- `lambda_under ∈ {0.001, 0.01, 0.1}`
- `theta_queue ∈ {0.0001, 0.001, 0.01}`

The best-performing set was:

```json
{
  "lambda_over": 0.01,
  "lambda_under": 0.01,
  "theta_queue": 0.001
}
```

## Performance Summary

"cont_kukanov": {
  "total_cash": 1113701.0,
  "avg_price": 222.7402
},
"best_ask": {
  "total_cash": 1114103.28,
  "avg_price": 222.8207
},
"twap": {
  "total_cash": 1254701.16,
  "avg_price": 223.0580
},
"vwap": {
  "total_cash": 1112845.40,
  "avg_price": 222.5691
},
"savings_bps": {
  "vs_best_ask": 3.61,
  "vs_twap": 14.25,
  "vs_vwap": -7.69
}

## Key Takeaways

The Cont-Kukanov allocator beat Best Ask by 3.61 bps and TWAP by 14.25 bps, validating the model’s effectiveness.
VWAP slightly outperformed, possibly due to favorable size-weighted pricing in this short time window.

## Suggested Improvement

To improve fill realism:

Model slippage: Penalize fills that require crossing multiple levels or trading in high-volatility intervals.
Queue position modeling: Estimate partial fill probabilities based on historical queue outflows and current depth.
