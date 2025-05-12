# Cont & Kukanov Smart Order Router Backtest

This project implements and evaluates a static smart order routing strategy based on the cost model proposed by **Cont & Kukanov (2014)**. The router aims to minimize execution cost when buying 5,000 shares across multiple venues by optimizing overfill, underfill, and queue risk penalties.

---

## 🔧 Implementation Summary

- **Allocator Logic**: Implemented exactly as per the `allocator_pseudocode.txt`, generating all feasible share allocations per snapshot and selecting the one with the lowest expected cost.
- **Backtest Engine**: Replays a message stream from `l1_day.csv`, feeds venue-level snapshots into the allocator, simulates order execution up to the posted ask size, and rolls forward unfilled quantity.
- **Parameter Search**: Conducted a grid search over:
  - `lambda_over ∈ {0.001, 0.01, 0.1}`
  - `lambda_under ∈ {0.001, 0.01, 0.1}`
  - `theta_queue ∈ {0.0001, 0.001, 0.01}`  
  Best parameters selected based on minimum total cost over the execution window.

---

## 📈 Results Summary

```json
"best_parameters": {
  "lambda_over": 0.01,
  "lambda_under": 0.01,
  "theta_queue": 0.001
},
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
