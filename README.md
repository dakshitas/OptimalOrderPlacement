# Smart Order Router – Cont & Kukanov Model (Static Allocator)

## Overview

This project implements and backtests a Smart Order Router (SOR) following the static allocator proposed by Cont & Kukanov in their paper _"Optimal Order Placement in Limit Order Markets"_. The router allocates a 5,000-share buy order across multiple venues based on expected cost minimization, accounting for rebates, fees, queue risk, and execution penalties.

## Methodology

- **Allocator Logic**: Exhaustively searches for the lowest-cost allocation using a discretized grid (100-share chunks), as described in `allocator_pseudocode.txt`.
- **Backtest Engine**: Simulates execution across timestamped market snapshots. At each step, it attempts to execute a chunk and rolls forward any unfilled quantity.
- **Parameter Search**: Grid search over `lambda_over`, `lambda_under`, and `theta_queue` to minimize total cost while filling all 5,000 shares.

## Results

The best performing configuration found was:

```json
{
  "lambda_over": 0.0,
  "lambda_under": 0.0,
  "theta_queue": 0.0
}
```

| Strategy           | Cash Spent | Avg. Fill Price | Shares Filled | Bps Savings |
|--------------------|------------|------------------|----------------|--------------|
| **Tuned Allocator**| \$1,113,732.00 | 222.7464          | 5000           | —            |
| Best Ask Baseline  | \$1,114,117.28 | 222.8235          | 5000           | 3.46         |
| VWAP Baseline      | \$1,114,117.28 | 222.8235          | 5000           | 3.46         |
| TWAP Baseline      | \$776,763.28   | 223.0796          | **3482**        | Incomplete   |

### Justification

- **Tuned Result**: Achieved a full fill at a lower cost than all baseline strategies. The optimizer chose to avoid additional penalties (setting all λ and θ to 0), which worked effectively in this data slice.
- **TWAP Underperformance**: The TWAP baseline failed to fill the entire order due to uneven liquidity distribution across time slices.
- **VWAP and Best Ask**: Both behave similarly as they greedily target best price venues but lack adaptive allocation, hence higher cost.


## Files

- `backtest.py`: Main script containing the allocator, backtest loop, baseline methods, and final output.
- `results.json`: Formatted JSON with tuned and baseline performance.
- `README.md`: This documentation.
- `results.pdf`: (optional) Plot showing cumulative cost over time.
