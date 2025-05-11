{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "8616a114-c609-47c8-baab-e517c915f360",
   "metadata": {},
   "source": [
    "## Blockhouse Internship Assessment\n",
    "### Dakshita Srinivasan"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "e9014d38-8d3e-458d-a0e0-9199abb5f82c",
   "metadata": {},
   "outputs": [],
   "source": [
    "## Importing Necessary Modules and Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "63846bc1-886b-4a10-9e8c-e9dd0d1a6db2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 59999 entries, 0 to 59998\n",
      "Data columns (total 74 columns):\n",
      " #   Column         Non-Null Count  Dtype  \n",
      "---  ------         --------------  -----  \n",
      " 0   ts_recv        59999 non-null  object \n",
      " 1   ts_event       59999 non-null  object \n",
      " 2   rtype          59999 non-null  int64  \n",
      " 3   publisher_id   59999 non-null  int64  \n",
      " 4   instrument_id  59999 non-null  int64  \n",
      " 5   action         59999 non-null  object \n",
      " 6   side           59999 non-null  object \n",
      " 7   depth          59999 non-null  int64  \n",
      " 8   price          59999 non-null  float64\n",
      " 9   size           59999 non-null  int64  \n",
      " 10  flags          59999 non-null  int64  \n",
      " 11  ts_in_delta    59999 non-null  int64  \n",
      " 12  sequence       59999 non-null  int64  \n",
      " 13  bid_px_00      59999 non-null  float64\n",
      " 14  ask_px_00      59999 non-null  float64\n",
      " 15  bid_sz_00      59999 non-null  int64  \n",
      " 16  ask_sz_00      59999 non-null  int64  \n",
      " 17  bid_ct_00      59999 non-null  int64  \n",
      " 18  ask_ct_00      59999 non-null  int64  \n",
      " 19  bid_px_01      59999 non-null  float64\n",
      " 20  ask_px_01      59999 non-null  float64\n",
      " 21  bid_sz_01      59999 non-null  int64  \n",
      " 22  ask_sz_01      59999 non-null  int64  \n",
      " 23  bid_ct_01      59999 non-null  int64  \n",
      " 24  ask_ct_01      59999 non-null  int64  \n",
      " 25  bid_px_02      59999 non-null  float64\n",
      " 26  ask_px_02      59999 non-null  float64\n",
      " 27  bid_sz_02      59999 non-null  int64  \n",
      " 28  ask_sz_02      59999 non-null  int64  \n",
      " 29  bid_ct_02      59999 non-null  int64  \n",
      " 30  ask_ct_02      59999 non-null  int64  \n",
      " 31  bid_px_03      59999 non-null  float64\n",
      " 32  ask_px_03      59999 non-null  float64\n",
      " 33  bid_sz_03      59999 non-null  int64  \n",
      " 34  ask_sz_03      59999 non-null  int64  \n",
      " 35  bid_ct_03      59999 non-null  int64  \n",
      " 36  ask_ct_03      59999 non-null  int64  \n",
      " 37  bid_px_04      59999 non-null  float64\n",
      " 38  ask_px_04      59999 non-null  float64\n",
      " 39  bid_sz_04      59999 non-null  int64  \n",
      " 40  ask_sz_04      59999 non-null  int64  \n",
      " 41  bid_ct_04      59999 non-null  int64  \n",
      " 42  ask_ct_04      59999 non-null  int64  \n",
      " 43  bid_px_05      59999 non-null  float64\n",
      " 44  ask_px_05      59999 non-null  float64\n",
      " 45  bid_sz_05      59999 non-null  int64  \n",
      " 46  ask_sz_05      59999 non-null  int64  \n",
      " 47  bid_ct_05      59999 non-null  int64  \n",
      " 48  ask_ct_05      59999 non-null  int64  \n",
      " 49  bid_px_06      59999 non-null  float64\n",
      " 50  ask_px_06      59999 non-null  float64\n",
      " 51  bid_sz_06      59999 non-null  int64  \n",
      " 52  ask_sz_06      59999 non-null  int64  \n",
      " 53  bid_ct_06      59999 non-null  int64  \n",
      " 54  ask_ct_06      59999 non-null  int64  \n",
      " 55  bid_px_07      59999 non-null  float64\n",
      " 56  ask_px_07      59999 non-null  float64\n",
      " 57  bid_sz_07      59999 non-null  int64  \n",
      " 58  ask_sz_07      59999 non-null  int64  \n",
      " 59  bid_ct_07      59999 non-null  int64  \n",
      " 60  ask_ct_07      59999 non-null  int64  \n",
      " 61  bid_px_08      59999 non-null  float64\n",
      " 62  ask_px_08      59999 non-null  float64\n",
      " 63  bid_sz_08      59999 non-null  int64  \n",
      " 64  ask_sz_08      59999 non-null  int64  \n",
      " 65  bid_ct_08      59999 non-null  int64  \n",
      " 66  ask_ct_08      59999 non-null  int64  \n",
      " 67  bid_px_09      59999 non-null  float64\n",
      " 68  ask_px_09      59999 non-null  float64\n",
      " 69  bid_sz_09      59999 non-null  int64  \n",
      " 70  ask_sz_09      59999 non-null  int64  \n",
      " 71  bid_ct_09      59999 non-null  int64  \n",
      " 72  ask_ct_09      59999 non-null  int64  \n",
      " 73  symbol         59999 non-null  object \n",
      "dtypes: float64(21), int64(48), object(5)\n",
      "memory usage: 33.9+ MB\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "(None,\n",
       "                           ts_recv                        ts_event  rtype  \\\n",
       " 0  2024-08-01T13:36:32.492082308Z  2024-08-01T13:36:32.491911683Z     10   \n",
       " 1  2024-08-01T13:36:32.492082308Z  2024-08-01T13:36:32.491911683Z     10   \n",
       " 2  2024-08-01T13:36:32.492082308Z  2024-08-01T13:36:32.491911683Z     10   \n",
       " 3  2024-08-01T13:36:32.492082308Z  2024-08-01T13:36:32.491911683Z     10   \n",
       " 4  2024-08-01T13:36:32.492082308Z  2024-08-01T13:36:32.491912675Z     10   \n",
       " \n",
       "    publisher_id  instrument_id action side  depth   price  size  ...  \\\n",
       " 0             2             38      T    A      0  222.81   190  ...   \n",
       " 1             2             38      T    A      0  222.81    10  ...   \n",
       " 2             2             38      T    A      0  222.81   100  ...   \n",
       " 3             2             38      T    A      0  222.81   121  ...   \n",
       " 4             2             38      A    N      0  222.83    10  ...   \n",
       " \n",
       "    ask_sz_08  bid_ct_08  ask_ct_08  bid_px_09  ask_px_09  bid_sz_09  \\\n",
       " 0        100          4          1     222.72     222.94        219   \n",
       " 1        100          4          1     222.72     222.94        219   \n",
       " 2        100          4          1     222.72     222.94        219   \n",
       " 3        100          4          1     222.72     222.94        219   \n",
       " 4        100          3          1     222.71     222.94       1163   \n",
       " \n",
       "    ask_sz_09  bid_ct_09  ask_ct_09  symbol  \n",
       " 0        317          3          5    AAPL  \n",
       " 1        317          3          5    AAPL  \n",
       " 2        317          3          5    AAPL  \n",
       " 3        317          3          5    AAPL  \n",
       " 4        317          7          5    AAPL  \n",
       " \n",
       " [5 rows x 74 columns])"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "df = pd.read_csv(\"l1_day.csv\")\n",
    "\n",
    "df_info = df.info()\n",
    "df_head = df.head()\n",
    "\n",
    "df_info, df_head"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d0233b4f-8f78-449c-a806-8cda4f29060e",
   "metadata": {},
   "source": [
    "1. **Timestamp Parsing & Sorting**  \n",
    "   - Converts `ts_event` to datetime and sorts data by `ts_event` and `publisher_id` to ensure consistent ordering.\n",
    "\n",
    "2. **Deduplication**  \n",
    "   - For each unique combination of `ts_event` and `publisher_id`, only the first message is kept. This ensures one quote per venue per timestamp.\n",
    "\n",
    "3. **Grouping Snapshots**  \n",
    "   - Groups the deduplicated data by `ts_event` to form market \"snapshots\"—each representing the state of the market across venues at that moment.\n",
    "\n",
    "4. **Venue Structuring**  \n",
    "   - For each timestamp group, builds a list of venue dictionaries containing:\n",
    "     - `ask` price\n",
    "     - `ask_size`\n",
    "     - `fee` (hardcoded as 0.003)\n",
    "     - `rebate` (hardcoded as 0.002)\n",
    "\n",
    "This creates a list called `venue_snapshots`, which will be used for order allocation and backtesting."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "a181c2b1-2c04-4dfb-8b54-931c8c82981f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(54537,\n",
       " {'timestamp': Timestamp('2024-08-01 13:36:32.491911683+0000', tz='UTC'),\n",
       "  'venues': [{'id': 2,\n",
       "    'ask': 222.83,\n",
       "    'ask_size': 36,\n",
       "    'fee': 0.003,\n",
       "    'rebate': 0.002}]})"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "\n",
    "# Step 1: Preprocess snapshots - keep first message per publisher_id per ts_event\n",
    "df['ts_event'] = pd.to_datetime(df['ts_event'])\n",
    "df_sorted = df.sort_values(['ts_event', 'publisher_id'])\n",
    "df_deduped = df_sorted.drop_duplicates(subset=['ts_event', 'publisher_id'], keep='first')\n",
    "\n",
    "# Step 2: Group snapshots by timestamp\n",
    "snapshots = list(df_deduped.groupby('ts_event'))\n",
    "\n",
    "# Step 3: Create a venue snapshot structure for each timestamp\n",
    "venue_snapshots = []\n",
    "for ts, group in snapshots:\n",
    "    venues = []\n",
    "    for _, row in group.iterrows():\n",
    "        venue = {\n",
    "            'id': row['publisher_id'],\n",
    "            'ask': row['ask_px_00'],\n",
    "            'ask_size': row['ask_sz_00'],\n",
    "            'fee': 0.003,      # default placeholder\n",
    "            'rebate': 0.002    # default placeholder\n",
    "        }\n",
    "        venues.append(venue)\n",
    "    venue_snapshots.append({'timestamp': ts, 'venues': venues})\n",
    "\n",
    "len(venue_snapshots), venue_snapshots[0]  # show how many snapshots and the first one"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "77674547-96ab-4a0e-95be-b2e589161b49",
   "metadata": {},
   "source": [
    "### Static Allocator Implementation\n",
    "\n",
    "This code defines a **static order allocator** based on the Cont & Kukanov model, used to optimally split a trade across multiple venues to minimize expected cost.\n",
    "\n",
    "#### Components:\n",
    "\n",
    "- **`compute_cost(...)`**: \n",
    "  - Calculates the expected cost of a proposed order allocation (`split`) across venues.\n",
    "  - Includes execution cost, maker rebates, underfill/overfill penalties (`lambda_under`, `lambda_over`), and a queue risk penalty (`theta`).\n",
    "\n",
    "- **`allocate(...)`**:\n",
    "  - Performs an exhaustive search over all feasible order splits (in `step` size increments) across venues.\n",
    "  - Only considers allocations that exactly sum to the `order_size`.\n",
    "  - Returns the split with the lowest total cost.\n",
    "\n",
    "#### Usage:\n",
    "The last lines test the allocator on the first market snapshot for a 500-share order. It returns:\n",
    "- `test_split`: optimal number of shares to send to each venue.\n",
    "- `test_cost`: total expected cost of that allocation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "ad9bbbd2-68b5-463c-97b7-30040914af0c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "([], inf)"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from typing import List, Tuple, Dict\n",
    "from itertools import product\n",
    "\n",
    "# Helper function to compute cost based on allocation\n",
    "def compute_cost(split, venues, order_size, lambdao, lambdau, theta):\n",
    "    executed = 0\n",
    "    cash_spent = 0\n",
    "    for i in range(len(venues)):\n",
    "        exe = min(split[i], venues[i]['ask_size'])\n",
    "        executed += exe\n",
    "        cash_spent += exe * (venues[i]['ask'] + venues[i]['fee'])\n",
    "        maker_rebate = max(split[i] - exe, 0) * venues[i]['rebate']\n",
    "        cash_spent -= maker_rebate\n",
    "\n",
    "    underfill = max(order_size - executed, 0)\n",
    "    overfill = max(executed - order_size, 0)\n",
    "    risk_pen = theta * (underfill + overfill)\n",
    "    cost_pen = lambdau * underfill + lambdao * overfill\n",
    "    return cash_spent + risk_pen + cost_pen, executed\n",
    "\n",
    "# Allocator function from pseudocode\n",
    "def allocate(order_size, venues, lambda_over, lambda_under, theta_queue, step=100):\n",
    "    splits = [[]]\n",
    "    for v in range(len(venues)):\n",
    "        new_splits = []\n",
    "        for alloc in splits:\n",
    "            used = sum(alloc)\n",
    "            max_v = min(order_size - used, venues[v]['ask_size'])\n",
    "            for q in range(0, max_v + 1, step):\n",
    "                new_splits.append(alloc + [q])\n",
    "        splits = new_splits\n",
    "\n",
    "    best_cost = float('inf')\n",
    "    best_split = []\n",
    "    for alloc in splits:\n",
    "        if sum(alloc) != order_size:\n",
    "            continue\n",
    "        cost, _ = compute_cost(alloc, venues, order_size, lambda_over, lambda_under, theta_queue)\n",
    "        if cost < best_cost:\n",
    "            best_cost = cost\n",
    "            best_split = alloc\n",
    "    return best_split, best_cost\n",
    "\n",
    "# Let's test it on one snapshot to confirm the structure\n",
    "sample_snapshot = venue_snapshots[0]['venues']\n",
    "test_split, test_cost = allocate(order_size=500, venues=sample_snapshot, lambda_over=0.01, lambda_under=0.01, theta_queue=0.001)\n",
    "\n",
    "test_split, test_cost"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "c1ecc8bd-490f-4886-a397-78d9de5023af",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1, 36)"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Inspect total available size in the first snapshot\n",
    "sample_snapshot = venue_snapshots[0]['venues']\n",
    "total_ask_size = sum(v['ask_size'] for v in sample_snapshot)\n",
    "\n",
    "len(sample_snapshot), total_ask_size"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d69ff051-3e95-4b2a-8ead-1bf315d173a3",
   "metadata": {},
   "source": [
    "### Backtest Function\n",
    "\n",
    "This function simulates the execution of a 5,000-share order using the `allocate` method across a stream of venue snapshots.\n",
    "\n",
    "#### How It Works:\n",
    "\n",
    "- Iterates through each timestamped snapshot.\n",
    "- At each step:\n",
    "  - Filters out venues with zero available ask size.\n",
    "  - Determines how many shares to execute (`snapshot_order`), capped by remaining demand and available liquidity.\n",
    "  - Calls the `allocate` function to optimally split the order.\n",
    "  - Executes as many shares as possible at the venue's ask + fee.\n",
    "- Updates the total cash spent and number of shares filled.\n",
    "\n",
    "#### Returns:\n",
    "A dictionary with:\n",
    "- Tuned parameters (`lambda_over`, `lambda_under`, `theta_queue`)\n",
    "- Total cash spent\n",
    "- Average fill price\n",
    "- Number of shares filled"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "97d237f0-269d-4cf5-9053-dda7d83f2989",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'params': {'lambda_over': 0.01, 'lambda_under': 0.01, 'theta_queue': 0.001},\n",
       " 'cash_spent': 1113732.0,\n",
       " 'avg_fill_price': 222.7464,\n",
       " 'shares_filled': 5000}"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def backtest(order_size: int, lambda_over: float, lambda_under: float, theta_queue: float) -> Dict:\n",
    "    total_cash = 0.0\n",
    "    total_filled = 0\n",
    "    remaining = order_size\n",
    "\n",
    "    for snapshot in venue_snapshots:\n",
    "        venues = snapshot['venues']\n",
    "        if remaining <= 0:\n",
    "            break\n",
    "\n",
    "        # Only consider venues with non-zero ask size\n",
    "        venues = [v for v in venues if v['ask_size'] > 0]\n",
    "        if not venues:\n",
    "            continue\n",
    "\n",
    "        # Limit the per-snapshot order chunk to a feasible size\n",
    "        snapshot_order = min(remaining, sum(v['ask_size'] for v in venues))\n",
    "        if snapshot_order == 0:\n",
    "            continue\n",
    "\n",
    "        split, _ = allocate(snapshot_order, venues, lambda_over, lambda_under, theta_queue)\n",
    "\n",
    "        if not split:\n",
    "            continue\n",
    "\n",
    "        # Execute trades and update cash + filled shares\n",
    "        for i, v in enumerate(venues):\n",
    "            executed = min(split[i], v['ask_size'])\n",
    "            total_cash += executed * (v['ask'] + v['fee'])\n",
    "            total_filled += executed\n",
    "            remaining -= executed\n",
    "            if remaining <= 0:\n",
    "                break\n",
    "\n",
    "    avg_price = total_cash / total_filled if total_filled > 0 else 0.0\n",
    "    return {\n",
    "        'params': {'lambda_over': lambda_over, 'lambda_under': lambda_under, 'theta_queue': theta_queue},\n",
    "        'cash_spent': round(total_cash, 2),\n",
    "        'avg_fill_price': round(avg_price, 4),\n",
    "        'shares_filled': total_filled\n",
    "    }\n",
    "\n",
    "# Run one test case to verify\n",
    "test_result = backtest(5000, lambda_over=0.01, lambda_under=0.01, theta_queue=0.001)\n",
    "test_result"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d004dfc9-c90e-4b16-944c-8ffee50a9cb6",
   "metadata": {},
   "source": [
    "### Grid Search for Optimal Parameters\n",
    "\n",
    "This section performs a brute-force grid search to find the best combination of execution risk parameters:\n",
    "\n",
    "- `lambda_over`: Penalty for overfilling\n",
    "- `lambda_under`: Penalty for underfilling\n",
    "- `theta_queue`: Penalty for queue risk\n",
    "\n",
    "#### Process:\n",
    "\n",
    "1. Defines search ranges for each parameter.\n",
    "2. Runs the `backtest` function for every parameter combination (27 in total).\n",
    "3. Filters out results that fail to fill the full 5,000 shares.\n",
    "4. Selects the configuration with the **lowest total cost** among valid runs."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "b66d2783-d74b-4a0c-962a-baa4ae409c0e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'params': {'lambda_over': 0.0, 'lambda_under': 0.0, 'theta_queue': 0.0},\n",
       " 'cash_spent': 1113732.0,\n",
       " 'avg_fill_price': 222.7464,\n",
       " 'shares_filled': 5000}"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Define grid search ranges\n",
    "lambda_over_vals = [0.0, 0.01, 0.05]\n",
    "lambda_under_vals = [0.0, 0.01, 0.05]\n",
    "theta_vals = [0.0, 0.001, 0.005]\n",
    "\n",
    "# Store results\n",
    "results = []\n",
    "for lambdao, lambdau, theta in product(lambda_over_vals, lambda_under_vals, theta_vals):\n",
    "    result = backtest(order_size=5000, lambda_over= lambdao, lambda_under=lambdau, theta_queue=theta)\n",
    "    results.append(result)\n",
    "\n",
    "# Find the best result by lowest cash spent (assuming all 5000 shares filled)\n",
    "valid_results = [r for r in results if r['shares_filled'] == 5000]\n",
    "best_result = min(valid_results, key=lambda x: x['cash_spent'])\n",
    "\n",
    "best_result"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d95209bc-4bf7-4e3d-9c3c-355c1728dc43",
   "metadata": {},
   "source": [
    "### Baseline: Best Ask Strategy\n",
    "\n",
    "This function implements a naïve execution strategy where the order is greedily routed to the **venue offering the lowest ask price** at each snapshot.\n",
    "\n",
    "#### Logic:\n",
    "- Iterates through the venue snapshots.\n",
    "- At each step, selects the venue with the **lowest available ask**.\n",
    "- Buys as many shares as possible from that venue, up to the remaining order size.\n",
    "- Continues until 5,000 shares are filled or data runs out."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "f9741f49-a3ea-4b7e-bab4-de2f78c3624b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'cash_spent': 1114117.28, 'avg_fill_price': 222.8235, 'shares_filled': 5000}"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Helper: Best Ask Strategy — greedy fill from lowest ask across all venues per snapshot\n",
    "def baseline_best_ask(order_size: int) -> Dict:\n",
    "    total_cash = 0.0\n",
    "    total_filled = 0\n",
    "    remaining = order_size\n",
    "\n",
    "    for snapshot in venue_snapshots:\n",
    "        venues = snapshot['venues']\n",
    "        if remaining <= 0:\n",
    "            break\n",
    "\n",
    "        # Select best ask venue\n",
    "        best_venue = min(venues, key=lambda v: v['ask'] if v['ask_size'] > 0 else float('inf'), default=None)\n",
    "        if best_venue and best_venue['ask_size'] > 0:\n",
    "            fill = min(best_venue['ask_size'], remaining)\n",
    "            total_cash += fill * (best_venue['ask'] + best_venue['fee'])\n",
    "            total_filled += fill\n",
    "            remaining -= fill\n",
    "\n",
    "    avg_price = total_cash / total_filled if total_filled > 0 else 0.0\n",
    "    return {\n",
    "        'cash_spent': round(total_cash, 2),\n",
    "        'avg_fill_price': round(avg_price, 4),\n",
    "        'shares_filled': total_filled\n",
    "    }\n",
    "\n",
    "best_ask_result = baseline_best_ask(5000)\n",
    "best_ask_result"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e7e75204-5e56-4e9f-a75a-1c3d0dab44a7",
   "metadata": {},
   "source": [
    "### Baseline: TWAP (Time-Weighted Average Price) Strategy\n",
    "\n",
    "This function implements a **TWAP baseline**, where the total order is split into equal-sized chunks and executed uniformly over time.\n",
    "\n",
    "#### Logic:\n",
    "- Divides the 5,000-share order into `num_slices` equal parts (default: 30).\n",
    "- At fixed time intervals across the snapshot stream, attempts to fill one chunk.\n",
    "- Sorts venues by lowest ask price and fills as much of the chunk as possible from the cheapest venues.\n",
    "- Continues this process until all chunks are attempted or the order is filled.\n",
    "\n",
    "#### Output:\n",
    "Returns a dictionary with:\n",
    "- `cash_spent`: Total amount spent\n",
    "- `avg_fill_price`: Average price per share\n",
    "- `shares_filled`: Number of shares successfully executed\n",
    "\n",
    "This baseline mimics a typical execution algorithm that spreads trades evenly over time, without considering venue-specific liquidity conditions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "3cd03dbe-c1f2-4adf-8b45-38b0e241bb98",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'cash_spent': 776763.28, 'avg_fill_price': 223.0796, 'shares_filled': 3482}"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Helper: TWAP Baseline — equal chunks spread uniformly over time\n",
    "def baseline_twap(order_size: int, num_slices: int = 30) -> Dict:\n",
    "    chunk_size = order_size // num_slices\n",
    "    total_cash = 0.0\n",
    "    total_filled = 0\n",
    "    remaining = order_size\n",
    "\n",
    "    interval = len(venue_snapshots) // num_slices\n",
    "    for i in range(num_slices):\n",
    "        if remaining <= 0:\n",
    "            break\n",
    "        idx = i * interval\n",
    "        if idx >= len(venue_snapshots):\n",
    "            break\n",
    "        venues = venue_snapshots[idx]['venues']\n",
    "        ask_venues = sorted(venues, key=lambda v: v['ask'])\n",
    "\n",
    "        chunk = min(chunk_size, remaining)\n",
    "        for v in ask_venues:\n",
    "            if chunk <= 0:\n",
    "                break\n",
    "            fill = min(chunk, v['ask_size'])\n",
    "            total_cash += fill * (v['ask'] + v['fee'])\n",
    "            total_filled += fill\n",
    "            remaining -= fill\n",
    "            chunk -= fill\n",
    "\n",
    "    avg_price = total_cash / total_filled if total_filled > 0 else 0.0\n",
    "    return {\n",
    "        'cash_spent': round(total_cash, 2),\n",
    "        'avg_fill_price': round(avg_price, 4),\n",
    "        'shares_filled': total_filled\n",
    "    }\n",
    "\n",
    "twap_result = baseline_twap(5000)\n",
    "twap_result"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e4af32b8-f214-44ff-9650-9b03106623e8",
   "metadata": {},
   "source": [
    "### Baseline: VWAP (Volume-Weighted Average Price) Strategy\n",
    "\n",
    "This function implements a **VWAP-style baseline**, where order execution is weighted by the available liquidity (ask sizes) at each venue.\n",
    "\n",
    "#### Logic:\n",
    "- For each snapshot:\n",
    "  - Calculates the total available ask size across all venues.\n",
    "  - Allocates a portion of the remaining order to each venue proportional to its displayed ask size.\n",
    "- Fills are limited by both the venue’s liquidity and the remaining order size.\n",
    "- Continues until all 5,000 shares are filled or data runs out.\n",
    "\n",
    "#### Output:\n",
    "Returns a dictionary with:\n",
    "- `cash_spent`: Total cost of execution\n",
    "- `avg_fill_price`: Average price per share\n",
    "- `shares_filled`: Total number of shares executed"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "de5b21a9-8d39-46f6-afc7-fd31de0c8c8e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'cash_spent': 1114117.28, 'avg_fill_price': 222.8235, 'shares_filled': 5000}"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Helper: VWAP Baseline — size-weighted execution from best ask venues\n",
    "def baseline_vwap(order_size: int) -> Dict:\n",
    "    total_cash = 0.0\n",
    "    total_filled = 0\n",
    "    remaining = order_size\n",
    "\n",
    "    for snapshot in venue_snapshots:\n",
    "        if remaining <= 0:\n",
    "            break\n",
    "        venues = snapshot['venues']\n",
    "        total_size = sum(v['ask_size'] for v in venues if v['ask_size'] > 0)\n",
    "        if total_size == 0:\n",
    "            continue\n",
    "\n",
    "        for v in venues:\n",
    "            if v['ask_size'] <= 0:\n",
    "                continue\n",
    "            weight = v['ask_size'] / total_size\n",
    "            target_fill = int(min(remaining, order_size) * weight)\n",
    "            fill = min(target_fill, v['ask_size'], remaining)\n",
    "            total_cash += fill * (v['ask'] + v['fee'])\n",
    "            total_filled += fill\n",
    "            remaining -= fill\n",
    "            if remaining <= 0:\n",
    "                break\n",
    "\n",
    "    avg_price = total_cash / total_filled if total_filled > 0 else 0.0\n",
    "    return {\n",
    "        'cash_spent': round(total_cash, 2),\n",
    "        'avg_fill_price': round(avg_price, 4),\n",
    "        'shares_filled': total_filled\n",
    "    }\n",
    "\n",
    "vwap_result = baseline_vwap(5000)\n",
    "vwap_result"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c2609d41-f932-48f1-828a-a7cd4ecf928c",
   "metadata": {},
   "source": [
    "### Final Output and Evaluation\n",
    "\n",
    "This section generates the final JSON report summarizing the tuned smart order router performance and comparing it against benchmark strategies.\n",
    "\n",
    "#### Components:\n",
    "\n",
    "- **`bps_savings(...)`**: Computes basis point savings between the tuned strategy and each baseline using:\n",
    "  \\[\n",
    "  \\text{{bps}} = 10,000 \\times \\frac{{\\text{{baseline cost}} - \\text{{tuned cost}}}}{{\\text{{baseline cost}}}}\n",
    "  \\]\n",
    "\n",
    "- **`final_output` Dictionary**:\n",
    "  - `best_parameters`: Optimal values for `lambda_over`, `lambda_under`, and `theta_queue`.\n",
    "  - `tuned_strategy`: Cost and average price from the best allocation strategy.\n",
    "  - `baseline_*`: Results from Best Ask, TWAP, and VWAP.\n",
    "  - `bps_savings`: Relative savings vs. each baseline in basis points.\n",
    "\n",
    "- **`formatted_output`**:\n",
    "  - Outputs the entire result as a pretty-printed JSON object for logging, submission, or further analysis."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "894902c8-5711-4814-93cf-ee7cb28a01a3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'{\\n  \"best_parameters\": {\\n    \"lambda_over\": 0.0,\\n    \"lambda_under\": 0.0,\\n    \"theta_queue\": 0.0\\n  },\\n  \"tuned_strategy\": {\\n    \"cash_spent\": 1113732.0,\\n    \"avg_fill_price\": 222.7464\\n  },\\n  \"baseline_best_ask\": {\\n    \"cash_spent\": 1114117.28,\\n    \"avg_fill_price\": 222.8235,\\n    \"shares_filled\": 5000\\n  },\\n  \"baseline_twap\": {\\n    \"cash_spent\": 776763.28,\\n    \"avg_fill_price\": 223.0796,\\n    \"shares_filled\": 3482\\n  },\\n  \"baseline_vwap\": {\\n    \"cash_spent\": 1114117.28,\\n    \"avg_fill_price\": 222.8235,\\n    \"shares_filled\": 5000\\n  },\\n  \"bps_savings\": {\\n    \"vs_best_ask\": 3.46,\\n    \"vs_twap\": \"incomplete\",\\n    \"vs_vwap\": 3.46\\n  }\\n}'"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import json\n",
    "\n",
    "# Calculate savings in basis points\n",
    "def bps_savings(baseline_cost, tuned_cost):\n",
    "    return round(10000 * (baseline_cost - tuned_cost) / baseline_cost, 2)\n",
    "\n",
    "final_output = {\n",
    "    \"best_parameters\": best_result['params'],\n",
    "    \"tuned_strategy\": {\n",
    "        \"cash_spent\": best_result['cash_spent'],\n",
    "        \"avg_fill_price\": best_result['avg_fill_price']\n",
    "    },\n",
    "    \"baseline_best_ask\": best_ask_result,\n",
    "    \"baseline_twap\": twap_result,\n",
    "    \"baseline_vwap\": vwap_result,\n",
    "    \"bps_savings\": {\n",
    "        \"vs_best_ask\": bps_savings(best_ask_result['cash_spent'], best_result['cash_spent']),\n",
    "        \"vs_twap\": bps_savings(twap_result['cash_spent'], best_result['cash_spent']) if twap_result['shares_filled'] == 5000 else \"incomplete\",\n",
    "        \"vs_vwap\": bps_savings(vwap_result['cash_spent'], best_result['cash_spent'])\n",
    "    }\n",
    "}\n",
    "\n",
    "# Display as formatted JSON string\n",
    "formatted_output = json.dumps(final_output, indent=2)\n",
    "formatted_output"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
