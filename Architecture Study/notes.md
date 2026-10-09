# MSML612 Project Notes: Transfer Learning for SPY Option Return Prediction

## 1. Layman's Overview

* **Core Goal**: Test if pretraining a neural network on historical stock market data (SPY ETF) helps it better predict whether buying a short-term stock option (a Call or Put) will actually make money after accounting for trading costs and broker fees[cite: 1, 2].
* **The Problem**: SPY daily option data with consistent weekday expiration cycles is relatively limited (starting late 2022)[cite: 1]. Additionally, high-frequency intraday prices contain noise[cite: 1]. Pretraining on older stock price data (2016–2021) aims to teach the model general morning market patterns before tuning it to predict option returns[cite: 1, 2].
* **Daily Workflow**:
  1. Every morning at 10:00 AM Eastern, the system analyzes intraday stock movements from 9:30 AM to 9:59 AM[cite: 1].
  2. It identifies one Call option and one Put option (nearest at-the-money, expiring on the next available trading day—no 0DTEs)[cite: 1].
  3. The model predicts the 1-hour return for both options[cite: 1, 2].
  4. If the highest predicted return exceeds a preset cost/profit threshold, it simulates buying that contract at 10:01 AM and selling it at 11:01 AM[cite: 1, 3]. Otherwise, it stays in cash[cite: 1, 3].
* **Primary Question**: Does pretraining on stock data make predictions more accurate and profitable compared to training the same network from scratch on option data alone[cite: 1]?

---

## 2. Proposed Architecture & Implementation Steps

### Model Architecture Breakdown

flowchart TD
    %% Inputs
    subgraph Inputs ["30-Minute Input Sequence (9:30–9:59 AM)"]
        SF["<b>Stock Features</b><br/>• Returns<br/>• Range/Body Normalization<br/>• Log Volume"]
        MI["<b>Market Indicators</b><br/>• Overnight Gap<br/>• Opening Range Width/Pos<br/>• Volatility"]
    end

    %% Backbone
    TCN["<b>Temporal Convolutional Network (TCN) Backbone</b><br/>• 3 Convolutional Blocks (~32 channels per block)<br/>• Pooling Layers<br/>• Dropout"]

    Rep["<b>30-Min Morning Representation</b>"]

    %% Stage 1: Pretraining Branch
    subgraph Stage1 ["Stage 1: Stock Pretraining"]
        PH["<b>Pretraining Heads</b><br/>• Target 1: 60-min SPY Return<br/>• Target 2: Log Realized Volatility<br/><i>(Standardized, MSE Loss)</i>"]
    end

    %% Stage 2: Fine-Tuning Branch
    subgraph Stage2 ["Stage 2: Fine-Tuning / Option Head"]
        OH["<b>Option Prediction Head</b><br/>Combines TCN Representation with Contract Features:<br/>• Moneyness & Relative Premium<br/>• Time to Expiration & Bid-Ask Spread"]
        Out["<b>Outputs: Call Return & Put Return</b><br/><i>(Huber Loss)</i>"]
    end

    %% Flow Connections
    Inputs --> TCN
    TCN --> Rep
    Rep --> Stage1
    Rep --> Stage2
    OH --> Out

    %% Styling
    classDef inputStyle fill:#e1f5fe,stroke:#0288d1,stroke-width:1.5px,color:#01579b;
    classDef tcnStyle fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#e65100;
    classDef repStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1.5px,color:#4a148c;
    classDef s1Style fill:#e8f5e9,stroke:#388e3c,stroke-width:1.5px,color:#1b5e20;
    classDef s2Style fill:#fbe9e7,stroke:#d84315,stroke-width:1.5px,color:#bf360c;

    class Inputs,SF,MI inputStyle;
    class TCN tcnStyle;
    class Rep repStyle;
    class PH s1Style;
    class OH,Out s2Style;

1. **Backbone**: A compact Temporal Convolutional Network (TCN) consisting of **3 convolutional blocks** (~32 channels each) with pooling and dropout to process 30-minute sequence data without overparameterization[cite: 2].
2. **Pretraining Heads (Stage 1)**: Temporary heads that output two targets over the next 60 minutes: standard SPY return and log realized volatility[cite: 2].
3. **Option Prediction Head (Stage 2)**: Replaces the pretraining heads with a dense layer head that concatenates the learned sequence representation with static option contract features (moneyness, relative premium, time to expiration, relative bid-ask spread) to predict 1-hour net Call and Put returns[cite: 2].

---

### Potential Implementation Steps (?)

#### Step 1: Data Acquisition & Preprocessing
* **Datasets**:
  * *Stock Pretraining*: SPY 1-minute OHLCV bars (Jan 2016 – Dec 2021)[cite: 2].
  * *Stock Validation*: Jan 2022 – Oct 2022[cite: 2].
  * *Options Training*: Nov 21, 2022 – Dec 2024 (~530 trading days)[cite: 2].
  * *Options Validation*: 2025[cite: 2].
  * *Options Test*: Jan 2026 – Aug 2026 (untouched holdout)[cite: 2].
* **Feature Extraction**:
  * *Minute-level*: Log volume, minute returns, normalized candle ranges and bodies[cite: 2].
  * *Day-level*: Overnight gap, morning realized volatility, opening-range width and price position[cite: 2].
  * *Option-level*: Relative bid-ask spread, moneyness, premium relative to SPY, remaining expiration time[cite: 2].
* **Strict Rules**: Partition-based normalization only (no data leakage across time splits); maintain full days intact[cite: 2].

#### Step 2: Stage 1 Pretraining (TCN)
* Train the TCN backbone + stock prediction heads using 30-minute sliding windows (sampled every 30 minutes) from 2016–2021 stock data[cite: 2].
* Optimize using equally weighted Mean Squared Error (MSE) loss against standardized SPY return and volatility targets[cite: 2].

#### Step 3: Stage 2 Fine-Tuning & Scratch Model Baseline
* Swap in the option head[cite: 2].
* **Pretrained Model**: Fine-tune the entire network on the option training set to predict Call/Put returns using Huber loss[cite: 2].
* **Scratch Model**: Train the exact same network architecture initialized with random weights on the same option budget and limited learning-rate search[cite: 2].
* **Data Ablation Experiments**: Run both settings on 100% and recent 50% of option training data across 3 fixed random seeds[cite: 3].

#### Step 4: Model Evaluation & Benchmarking
* Evaluate prediction accuracy using Mean Absolute Error (MAE) as the primary metric, along with RMSE and seed variability[cite: 3].
* Compare against baselines: Training-period constant forecast and XGBoost with flattened minute observations + contract features[cite: 3].
* Compute paired error differences using 5-day block bootstrap confidence intervals[cite: 3].

#### Step 5: Backtesting & Trading Execution Strategy
* Average predicted returns across 3 seeds per neural configuration[cite: 3].
* Select the trade candidate with higher predicted return, subject to a validation-tuned decision threshold (0%, 1%, 2%, or 5%) requiring at least 30 trades and positive validation net profit[cite: 3].
* Simulate trading with $10,000 initial cash, max 1 contract per day, 10:01 AM entry fill, 11:01 AM exit fill, including execution delay, broker fees, and sensitivity testing ($0.01/$0.02 adverse slippage per side)[cite: 1, 2, 3].
* Report Net Profit, Profit per Trade, Trade Count, No-Trade Frequency, and Max Drawdown[cite: 3].