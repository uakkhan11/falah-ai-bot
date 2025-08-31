# comprehensive_ml_all_timeframes_with_exports.py

import os
import pandas as pd
import numpy as np
import ta
import warnings
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

BASE_DIR = "/root/falah-ai-bot"
DATA_PATHS = {
    '15minute': os.path.join(BASE_DIR, "scalping_data"),
    '1hour':    os.path.join(BASE_DIR, "intraday_swing_data"),
    'daily':    os.path.join(BASE_DIR, "swing_data"),
}

TIMEFRAME_CONFIGS = {
    '1minute':  {'source': '15minute', 'agg': 1, 'pt': 0.015, 'sl': 0.010, 'mh': 60},
    '5minute':  {'source': '15minute', 'agg': 3, 'pt': 0.020, 'sl': 0.012, 'mh': 36},
    '15minute': {'source': '15minute', 'agg': 1, 'pt': 0.025, 'sl': 0.015, 'mh': 25},
    '1hour':    {'source': '1hour',    'agg': 1, 'pt': 0.035, 'sl': 0.020, 'mh': 24},
    '4hour':    {'source': '1hour',    'agg': 4, 'pt': 0.050, 'sl': 0.025, 'mh': 12},
    'daily':    {'source': 'daily',    'agg': 1, 'pt': 0.060, 'sl': 0.030, 'mh': 10},
}

def list_symbols():
    return [f[:-4] for f in os.listdir(DATA_PATHS['daily']) if f.endswith('.csv')]

def load_resample(symbol, cfg):
    path = os.path.join(DATA_PATHS[cfg['source']], f"{symbol}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'].dt.year == 2025].reset_index(drop=True)
    if len(df) < 50:
        return None
    if cfg['agg'] > 1:
        bars = []
        for i in range(0, len(df) - cfg['agg'] + 1, cfg['agg']):
            chunk = df.iloc[i:i+cfg['agg']]
            bars.append({
                'date': chunk['date'].iloc[-1],
                'open': chunk['open'].iloc[0],
                'high': chunk['high'].max(),
                'low': chunk['low'].min(),
                'close': chunk['close'].iloc[-1],
                'volume': chunk['volume'].sum()
            })
        df = pd.DataFrame(bars)
    return df if len(df) >= 50 else None

def make_features(df):
    df = df.copy().reset_index(drop=True)
    df['ret'] = df['close'].pct_change().fillna(0)
    df['rsi'] = ta.momentum.rsi(df['close'], 14).fillna(50)
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], 14).fillna(25)
    df['sma20'] = df['close'].rolling(20).mean().fillna(method='bfill')
    df['vol_r'] = df['volume'] / df['volume'].rolling(20).mean().fillna(1)
    return df.fillna(method='bfill').fillna(method='ffill')

def backtest_symbol(df, model, scaler, cfg, symbol, timeframe):
    """Detailed backtest for individual symbol with trade tracking"""
    trades = []
    position = None
    
    for i in range(20, len(df) - cfg['mh']):  # Start after feature window
        current_bar = df.iloc[i]
        
        # Exit existing position
        if position:
            bars_held = i - position['entry_idx']
            current_return = (current_bar['close'] - position['entry_price']) / position['entry_price']
            
            exit_reason = None
            if current_return >= cfg['pt']:
                exit_reason = 'Profit Target'
            elif current_return <= -cfg['sl']:
                exit_reason = 'Stop Loss'
            elif bars_held >= cfg['mh']:
                exit_reason = 'Max Hold'
            
            if exit_reason:
                pnl = current_return * 100000  # Assuming 100k position size
                trade = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'entry_date': position['entry_date'],
                    'exit_date': current_bar['date'],
                    'entry_price': position['entry_price'],
                    'exit_price': current_bar['close'],
                    'return_pct': current_return,
                    'pnl': pnl,
                    'bars_held': bars_held,
                    'exit_reason': exit_reason,
                    'ml_confidence': position['confidence']
                }
                trades.append(trade)
                position = None
        
        # Check for new entry
        if position is None:
            # Get features for current bar
            features = [current_bar['ret'], current_bar['rsi'], current_bar['adx'], 
                       current_bar['sma20'], current_bar['vol_r']]
            
            if not any(np.isnan(features)):
                features_scaled = scaler.transform([features])
                prediction = model.predict(features_scaled)[0]
                confidence = model.predict_proba(features_scaled)[0][1]
                
                # Enter if model predicts success with high confidence
                if prediction == 1 and confidence > 0.6:
                    position = {
                        'entry_idx': i,
                        'entry_date': current_bar['date'],
                        'entry_price': current_bar['close'],
                        'confidence': confidence
                    }
    
    return trades

def train_backtest(symbols, timeframe):
    cfg = TIMEFRAME_CONFIGS[timeframe]
    all_features, all_targets = [], []
    symbol_data = {}
    
    # Collect data from all symbols
    for symbol in symbols:
        df = load_resample(symbol, cfg)
        if df is None:
            continue
        df = make_features(df)
        
        # Create targets
        targets = []
        for i in range(len(df) - cfg['mh']):
            entry_price = df['close'].iloc[i]
            hit_profit = False
            hit_stop = False
            
            for j in range(1, cfg['mh'] + 1):
                future_price = df['close'].iloc[i + j]
                ret = (future_price - entry_price) / entry_price
                if ret >= cfg['pt']:
                    targets.append(1)
                    hit_profit = True
                    break
                elif ret <= -cfg['sl']:
                    targets.append(0)
                    hit_stop = True
                    break
            
            if not hit_profit and not hit_stop:
                final_price = df['close'].iloc[min(i + cfg['mh'], len(df) - 1)]
                final_ret = (final_price - entry_price) / entry_price
                targets.append(1 if final_ret > 0 else 0)
        
        targets.extend([0] * cfg['mh'])  # Pad to match df length
        
        # Extract features
        features = df[['ret', 'rsi', 'adx', 'sma20', 'vol_r']].values
        
        # Clean data
        mask = ~np.isnan(features).any(axis=1)
        clean_features = features[mask]
        clean_targets = np.array(targets)[mask]
        
        if len(clean_features) > 50:
            all_features.append(clean_features)
            all_targets.append(clean_targets)
            symbol_data[symbol] = df
    
    if not all_features:
        print(f"{timeframe}: No valid data")
        return
    
    # Combine all data and train model
    X_combined = np.vstack(all_features)
    y_combined = np.concatenate(all_targets)
    
    if len(np.unique(y_combined)) < 2:
        print(f"{timeframe}: Only one class present")
        return
    
    # Scale and train
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_combined, test_size=0.3, random_state=42, stratify=y_combined
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    # Backtest each symbol individually and collect detailed trades
    all_trades = []
    for symbol, df in symbol_data.items():
        trades = backtest_symbol(df, model, scaler, cfg, symbol, timeframe)
        all_trades.extend(trades)
    
    # Calculate summary statistics
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['return_pct'] > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # EXPORT DETAILED TRADES TO CSV - THIS IS WHERE THE EXPORT HAPPENS
        trades_df.to_csv(f"{timeframe}_ml_trades_detailed.csv", index=False)
        
        print(f"{timeframe}: Acc {test_acc*100:.1f}%, Trades {total_trades}, Wins {winning_trades} ({win_rate:.1f}%)")
        print(f"  Detailed trades saved to: {timeframe}_ml_trades_detailed.csv")
    else:
        print(f"{timeframe}: No trades generated")

if __name__ == "__main__":
    # Clean up any existing CSV files
    for f in glob.glob("*_ml_trades_detailed.csv"):
        os.remove(f)
    
    symbols = list_symbols()
    
    for tf in ['1minute', '5minute', '15minute', '1hour', '4hour', 'daily']:
        print(f"=== ML {tf} ===")
        train_backtest(symbols, tf)
    
    # COMBINE ALL TIMEFRAMES INTO MASTER FILE - THIS IS WHERE THE COMBINATION HAPPENS
    print("\n=== Combining all results ===")
    all_files = glob.glob("*_ml_trades_detailed.csv")
    
    if all_files:
        combined_trades = []
        for file in all_files:
            df = pd.read_csv(file)
            combined_trades.append(df)
        
        master_df = pd.concat(combined_trades, ignore_index=True)
        master_df.to_csv("all_timeframes_ml_trades.csv", index=False)
        
        print(f"Master file created: all_timeframes_ml_trades.csv")
        print(f"Total trades across all timeframes: {len(master_df)}")
    else:
        print("No trade files found to combine")
