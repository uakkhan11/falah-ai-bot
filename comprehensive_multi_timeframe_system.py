# ml_results_analyzer.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def analyze_ml_trades(csv_file='all_timeframes_ml_trades.csv'):
    """Comprehensive analysis of ML trading results"""
    
    print("🔍 COMPREHENSIVE ML TRADING ANALYSIS")
    print("=" * 50)
    
    # Load data
    df = pd.read_csv(csv_file)
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    df['exit_date'] = pd.to_datetime(df['exit_date'])
    
    # Basic stats
    total_trades = len(df)
    winning_trades = len(df[df['return_pct'] > 0])
    overall_win_rate = winning_trades / total_trades * 100
    total_pnl = df['pnl'].sum()
    
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"Total Trades: {total_trades:,}")
    print(f"Winning Trades: {winning_trades:,} ({overall_win_rate:.1f}%)")
    print(f"Total PnL: ₹{total_pnl:,.0f}")
    print(f"Average PnL per Trade: ₹{df['pnl'].mean():.0f}")
    
    # 1. Performance by Timeframe
    print(f"\n🕐 PERFORMANCE BY TIMEFRAME:")
    print("-" * 40)
    tf_analysis = df.groupby('timeframe').agg({
        'pnl': ['count', 'sum', 'mean'],
        'return_pct': lambda x: (x > 0).mean() * 100,
        'ml_confidence': 'mean',
        'exit_reason': lambda x: (x == 'Profit Target').mean() * 100
    }).round(2)
    
    tf_analysis.columns = ['Trades', 'Total_PnL', 'Avg_PnL', 'Win_Rate', 'Avg_Confidence', 'Profit_Target_%']
    print(tf_analysis.to_string())
    
    # 2. Performance by ML Confidence Levels
    print(f"\n🎯 PERFORMANCE BY ML CONFIDENCE:")
    print("-" * 35)
    df['confidence_bucket'] = pd.cut(df['ml_confidence'], 
                                    bins=[0.6, 0.7, 0.8, 0.9, 1.0], 
                                    labels=['0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9+'])
    
    conf_analysis = df.groupby('confidence_bucket').agg({
        'pnl': ['count', 'sum', 'mean'],
        'return_pct': lambda x: (x > 0).mean() * 100,
        'exit_reason': lambda x: (x == 'Profit Target').mean() * 100
    }).round(2)
    
    conf_analysis.columns = ['Trades', 'Total_PnL', 'Avg_PnL', 'Win_Rate', 'Profit_Target_%']
    print(conf_analysis.to_string())
    
    # 3. Top and Bottom Performing Symbols
    print(f"\n🏆 TOP 10 PERFORMING SYMBOLS:")
    print("-" * 30)
    symbol_perf = df.groupby('symbol').agg({
        'pnl': ['count', 'sum', 'mean'],
        'return_pct': lambda x: (x > 0).mean() * 100,
        'ml_confidence': 'mean'
    }).round(2)
    
    symbol_perf.columns = ['Trades', 'Total_PnL', 'Avg_PnL', 'Win_Rate', 'Avg_Confidence']
    top_symbols = symbol_perf.nlargest(10, 'Total_PnL')
    print(top_symbols.to_string())
    
    print(f"\n❌ BOTTOM 5 PERFORMING SYMBOLS:")
    print("-" * 32)
    bottom_symbols = symbol_perf.nsmallest(5, 'Total_PnL')
    print(bottom_symbols.to_string())
    
    # 4. Exit Reason Analysis
    print(f"\n🚪 EXIT REASON DISTRIBUTION:")
    print("-" * 28)
    exit_analysis = df.groupby('exit_reason').agg({
        'pnl': ['count', 'sum', 'mean'],
        'return_pct': ['mean', 'std'],
        'bars_held': 'mean'
    }).round(2)
    
    exit_analysis.columns = ['Count', 'Total_PnL', 'Avg_PnL', 'Avg_Return', 'Return_Std', 'Avg_Bars_Held']
    print(exit_analysis.to_string())
    
    # 5. Monthly Performance Trend
    print(f"\n📅 MONTHLY PERFORMANCE TREND:")
    print("-" * 29)
    df['month'] = df['entry_date'].dt.to_period('M')
    monthly_perf = df.groupby('month').agg({
        'pnl': ['count', 'sum'],
        'return_pct': lambda x: (x > 0).mean() * 100
    }).round(2)
    
    monthly_perf.columns = ['Trades', 'Total_PnL', 'Win_Rate']
    print(monthly_perf.to_string())
    
    # 6. Risk Metrics
    print(f"\n⚠️ RISK METRICS:")
    print("-" * 14)
    
    # Consecutive losses
    df_sorted = df.sort_values(['symbol', 'entry_date'])
    df_sorted['is_loss'] = df_sorted['pnl'] < 0
    
    max_consecutive_losses = 0
    current_consecutive = 0
    
    for loss in df_sorted['is_loss']:
        if loss:
            current_consecutive += 1
            max_consecutive_losses = max(max_consecutive_losses, current_consecutive)
        else:
            current_consecutive = 0
    
    # Drawdown analysis
    df_sorted['cumulative_pnl'] = df_sorted['pnl'].cumsum()
    df_sorted['running_max'] = df_sorted['cumulative_pnl'].expanding().max()
    df_sorted['drawdown'] = df_sorted['cumulative_pnl'] - df_sorted['running_max']
    
    max_drawdown = df_sorted['drawdown'].min()
    
    print(f"Maximum Consecutive Losses: {max_consecutive_losses}")
    print(f"Maximum Drawdown: ₹{max_drawdown:,.0f}")
    print(f"Largest Single Loss: ₹{df['pnl'].min():,.0f}")
    print(f"Largest Single Gain: ₹{df['pnl'].max():,.0f}")
    print(f"Standard Deviation of Returns: {df['return_pct'].std():.3f}")
    
    # 7. Recommendations
    print(f"\n💡 KEY INSIGHTS & RECOMMENDATIONS:")
    print("-" * 35)
    
    best_tf = tf_analysis.loc[tf_analysis['Total_PnL'].idxmax()].name
    best_conf = conf_analysis.loc[conf_analysis['Win_Rate'].idxmax()].name
    
    recommendations = [
        f"✅ Best Timeframe: {best_tf} (₹{tf_analysis.loc[best_tf, 'Total_PnL']:,.0f} total PnL)",
        f"✅ Optimal Confidence: {best_conf} range ({conf_analysis.loc[best_conf, 'Win_Rate']:.1f}% win rate)",
        f"✅ Focus Symbols: {', '.join(top_symbols.head(3).index.tolist())}",
        f"⚠️ Risk Management: Max {max_consecutive_losses} consecutive losses observed",
        f"📈 Profit Targets: {(df['exit_reason'] == 'Profit Target').mean()*100:.1f}% of exits hit profit targets"
    ]
    
    for rec in recommendations:
        print(f"  {rec}")
    
    # Save detailed analysis
    symbol_perf.to_csv('symbol_performance_analysis.csv')
    tf_analysis.to_csv('timeframe_performance_analysis.csv')
    monthly_perf.to_csv('monthly_performance_analysis.csv')
    
    print(f"\n📁 FILES SAVED:")
    print("  • symbol_performance_analysis.csv")
    print("  • timeframe_performance_analysis.csv") 
    print("  • monthly_performance_analysis.csv")

if __name__ == "__main__":
    analyze_ml_trades()
