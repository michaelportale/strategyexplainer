"""Analysis script for batch backtest results - find your edge in the data."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BatchResultsAnalyzer:
    """Comprehensive analysis of batch backtest results."""
    
    def __init__(self, csv_path: str, json_path: str = None):
        """Initialize analyzer with results data.
        
        Args:
            csv_path: Path to batch results CSV
            json_path: Optional path to detailed JSON results
        """
        self.csv_path = csv_path
        self.json_path = json_path
        
        # Load data
        self.df = pd.read_csv(csv_path)
        self.metadata = None
        
        if json_path and Path(json_path).exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
                self.metadata = data.get('metadata', {})
        
        # Clean and prepare data
        self._prepare_data()
        
    def _prepare_data(self):
        """Clean and prepare the data for analysis."""
        # Filter successful runs only
        self.df_success = self.df[self.df['run_status'] == 'SUCCESS'].copy()
        
        # Convert numeric columns
        numeric_cols = [
            'total_return', 'annual_return', 'monthly_return', 'daily_return_avg',
            'annual_volatility', 'max_drawdown', 'sharpe_ratio', 'sortino_ratio',
            'win_rate', 'total_trades', 'profit_factor', 'expectancy', 'largest_win', 
            'largest_loss', 'avg_win', 'avg_loss', 'max_consecutive_wins', 
            'max_consecutive_losses', 'calmar_ratio', 'recovery_factor', 'kelly_criterion'
        ]
        
        for col in numeric_cols:
            if col in self.df_success.columns:
                self.df_success[col] = pd.to_numeric(self.df_success[col], errors='coerce')
        
        # Create parameter combination labels
        self.df_success['param_combo'] = (
            self.df_success['ticker'] + '_' +
            self.df_success['stop_loss_pct'].astype(str) + '_' +
            self.df_success['take_profit_pct'].astype(str) + '_' +
            self.df_success['max_drawdown_pct'].astype(str) + '_' +
            self.df_success['position_size_pct'].astype(str)
        )
        
        print(f"✅ Loaded {len(self.df)} total runs ({len(self.df_success)} successful)")
    
    def summary_stats(self):
        """Display summary statistics."""
        print("\n" + "="*60)
        print("📊 BATCH BACKTEST SUMMARY STATISTICS")
        print("="*60)
        
        if self.metadata:
            print(f"🕐 Run timestamp: {self.metadata.get('timestamp', 'N/A')}")
            print(f"🎯 Total combinations: {self.metadata.get('total_combinations', 'N/A')}")
            print(f"✅ Successful runs: {self.metadata.get('successful_runs', 'N/A')}")
            print(f"❌ Failed runs: {self.metadata.get('failed_runs', 'N/A')}")
            success_rate = self.metadata.get('successful_runs', 0) / self.metadata.get('total_combinations', 1) * 100
            print(f"📈 Success rate: {success_rate:.1f}%")
        
        if len(self.df_success) > 0:
            print(f"\n📋 KEY METRICS ACROSS ALL SUCCESSFUL RUNS:")
            metrics_summary = self.df_success[['total_return', 'annual_return', 'sharpe_ratio', 
                                             'max_drawdown', 'win_rate', 'total_trades']].describe()
            print(metrics_summary.round(4))
    
    def top_performers(self, n: int = 10):
        """Show top performing strategies by various metrics."""
        print(f"\n🏆 TOP {n} PERFORMERS")
        print("="*60)
        
        metrics = [
            ('total_return', 'Total Return', '{:.1%}'),
            ('sharpe_ratio', 'Sharpe Ratio', '{:.2f}'),
            ('calmar_ratio', 'Calmar Ratio', '{:.2f}'),
            ('profit_factor', 'Profit Factor', '{:.2f}')
        ]
        
        for metric, label, fmt in metrics:
            if metric in self.df_success.columns:
                print(f"\n📈 By {label}:")
                top_n = self.df_success.nlargest(n, metric)
                for i, (_, row) in enumerate(top_n.iterrows(), 1):
                    ret = row['total_return']
                    sharpe = row['sharpe_ratio']
                    dd = row['max_drawdown']
                    trades = row['total_trades']
                    ticker = row['ticker']
                    sl = row['stop_loss_pct']
                    tp = row['take_profit_pct']
                    
                    print(f"  {i:2d}. {ticker} | SL:{sl} TP:{tp} | "
                          f"Ret:{ret:.1%} Sharpe:{sharpe:.2f} DD:{dd:.1%} Trades:{trades:.0f}")
    
    def parameter_analysis(self):
        """Analyze performance by parameter values."""
        print(f"\n🔍 PARAMETER IMPACT ANALYSIS")
        print("="*60)
        
        # Analyze by ticker
        print("\n📊 By Ticker:")
        ticker_stats = self.df_success.groupby('ticker').agg({
            'total_return': ['mean', 'std', 'count'],
            'sharpe_ratio': 'mean',
            'max_drawdown': 'mean'
        }).round(3)
        ticker_stats.columns = ['Avg_Return', 'Std_Return', 'Count', 'Avg_Sharpe', 'Avg_MaxDD']
        print(ticker_stats.sort_values('Avg_Return', ascending=False))
        
        # Analyze by stop loss
        print("\n🛑 By Stop Loss:")
        sl_stats = self.df_success.groupby('stop_loss_pct').agg({
            'total_return': ['mean', 'count'],
            'sharpe_ratio': 'mean',
            'max_drawdown': 'mean',
            'win_rate': 'mean'
        }).round(3)
        sl_stats.columns = ['Avg_Return', 'Count', 'Avg_Sharpe', 'Avg_MaxDD', 'Avg_WinRate']
        print(sl_stats.sort_values('Avg_Return', ascending=False))
        
        # Analyze by take profit
        print("\n💰 By Take Profit:")
        tp_stats = self.df_success.groupby('take_profit_pct').agg({
            'total_return': ['mean', 'count'],
            'sharpe_ratio': 'mean',
            'profit_factor': 'mean'
        }).round(3)
        tp_stats.columns = ['Avg_Return', 'Count', 'Avg_Sharpe', 'Avg_ProfitFactor']
        print(tp_stats.sort_values('Avg_Return', ascending=False))
        
        # Analyze by position size
        print("\n📏 By Position Size:")
        pos_stats = self.df_success.groupby('position_size_pct').agg({
            'total_return': ['mean', 'count'],
            'sharpe_ratio': 'mean',
            'max_drawdown': 'mean'
        }).round(3)
        pos_stats.columns = ['Avg_Return', 'Count', 'Avg_Sharpe', 'Avg_MaxDD']
        print(pos_stats.sort_values('Avg_Return', ascending=False))
    
    def risk_return_analysis(self):
        """Analyze risk-return characteristics."""
        print(f"\n⚖️  RISK-RETURN ANALYSIS")
        print("="*60)
        
        # Risk-return buckets
        df = self.df_success.copy()
        
        # Create risk buckets based on max drawdown
        df['risk_bucket'] = pd.cut(df['max_drawdown'], 
                                  bins=[-1, -0.20, -0.10, -0.05, 0], 
                                  labels=['High Risk (>20%)', 'Med Risk (10-20%)', 
                                         'Low Risk (5-10%)', 'Very Low Risk (<5%)'])
        
        # Create return buckets
        df['return_bucket'] = pd.cut(df['total_return'], 
                                   bins=[-1, 0, 0.10, 0.25, 1], 
                                   labels=['Negative', 'Low (0-10%)', 
                                          'Medium (10-25%)', 'High (>25%)'])
        
        # Risk-return matrix
        print("\n📊 Risk-Return Distribution (Count of Strategies):")
        risk_return_matrix = pd.crosstab(df['risk_bucket'], df['return_bucket'], margins=True)
        print(risk_return_matrix)
        
        # Best risk-adjusted performers
        print(f"\n🛡️  BEST RISK-ADJUSTED PERFORMERS (Sharpe > 1.0):")
        high_sharpe = df[df['sharpe_ratio'] > 1.0].nlargest(10, 'sharpe_ratio')
        for _, row in high_sharpe.iterrows():
            print(f"  {row['ticker']} | Sharpe: {row['sharpe_ratio']:.2f} | "
                  f"Return: {row['total_return']:.1%} | MaxDD: {row['max_drawdown']:.1%}")
    
    def generate_heatmaps(self, save_plots: bool = True):
        """Generate heatmaps for parameter analysis."""
        print(f"\n🎨 GENERATING HEATMAPS...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Parameter Performance Heatmaps', fontsize=16, fontweight='bold')
        
        # Define heatmap configurations
        heatmap_configs = [
            {
                'values': 'total_return',
                'index': 'stop_loss_pct',
                'columns': 'take_profit_pct',
                'title': 'Stop Loss vs Take Profit\n(Average Total Return)',
                'format': '.2%',
                'ax': axes[0,0]
            },
            {
                'values': 'sharpe_ratio',
                'index': 'stop_loss_pct',
                'columns': 'take_profit_pct',
                'title': 'Stop Loss vs Take Profit\n(Average Sharpe Ratio)',
                'format': '.2f',
                'ax': axes[0,1]
            },
            {
                'values': 'total_return',
                'index': 'ticker',
                'columns': 'position_size_pct',
                'title': 'Ticker vs Position Size\n(Average Total Return)',
                'format': '.2%',
                'ax': axes[1,0]
            },
            {
                'values': 'total_return',
                'index': 'max_drawdown_pct',
                'columns': 'position_size_pct',
                'title': 'Max Drawdown vs Position Size\n(Average Total Return)',
                'format': '.2%',
                'ax': axes[1,1]
            }
        ]
        
        # Generate each heatmap with safety checks
        for config in heatmap_configs:
            try:
                # Create pivot table
                pivot = self.df_success.pivot_table(
                    values=config['values'],
                    index=config['index'],
                    columns=config['columns'],
                    aggfunc='mean'
                )
                
                # Only plot if pivot table has data
                if not pivot.empty and pivot.size > 0:
                    sns.heatmap(pivot, annot=True, fmt=config['format'], cmap='RdYlGn', 
                               ax=config['ax'], cbar_kws={'label': config['title'].split('\n')[1]})
                    config['ax'].set_title(config['title'])
                else:
                    # Hide empty subplot
                    config['ax'].set_visible(False)
                    print(f"  ⚠️  Skipping empty heatmap: {config['title']}")
                    
            except Exception as e:
                print(f"  ❌ Error generating heatmap '{config['title']}': {e}")
                config['ax'].set_visible(False)
        
        plt.tight_layout()
        
        if save_plots:
            output_path = Path(self.csv_path).parent / f"heatmaps_{Path(self.csv_path).stem}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  💾 Saved heatmaps to: {output_path}")
        
        plt.show()
    
    def correlation_analysis(self):
        """Analyze correlations between parameters and performance."""
        print(f"\n🔗 CORRELATION ANALYSIS")
        print("="*60)
        
        # Select numeric parameter columns
        param_cols = ['stop_loss_pct', 'take_profit_pct', 'max_drawdown_pct', 'position_size_pct']
        perf_cols = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor']
        
        # Convert categorical to numeric for correlation
        df_corr = self.df_success.copy()
        
        # One-hot encode ticker
        ticker_dummies = pd.get_dummies(df_corr['ticker'], prefix='ticker')
        df_corr = pd.concat([df_corr, ticker_dummies], axis=1)
        
        # Calculate correlations with performance metrics
        all_cols = param_cols + perf_cols + [col for col in ticker_dummies.columns]
        corr_matrix = df_corr[all_cols].corr()
        
        # Show correlations with key performance metrics
        print("\n📈 Correlations with Total Return:")
        return_corrs = corr_matrix['total_return'].drop('total_return').sort_values(key=abs, ascending=False)
        for param, corr in return_corrs.head(10).items():
            print(f"  {param:20s}: {corr:6.2f}")
            
        print("\n🛡️  Correlations with Sharpe Ratio:")
        sharpe_corrs = corr_matrix['sharpe_ratio'].drop('sharpe_ratio').sort_values(key=abs, ascending=False)
        for param, corr in sharpe_corrs.head(10).items():
            print(f"  {param:20s}: {corr:6.2f}")
    
    def export_insights(self, output_file: str = None):
        """Export key insights and top strategies to file."""
        if output_file is None:
            output_file = Path(self.csv_path).parent / f"insights_{Path(self.csv_path).stem}.txt"
        
        with open(output_file, 'w') as f:
            f.write("BATCH BACKTEST INSIGHTS REPORT\n")
            f.write("="*50 + "\n\n")
            
            # Top strategies
            f.write("TOP 10 STRATEGIES BY TOTAL RETURN:\n")
            f.write("-" * 40 + "\n")
            top_10 = self.df_success.nlargest(10, 'total_return')
            for i, (_, row) in enumerate(top_10.iterrows(), 1):
                f.write(f"{i:2d}. {row['ticker']} | SL:{row['stop_loss_pct']} TP:{row['take_profit_pct']} "
                       f"| Return:{row['total_return']:.1%} Sharpe:{row['sharpe_ratio']:.2f}\n")
            
            # Parameter insights
            f.write(f"\n\nPARAMETER INSIGHTS:\n")
            f.write("-" * 20 + "\n")
            
            # Best ticker
            best_ticker = self.df_success.groupby('ticker')['total_return'].mean().idxmax()
            best_ticker_return = self.df_success.groupby('ticker')['total_return'].mean().max()
            f.write(f"Best performing ticker: {best_ticker} (avg return: {best_ticker_return:.1%})\n")
            
            # Best stop loss
            best_sl = self.df_success.groupby('stop_loss_pct')['total_return'].mean().idxmax()
            best_sl_return = self.df_success.groupby('stop_loss_pct')['total_return'].mean().max()
            f.write(f"Best stop loss setting: {best_sl} (avg return: {best_sl_return:.1%})\n")
            
            # Best take profit  
            best_tp = self.df_success.groupby('take_profit_pct')['total_return'].mean().idxmax()
            best_tp_return = self.df_success.groupby('take_profit_pct')['total_return'].mean().max()
            f.write(f"Best take profit setting: {best_tp} (avg return: {best_tp_return:.1%})\n")
            
        print(f"💾 Insights exported to: {output_file}")
    
    def run_full_analysis(self, save_plots: bool = True):
        """Run complete analysis suite."""
        self.summary_stats()
        self.top_performers()
        self.parameter_analysis() 
        self.risk_return_analysis()
        self.correlation_analysis()
        self.generate_heatmaps(save_plots)
        self.export_insights()


def find_latest_results(output_dir: str = "backend/outputs") -> tuple:
    """Find the most recent batch results files."""
    output_path = Path(output_dir)
    
    # Find latest CSV
    csv_files = list(output_path.glob("batch_summary_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No batch summary CSV files found in {output_dir}")
    
    latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
    
    # Find corresponding JSON
    timestamp = latest_csv.stem.replace("batch_summary_", "")
    json_file = output_path / f"batch_detailed_{timestamp}.json"
    
    return str(latest_csv), str(json_file) if json_file.exists() else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze batch backtest results")
    parser.add_argument("--csv", help="Path to batch results CSV file")
    parser.add_argument("--json", help="Path to detailed JSON results file")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top performers to show")
    
    args = parser.parse_args()
    
    try:
        # Use provided files or find latest
        if args.csv:
            csv_path = args.csv
            json_path = args.json
        else:
            print("🔍 Finding latest batch results...")
            csv_path, json_path = find_latest_results()
            print(f"📊 Using: {csv_path}")
            if json_path:
                print(f"📋 Using: {json_path}")
        
        # Run analysis
        analyzer = BatchResultsAnalyzer(csv_path, json_path)
        analyzer.run_full_analysis(save_plots=not args.no_plots)
        
        print(f"\n🎯 Analysis complete! Your edge is in the data.")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Run batch_backtest.py first to generate results.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise 