import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_equity_curve(df, output_dir="outputs", filename="equity_curve.png"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["Equity"], label="Equity Curve", color="blue")
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_drawdown(df, output_dir="outputs", filename="drawdown.png"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rolling_max = df["Equity"].cummax()
    drawdown = (df["Equity"] - rolling_max) / rolling_max

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, drawdown, label="Drawdown", color="red")
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

if __name__ == "__main__":
    df = pd.read_csv("backtest_results.csv", index_col=0, parse_dates=True)
    plot_equity_curve(df)
    plot_drawdown(df)