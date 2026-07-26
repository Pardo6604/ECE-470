import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves to file instead of opening a window
import matplotlib.pyplot as plt

# ── Configuration (all tuneable parameters in one place) ─────────────────────
CONFIG = {
    "data_path": "stock_data_with_sharpe_ratio.pkl",
    "risk_free_rate": 0.04,       # annual risk-free rate (e.g. ~4% Canadian T-bill)
    "population_size": 100,
    "num_generations": 200,
    "mutation_rate": 0.1,
    "tournament_size": 5,
    "elite_count": 5,
    "trading_days_per_year": 252,
}


# ── Data Loading ──────────────────────────────────────────────────────────────
def load_data(path):
    raw = pd.read_pickle(path)
    closes = pd.DataFrame({ticker: df["Close"] for ticker, df in raw.items()})
    closes.index = pd.to_datetime(closes.index, utc=True)
    return closes.sort_index().dropna()


def compute_returns(closes):
    return closes.pct_change().dropna()


# ── Fitness Function: Portfolio Sharpe Ratio ──────────────────────────────────
def sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate, trading_days):
    w = np.array(weights) / 100.0
    port_return = np.dot(w, mean_returns) * trading_days
    port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix * trading_days, w)))
    if port_vol == 0:
        return -np.inf
    return (port_return - risk_free_rate) / port_vol


# ── Chromosome Helpers ────────────────────────────────────────────────────────
def normalize_to_100(weights):
    """Scale a list of non-negative numbers to integer weights summing to 100."""
    total = sum(weights)
    if total == 0:
        return equal_weight_chromosome(len(weights))
    scaled = [int(w * 100 / total) for w in weights]
    remainder = 100 - sum(scaled)
    # Distribute remainder to indices with highest fractional parts
    fractions = sorted(
        range(len(weights)), key=lambda i: (weights[i] * 100 / total) % 1, reverse=True
    )
    for i in range(remainder):
        scaled[fractions[i]] += 1
    return scaled


def equal_weight_chromosome(n):
    """First parent: divide money evenly across all stocks."""
    base = 100 // n
    weights = [base] * n
    remainder = 100 - sum(weights)
    for i in range(remainder):
        weights[i] += 1
    return weights


def random_chromosome(n):
    """Random valid integer allocation summing to 100, zeros allowed."""
    weights = [0] * n
    for _ in range(100):
        weights[random.randint(0, n - 1)] += 1
    return weights


# ── Initial Population ────────────────────────────────────────────────────────
def init_population(n_stocks, pop_size):
    population = [equal_weight_chromosome(n_stocks)]  # required first parent
    while len(population) < pop_size:
        population.append(random_chromosome(n_stocks))
    return population


# ── Selection: Tournament ─────────────────────────────────────────────────────
def tournament_select(population, fitness_scores, tournament_size):
    candidates = random.sample(list(enumerate(population)), tournament_size)
    winner = max(candidates, key=lambda x: fitness_scores[x[0]])
    return winner[1]


# ── Crossover: Single-Point ───────────────────────────────────────────────────
def crossover(parent1, parent2):
    n = len(parent1)
    point = random.randint(1, n - 1)
    child = parent1[:point] + parent2[point:]
    return normalize_to_100(child)


# ── Mutation: Transfer weight between two stocks ──────────────────────────────
def mutate(weights, mutation_rate):
    weights = weights[:]
    n = len(weights)
    for _ in range(n):
        if random.random() < mutation_rate:
            donors = [i for i in range(n) if weights[i] > 0]
            if len(donors) < 1:
                continue
            i = random.choice(donors)
            j = random.choice([k for k in range(n) if k != i])
            transfer = random.randint(1, weights[i])
            weights[i] -= transfer
            weights[j] += transfer
    return weights


# ── Genetic Algorithm Main Loop ───────────────────────────────────────────────
def run_ga(mean_returns, cov_matrix, config):
    n_stocks = len(mean_returns)
    pop_size = config["population_size"]
    n_gen = config["num_generations"]
    mut_rate = config["mutation_rate"]
    tourn_size = config["tournament_size"]
    elite_count = config["elite_count"]
    rf = config["risk_free_rate"]
    td = config["trading_days_per_year"]

    population = init_population(n_stocks, pop_size)
    best_per_gen = []

    for gen in range(n_gen):
        fitness = [
            sharpe_ratio(ind, mean_returns, cov_matrix, rf, td) for ind in population
        ]
        best_per_gen.append(max(fitness))

        if (gen + 1) % 50 == 0:
            print(f"  Generation {gen + 1}/{n_gen} — Best Sharpe: {max(fitness):.4f}")

        ranked = sorted(zip(fitness, population), key=lambda x: x[0], reverse=True)
        next_gen = [ind for _, ind in ranked[:elite_count]]

        while len(next_gen) < pop_size:
            p1 = tournament_select(population, fitness, tourn_size)
            p2 = tournament_select(population, fitness, tourn_size)
            child = crossover(p1, p2)
            child = mutate(child, mut_rate)
            next_gen.append(child)

        population = next_gen

    fitness = [sharpe_ratio(ind, mean_returns, cov_matrix, rf, td) for ind in population]
    best_idx = int(np.argmax(fitness))
    return population[best_idx], fitness[best_idx], best_per_gen


# ── Baselines ─────────────────────────────────────────────────────────────────
def evaluate_baselines(n_stocks, mean_returns, cov_matrix, config):
    rf = config["risk_free_rate"]
    td = config["trading_days_per_year"]

    equal = equal_weight_chromosome(n_stocks)
    rand = random_chromosome(n_stocks)

    return {
        "Equal Weight": (equal, sharpe_ratio(equal, mean_returns, cov_matrix, rf, td)),
        "Random":       (rand,  sharpe_ratio(rand,  mean_returns, cov_matrix, rf, td)),
    }


# ── Visualizations ────────────────────────────────────────────────────────────
def plot_convergence(best_per_gen):
    plt.figure(figsize=(10, 5))
    plt.plot(best_per_gen, color="steelblue", linewidth=2)
    plt.title("Sharpe Ratio Convergence Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Best Sharpe Ratio")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("convergence.png", dpi=150)
    plt.close()
    print("Saved: convergence.png")


def plot_allocation(tickers, weights, title):
    pairs = [(t, w) for t, w in zip(tickers, weights) if w > 0]
    pairs.sort(key=lambda x: x[1], reverse=True)
    labels, vals = zip(*pairs)
    plt.figure(figsize=(12, 5))
    plt.bar(labels, vals, color="steelblue")
    plt.title(title)
    plt.xlabel("Stock")
    plt.ylabel("Allocation (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fname = title.replace(" ", "_") + ".png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")


def plot_comparison(results):
    names = list(results.keys())
    sharpes = [results[n]["sharpe"] for n in names]
    colors = ["steelblue" if n == "GA Portfolio" else "lightcoral" for n in names]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, sharpes, color=colors)
    plt.title("Portfolio Comparison — Sharpe Ratio")
    plt.ylabel("Sharpe Ratio")
    plt.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, sharpes):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig("comparison.png", dpi=150)
    plt.close()
    print("Saved: comparison.png")


# ── Entry Point ───────────────────────────────────────────────────────────────
def main():
    closes = load_data(CONFIG["data_path"])
    tickers = list(closes.columns)
    n_stocks = len(tickers)
    daily_returns = compute_returns(closes)
    mean_returns = daily_returns.mean().values
    cov_matrix = daily_returns.cov().values

    print(f"Loaded {n_stocks} stocks, {len(daily_returns)} trading days.")
    print(f"Running GA: {CONFIG['num_generations']} generations, "
          f"population {CONFIG['population_size']}...\n")

    best_weights, best_sharpe, convergence = run_ga(mean_returns, cov_matrix, CONFIG)
    baselines = evaluate_baselines(n_stocks, mean_returns, cov_matrix, CONFIG)

    results = {
        "GA Portfolio":    {"weights": best_weights,            "sharpe": best_sharpe},
        "Equal Weight":    {"weights": baselines["Equal Weight"][0], "sharpe": baselines["Equal Weight"][1]},
        "Random Portfolio":{"weights": baselines["Random"][0],       "sharpe": baselines["Random"][1]},
    }

    print("\n── Portfolio Comparison ──────────────────────────")
    for name, res in results.items():
        print(f"  {name:20s}: Sharpe = {res['sharpe']:.4f}")

    print("\n── GA Best Allocation ────────────────────────────")
    for ticker, w in sorted(zip(tickers, best_weights), key=lambda x: x[1], reverse=True):
        if w > 0:
            print(f"  {ticker}: {w}%")
    print(f"  Total: {sum(best_weights)}%")

    plot_convergence(convergence)
    plot_allocation(tickers, best_weights, "GA Portfolio Allocation")
    plot_comparison(results)


if __name__ == "__main__":
    main()
