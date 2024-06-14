import pandas as pd
import numpy as np
import timeit
import time

def knapsack_dynamic(weights, values, capacity):
    n = len(weights)
    dp = np.zeros((n+1, capacity+1))

    for i in range(1, n+1):
        for w in range(1, capacity+1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1]
                               [w-weights[i-1]] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]

    # Rekonstrukcja rozwiązania
    w = capacity
    items_selected = []
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            items_selected.append(i-1)
            w -= weights[i-1]

    return dp, dp[n][capacity], items_selected


def knapsack_greedy(weights, values, capacity):
    n = len(weights)
    ratio = [(values[i]/weights[i], i) for i in range(n)]
    ratio.sort(reverse=True, key=lambda x: x[0])

    total_value = 0
    total_weight = 0
    items_selected = []

    for r, i in ratio:
        if total_weight + weights[i] <= capacity:
            items_selected.append(i)
            total_weight += weights[i]
            total_value += values[i]

    return total_value, items_selected


def read_input(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    n = int(lines[0].strip())
    weights = list(map(int, lines[1].strip().split()))
    values = list(map(int, lines[2].strip().split()))
    capacity = int(lines[3].strip())

    return n, weights, values, capacity


def run_experiments(weights, values, capacity):
    n = len(weights)

    # Dynamic Programming
    dp_start_time = timeit.default_timer()
    dp, max_value_dp, items_selected_dp = knapsack_dynamic(
        weights, values, capacity)
    dp_end_time = timeit.default_timer()
    dp_time = (dp_end_time - dp_start_time) * 1000

    # Greedy Algorithm
    greedy_start_time = timeit.default_timer()
    max_value_greedy, items_selected_greedy = knapsack_greedy(
        weights, values, capacity)
    greedy_end_time = timeit.default_timer()
    greedy_time = (greedy_end_time - greedy_start_time) * 1000

    # Calculate relative error
    relative_error = (max_value_dp - max_value_greedy) / max_value_dp

    # Results
    return dp_time, greedy_time, relative_error, max_value_dp, max_value_greedy


def save_results_to_file(results, file_name):
    df = pd.DataFrame(results)
    df.to_csv(file_name, index=False)
    print(f"Wyniki zapisane do {file_name}")


def main():
    file_path = 'plecak.txt'

    # Read initial data from file
    n, weights, values, initial_capacity = read_input(file_path)

    # Experiments for varying number of containers
    results_varying_n = {
        'Number of Containers': [],
        'DP Time (ms)': [],
        'Greedy Time (ms)': [],
        'Relative Error': [],
        'DP Value': [],
        'Greedy Value': []
    }

    for i in range(1, n+1):  # Increase number of containers from 1 to n
        dp_time, greedy_time, relative_error, max_value_dp, max_value_greedy = run_experiments(
            weights[:i], values[:i], initial_capacity)
        results_varying_n['Number of Containers'].append(i)
        results_varying_n['DP Time (ms)'].append(dp_time)
        results_varying_n['Greedy Time (ms)'].append(greedy_time)
        results_varying_n['Relative Error'].append(relative_error)
        results_varying_n['DP Value'].append(max_value_dp)
        results_varying_n['Greedy Value'].append(max_value_greedy)

    save_results_to_file(results_varying_n, "results_varying_n.csv")

    # Experiments for varying capacity (b)
    results_varying_capacity = {
        'Capacity': [],
        'DP Time (ms)': [],
        'Greedy Time (ms)': [],
        'Relative Error': [],
        'DP Value': [],
        'Greedy Value': []
    }

    for capacity in range(initial_capacity // 2, initial_capacity * 2 + 1, initial_capacity // 2):  # Adjust capacity
        dp_time, greedy_time, relative_error, max_value_dp, max_value_greedy = run_experiments(
            weights, values, capacity)
        results_varying_capacity['Capacity'].append(capacity)
        results_varying_capacity['DP Time (ms)'].append(dp_time)
        results_varying_capacity['Greedy Time (ms)'].append(greedy_time)
        results_varying_capacity['Relative Error'].append(relative_error)
        results_varying_capacity['DP Value'].append(max_value_dp)
        results_varying_capacity['Greedy Value'].append(max_value_greedy)
        print(relative_error)


    czas = int(time.time())
    file_name = f"results_varying_capacity_{czas}.csv"
    save_results_to_file(results_varying_capacity,file_name)

if __name__ == "__main__":
    main()
