import numpy as np
import timeit

def knapsack_dynamic(weights, values, capacity):
    n = len(weights)
    dp = np.zeros((n + 1, capacity + 1))

    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]

    # Rekonstrukcja rozwiązania
    w = capacity
    items_selected = []
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            items_selected.append(i - 1)
            w -= weights[i - 1]

    return dp, dp[n][capacity], items_selected

def knapsack_greedy(weights, values, capacity):
    n = len(weights)
    ratio = [(values[i] / weights[i], i) for i in range(n)]
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
    dp_time = timeit.timeit(lambda: knapsack_dynamic(weights, values, capacity), number=1) * 1000
    dp, max_value_dp, items_selected_dp = knapsack_dynamic(weights, values, capacity)

    # Greedy Algorithm
    greedy_time = timeit.timeit(lambda: knapsack_greedy(weights, values, capacity), number=1) * 1000
    max_value_greedy, items_selected_greedy = knapsack_greedy(weights, values, capacity)

    # Calculate relative error
    relative_error = (max_value_dp - max_value_greedy) / max_value_dp

    # Results
    return dp_time, greedy_time, relative_error, max_value_dp, max_value_greedy, dp

def print_dp_table(dp):
    print("Dynamic Programming Table (dp):")
    for row in dp:
        print(" ".join(f"{int(val):2}" for val in row))
    print()

def main():
    file_path = 'plecak.txt'

    # Read initial data from file
    n, weights, values, initial_capacity = read_input(file_path)

    # Run experiment for the given input
    dp_time, greedy_time, relative_error, max_value_dp, max_value_greedy, dp = run_experiments(weights, values, initial_capacity)

    # Print results
    print("Results for the given input:")
    print(f"Number of Containers: {n}")
    print(f"Capacity: {initial_capacity}")
    print(f"DP Time (ms): {dp_time:.2f}")
    print(f"Greedy Time (ms): {greedy_time:.2f}")
    print(f"Relative Error: {relative_error:.2f}")
    print(f"DP Value: {max_value_dp}")
    print(f"Greedy Value: {max_value_greedy}")
    print()

    # Print DP table
    print_dp_table(dp)

if __name__ == "__main__":
    main()
