import os
import json

import matplotlib.pyplot as plt

results_dir = 'results/scratch/10k-run-1/out'
results = []
logfiles = 0
model_types = set()
graph_nodes = set()
result_files_set = set()
log_files_set = set()

for root, dirs, files in os.walk(results_dir):
    for file in files:
        if file.endswith('-results.json'):
            result_files_set.add(file.split('-results.json')[0])
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                    results.append(data)
                    model_types.add(data['model_type'])
                    graph_nodes.add(data['graph_nodes'])
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in {file_path}: {e}")
        if file.endswith('.log'):
            logfiles += 1
            log_files_set.add(file.split('.log')[0])

print(f"Loaded {len(results)} results.json files.")
print(f"Found {logfiles} log files.")
print("Number of runs that timed out:", logfiles - len(results))
print("Runs that timed out:")
for run in log_files_set - result_files_set:
    print(run)
print()


for model_type in model_types:
    model_results = [result for result in results if result['model_type'] == model_type]
    if model_results:
        for graph_node in graph_nodes:
            filtered_results = [result for result in model_results if result['graph_nodes'] == graph_node]
            if filtered_results:
                exact_solution_energy = filtered_results[0]['exact_solution_energy']
                # save results that are close to the exact solution
                close = 0.1
                close_results = [result for result in filtered_results if abs(result['ground_state_energy']['mean'] - exact_solution_energy) < close * abs(exact_solution_energy)]
                # write the close results to a file
                with open(f'results/summary/close_results_{model_type}_graph_{graph_node}.json', 'w') as f:
                    json.dump(close_results, f, indent=4)
                print(f"Model Type: {model_type}, Graph Node: {graph_node}")
                print(f"There are {len(close_results)} results within {close * 100:.1f}% to the exact solution:", exact_solution_energy)
                print(f"Percentage of results within {close * 100:.1f}% to the exact solution: {len(close_results) / len(filtered_results) * 100:.2f}%")
                means = [result['ground_state_energy']['mean'] for result in filtered_results if result['ground_state_energy']['mean'] > -50]
                plt.figure(figsize=(10, 5))
                plt.title(f'Model Type: {model_type}, Graph Node: {graph_node}')
                plt.xlabel('Run Index')
                plt.ylabel('Ground State Energy Mean')
                plt.grid()
                plt.axhline(y=exact_solution_energy, linewidth=2, color='r', label='Exact')
                plt.scatter(range(len(means)), means, marker='o', label=f'Graph Node: {graph_node}')
                plt.legend()
                plt.savefig(f'results/graphs/{model_type}_graph_{graph_node}_ground_state_energy.png')
                plt.close()
