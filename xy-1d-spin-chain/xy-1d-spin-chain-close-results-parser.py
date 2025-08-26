import os
import json

import matplotlib.pyplot as plt

from collections import Counter

results_dir = 'results/summary'


overall_summary = {
    'seeds': [],
    'num_samples': [],
    'num_iterations': [],
}

# duration summary is a dictionary that will hold the durations for each model and the samples * iterations value
# example => 'FF': {1000: [120.5, 100.2], 2000: [130.2, ...], ...} where 1000 is the samples * iterations value
duration_summary = {
    'FF': {},
    'FF2': {},
    'Jastrow': {},
    'RBM': {},
}

# for FF, FF2, Jastrow we want to know learning rate, iterations, samples, seed, time to calculate, percent error
# for RBM we want to know alpha, iterations, samples, seed, time to calculate, percent error

summary_data = {
    'FF': {
        'learning_rate': [],
        'iterations': [],
        'samples': [],
        'seed': [],
        'seconds_to_calculate': [],
        'percent_error': []
    },
    'FF2': {
        'learning_rate': [],
        'iterations': [],
        'samples': [],
        'seed': [],
        'seconds_to_calculate': [],
        'percent_error': []
    },
    'Jastrow': {
        'learning_rate': [],
        'iterations': [],
        'samples': [],
        'seed': [],
        'seconds_to_calculate': [],
        'percent_error': []
    },
    'RBM': {
        'alpha': [],
        'iterations': [],
        'samples': [],
        'seed': [],
        'seconds_to_calculate': [],
        'percent_error': []
    }
}

# this will be used to count which combinations of parameters produced results
combo_counts = {
    'FF': {},
    'FF2': {},
    'Jastrow': {},
    'RBM': {},
}

for file in os.listdir(results_dir):
    if file.endswith('.json'):
        file_path = os.path.join(results_dir, file)
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                for entry in data:
                    model_type = entry['model_type']
                    match model_type:
                        case 'FF' | 'FF2' | 'Jastrow':
                            combo_key = (entry['learning_rate'], entry['num_iterations'], entry['num_samples'], entry['seed'])
                            if combo_key not in combo_counts[model_type]:
                                combo_counts[model_type][combo_key] = 0
                            combo_counts[model_type][combo_key] += 1
                            summary_data[model_type]['learning_rate'].append(entry['learning_rate'])
                            summary_data[model_type]['iterations'].append(entry['num_iterations'])
                            summary_data[model_type]['samples'].append(entry['num_samples'])
                            summary_data[model_type]['seed'].append(entry['seed'])
                            summary_data[model_type]['seconds_to_calculate'].append(entry['seconds_to_calculate'])
                        case 'RBM' | 'RBMSymmetry':
                            combo_key = (entry['alpha'], entry['num_iterations'], entry['num_samples'], entry['seed'])
                            if combo_key not in combo_counts[model_type]:
                                combo_counts[model_type][combo_key] = 0
                            combo_counts[model_type][combo_key] += 1
                            summary_data[model_type]['alpha'].append(entry['alpha'])
                            summary_data[model_type]['iterations'].append(entry['num_iterations'])
                            summary_data[model_type]['samples'].append(entry['num_samples'])
                            summary_data[model_type]['seed'].append(entry['seed'])
                            summary_data[model_type]['seconds_to_calculate'].append(entry['seconds_to_calculate'])
                        case _:
                            print(f"Unknown model type: {model_type}")
                    exact_solution_energy = entry['exact_solution_energy']
                    ground_state_energy = entry['ground_state_energy']['mean']
                    percent_error = abs((exact_solution_energy - ground_state_energy) / exact_solution_energy) * 100

                    summary_data[model_type]['percent_error'].append(percent_error)
                    
                    overall_summary['seeds'].append(entry['seed'])
                    overall_summary['num_samples'].append(entry['num_samples'])
                    overall_summary['num_iterations'].append(entry['num_iterations'])
                    
                    # Update duration summary
                    samples_iterations = entry['num_samples'] * entry['num_iterations']
                    if model_type not in duration_summary:
                        duration_summary[model_type] = {}
                    if samples_iterations not in duration_summary[model_type]:
                        duration_summary[model_type][samples_iterations] = []
                    duration_summary[model_type][samples_iterations].append(entry['seconds_to_calculate'])

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in {file_path}: {e}")
            except Exception as e:
                print(f"An error occurred while processing {file_path}: {e}")

# print overall summary data
print("Overall Summary Data:")
print(f"  Total Seeds: {len(set(overall_summary['seeds']))}")
print(f"  Total Samples: {sum(overall_summary['num_samples'])}")
print(f"  Total Iterations: {sum(overall_summary['num_iterations'])}")
for key, values in overall_summary.items():
    frequency = Counter(values)
    sorted_frequency = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
    print(f"  {key} (Frequency): {dict(sorted_frequency)}")
    # graph the frequency of each metric
    plt.figure(figsize=(10, 5))
    plt.bar([str(k) for k in sorted_frequency], [v for k, v in sorted_frequency], color='blue')
    plt.title(f"Overall - {key} Frequency")
    plt.xlabel(key)
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'results/graphs/overall_{key}_frequency.png')
    plt.close()



# print summary data
for model_type, metrics in summary_data.items():
    print(f"Model Type: {model_type}")
    print(f"  Total Entries: {len(metrics['percent_error'])}")
    for metric, values in metrics.items():
        if metric == 'percent_error':
            print(f"  Average Percent Error: {sum(values) / len(values) if values else 0:.2f}%")
        elif metric == 'seconds_to_calculate':
            print(f"  Average Seconds to Calculate: {sum(values) / len(values) if values else 0:.2f} seconds")
        else:
            frequency = Counter(values)
            sorted_frequency = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
            print(f"  {metric} (Frequency): {dict(sorted_frequency)}")
            # graph the frequency of each metric
            plt.figure(figsize=(10, 5))
            plt.bar([str(k) for k in sorted_frequency], [v for k, v in sorted_frequency], color='blue')
            plt.title(f"{model_type} - {metric} Frequency")
            plt.xlabel(metric)
            plt.ylabel('Frequency')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'results/graphs/{model_type}_{metric}_frequency.png')
            plt.close()
    print()

# print combo counts
print("Combination Counts:")
for model_type, combos in combo_counts.items():
    print(f"Model Type: {model_type}")
    print(f"  Total Combinations: {len(combos)}")
    sorted_combos = sorted(combos.items(), key=lambda x: x[1], reverse=True)
    # Print the top 10 combinations
    print(f"  Top Combinations (up to 10):")
    for combo, count in sorted_combos[:10]:
        print(f"    Parameters: {combo}, Count: {count}")   
    #for combo, count in sorted_combos:
    #    print(f"  Parameters: {combo}, Count: {count}")
    print()

# print duration summary
print("Duration Summary:")
for model_type, durations in duration_summary.items():
    print(f"Model Type: {model_type}")
    sorted_samples_iterations = sorted(durations.items())
    for samples_iterations, times in sorted_samples_iterations:
        average_time = sum(times) / len(times) if times else 0
        print(f"  Samples * Iterations: {samples_iterations}, Count: {len(times)}, Average Duration: {average_time:.2f} seconds")

