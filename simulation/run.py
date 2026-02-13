from assignment_simpy import SimPyAssignment
import pandas as pd
from tqdm import tqdm
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "forecasting_bench", "granite-tsfm")))

def run_simulation(scheduler_type, dataset_config, model,quantile=0.5):
    base_dir = f"../outputs/decision/{model}/{dataset_config['dataset']}"
    results_dir = f"{base_dir}/results_{scheduler_type}.csv"
    results = pd.DataFrame(columns=['offset', 'cpu_util', 'VR','VS', 'PAR', 'PICP', 'MPIW', 'WinklerScore'])
    
    for test_split_offset in tqdm(range(288,313), desc=f"handling {scheduler_type}-{model}-{dataset_config['dataset']}-{quantile}", unit="offset"):
        assignment_obj = SimPyAssignment(
            num_hosts=dataset_config['num_hosts'],
            host_capacity=dataset_config['host_capacity'],
            threshold_rate=dataset_config['threshold_rate'],
            scheduler_type=scheduler_type,
            forecast_path=f"{base_dir}/{test_split_offset}",
            quantile=quantile
        )
        assignment_obj.schedule()
        risk_metrics = assignment_obj.MonteCarloSimulation(num_simulations=100, confidence_level=0.95, random_seed=42)
        with open(f"{base_dir}/{test_split_offset}/mapping_{scheduler_type}.json", "w") as f:
            json.dump(assignment_obj.mapping, f, indent=4)

        metrics = {
            'offset': test_split_offset,
            'cpu_util': assignment_obj.compute_util(),
            'VR': assignment_obj.compute_SLAV(),
            'VS': assignment_obj.compute_CVS(),
            'PAR': assignment_obj.compute_PAR(),
            **risk_metrics
        }
        if metrics['offset'] not in results['offset'].values:
            results = pd.concat([results, pd.DataFrame([metrics])], ignore_index=True)
        else:
            results.loc[results['offset'] == metrics['offset'], metrics.keys()] = metrics.values()

    results.sort_values('offset', inplace=True)
    valid_mask = (results[['cpu_util', 'VR', 'VS', 'PAR','PICP', 'MPIW', 'WinklerScore']] >= 0).all(axis=1)
    valid_results = results[valid_mask]
    mean_results = pd.DataFrame([{
        'offset': 'mean',
        'cpu_util': valid_results['cpu_util'].mean(),
        'VR': valid_results['VR'].mean(),
        'VS': valid_results['VS'].mean(),
        'PAR': valid_results['PAR'].mean(),
        'PICP': valid_results['PICP'].mean(),
        'MPIW': valid_results['MPIW'].mean(),
        'WinklerScore': valid_results['WinklerScore'].mean(),
    }])

    final_results = pd.concat([results, mean_results], ignore_index=True)
    final_results.to_csv(results_dir, index=False)

def main():
    scheduler_types = ["FFD","BFD","ACO","Gurobi"]
    datasets_config = [{"dataset":"huawei2025","num_hosts":32,"host_capacity":4,"threshold_rate":1},{"dataset":"azure2019","num_hosts":18,"host_capacity":4,"threshold_rate":1},{"dataset":"borg2019_d_trace","num_hosts":14,"host_capacity":1,"threshold_rate":1},{"dataset":"borg2019_e_trace","num_hosts":6,"host_capacity":1,"threshold_rate":1} ]
    models = ["autoarima","autoets","autotheta","chronos2","moirai2","sundial","kairos_50m","toto","timesfm25","flowstate-9.1m","deepar","d_linear","patch_tst","tft"]

    for scheduler_type in scheduler_types:
        for model in models:
            for dataset_config in datasets_config:
                for quantile in [0.5]:
                    run_simulation(scheduler_type, dataset_config, model,quantile)

if __name__ == "__main__":
    main()
