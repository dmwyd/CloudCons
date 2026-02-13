import os
import pandas as pd
import glob

BASE_DIR = "outputs/decision"
models = [
    "autoarima",
    "autoets",
    "autotheta",
    "d_linear",
    "deepar",
    "patch_tst",
    "tft",
    "moirai2",
    "chronos2",
    "timesfm25",
    "sundial",
    "toto",
    "flowstate-9.1m",
    "kairos_50m",
]
dataset_name = [
    "azure2019",
    "huawei2025",
    "borg2019_d_trace",
    "borg2019_e_trace",
]

def process_decision_files():
    base_dir = BASE_DIR
    all_results = []

    for model_dir in models:
        model_path = os.path.join(base_dir, model_dir)
        if not os.path.isdir(model_path):
            continue

        for dataset_dir in dataset_name:
            dataset_path = os.path.join(model_path, dataset_dir)
            if not os.path.isdir(dataset_path):
                continue

            csv_files = glob.glob(os.path.join(dataset_path, "results*.csv"))

            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    mean_records = df[df['offset'] == 'mean']

                    for _, record in mean_records.iterrows():
                        result_record = record.to_dict()
                        result_record['model'] = model_dir
                        result_record['datasets'] = dataset_dir
                        result_record['scheduler'] = os.path.splitext(os.path.basename(csv_file))[0].split('_')[-1]
                        all_results.append(result_record)

                except Exception as e:
                    print(f"Error processing file {csv_file}: {str(e)}")

    result_df = pd.DataFrame(all_results)
    grouped = result_df.groupby(['datasets', 'scheduler'])

    for (dataset_name, scheduler_name), group_df in grouped:
        sorted_df = group_df.sort_values('SLAV', ascending=True)
        output_file = os.path.join(base_dir, f"results_{dataset_name}_{scheduler_name}.csv")
        sorted_df.to_csv(output_file, index=False)



if __name__ == "__main__":
    process_decision_files()
