DEEP_MODEL=$1

for ds in huawei2025 azure2019 borg2019_d borg2019_e; do
  echo "Running $DEEP_MODEL on " $ds
  for prediction_length in 24; do
    python -m DeepModel.hparam_global model=$DEEP_MODEL dataset._target_=predictor.data.Dataset  dataset.dataset_name=$ds dataset.prediction_length=$prediction_length dataset.to_univariate=false dataset.granularity=1H context_length=120;
  done;
done;
