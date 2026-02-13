DEEP_MODEL=$1

for ds in huawei2025 azure2019 borg2019_d_trace borg2019_e_trace; do
  echo "Running $DEEP_MODEL on " $ds
  for prediction_length in 48 96 192 336; do
    if [ "$ds" = "azure2019" ]; then
      to_univariate=False
    else
      to_univariate=True
    fi
    python -m DeepModel.hparam_global model=$DEEP_MODEL test=false dataset.dataset_name=$ds dataset.prediction_length=$prediction_length dataset.to_univariate=$to_univariate dataset.granularity=5T context_length=512;
  done;
done;

for ds in huawei2025 azure2019 borg2019_d_trace borg2019_e_trace; do
  echo "Running $DEEP_MODEL on " $ds
  for prediction_length in 12 24 36 48; do
    if [ "$ds" = "azure2019" ]; then
      to_univariate=False
    else
      to_univariate=True
    fi
    python -m DeepModel.hparam_global model=$DEEP_MODEL test=false dataset.dataset_name=$ds dataset.prediction_length=$prediction_length dataset.to_univariate=$to_univariate dataset.granularity=30T context_length=96;
  done;
done;

for ds in huawei2025 azure2019 borg2019_d_trace borg2019_e_trace; do
  echo "Running $DEEP_MODEL on " $ds
  for prediction_length in 6 8 10 12; do
    if [ "$ds" = "azure2019" ]; then
      to_univariate=False
    else
      to_univariate=True
    fi
    python -m DeepModel.hparam_global model=$DEEP_MODEL test=false dataset.dataset_name=$ds dataset.prediction_length=$prediction_length dataset.to_univariate=$to_univariate dataset.granularity=1H context_length=36;
  done;
done;

for ds in huawei2025 azure2019 borg2019_d_trace borg2019_e_trace; do
  echo "Running $DEEP_MODEL on " $ds
  for prediction_length in 48 96 192 336; do
    if [ "$ds" = "azure2019" ]; then
      to_univariate=False
    else
      to_univariate=True
    fi
    python -m DeepModel.hparam_global model=$DEEP_MODEL test=true dataset.dataset_name=$ds dataset.prediction_length=$prediction_length dataset.to_univariate=$to_univariate dataset.granularity=5T context_length=512;
  done;
done;

for ds in huawei2025 azure2019 borg2019_d_trace borg2019_e_trace; do
  echo "Running $DEEP_MODEL on " $ds
  for prediction_length in 12 24 36 48; do
    if [ "$ds" = "azure2019" ]; then
      to_univariate=False
    else
      to_univariate=True
    fi
    python -m DeepModel.hparam_global model=$DEEP_MODEL test=true dataset.dataset_name=$ds dataset.prediction_length=$prediction_length dataset.to_univariate=$to_univariate dataset.granularity=30T context_length=96;
  done;
done;

for ds in huawei2025 azure2019 borg2019_d_trace borg2019_e_trace; do
  echo "Running $DEEP_MODEL on " $ds
  for prediction_length in 6 8 10 12; do
    if [ "$ds" = "azure2019" ]; then
      to_univariate=False
    else
      to_univariate=True
    fi
    python -m DeepModel.hparam_global model=$DEEP_MODEL test=true dataset.dataset_name=$ds dataset.prediction_length=$prediction_length dataset.to_univariate=$to_univariate dataset.granularity=1H context_length=36;
  done;
done;


