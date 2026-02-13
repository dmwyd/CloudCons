from dataclasses import dataclass
from typing import Any
from pathlib import Path
from typing import List

import hydra
import optuna
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from optuna.storages import JournalStorage, JournalFileStorage
import torch
from lightning.pytorch.callbacks import LearningRateMonitor,EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau

from gluonts.time_feature import get_seasonality
from gluonts.model.predictor import Predictor
from gluonts.model import evaluate_model
import os
from torch.utils.tensorboard import SummaryWriter

import ray
import os
import shutil
from dotenv import load_dotenv
load_dotenv()
# Check if GPUs are available
# Set the memory growth for each visible GPU
physical_gpus = 1
AGENTS_PER_GPU = 3  # Replace with your total number of tasks
TOTAL_AGENTS = AGENTS_PER_GPU * physical_gpus
GPU_SHARE_PER_TASK = 0.3  # Replace with the number of GPUs you want to use



original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load


@dataclass
class Experiment:
    cfg : DictConfig
    def __post_init__(self):
        self.dataset = instantiate(self.cfg.dataset)
        
    def get_params(self, trial) -> dict[str, Any]:
        def get_hparams(trial, **kwargs) -> dict:

            hparams = {
                "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-5, log=True),
            }
            hparams["num_batches_per_epoch"] = 100
            hparams["context_length"] = self.cfg.context_length

            for param, info in kwargs.items():
                hparam_type, suggest_args = info[0], info[1:]
                if hparam_type == "categorical":
                    hp = trial.suggest_categorical(param, *suggest_args)
                elif hparam_type == "discrete_uniform":
                    hp = trial.suggest_discrete_uniform(param, *suggest_args)
                elif hparam_type == "float":
                    low, high = suggest_args[0]
                    log = suggest_args[1]["log"]
                    hp = trial.suggest_float(param, low, high, log=log)
                elif hparam_type == "int":
                    hp = trial.suggest_int(param, *suggest_args)
                elif hparam_type == "loguniform":
                    hp = trial.suggest_loguniform(param, *suggest_args)
                elif hparam_type == "uniform":
                    hp = trial.suggest_uniform(param, *suggest_args)
                else:
                    raise ValueError(f"hyperparameter type '{hparam_type}' is not recognised")
                hparams[param] = hp
            return hparams
        return get_hparams(trial, **instantiate(self.cfg.model.optuna, _convert_="all"))
    
    def get_model_params(self, selected_hparams) -> dict:
        model_config = self.cfg.model.estimator
        # Check the model type and set parameters accordingly
        if model_config._target_.endswith("DeepAREstimator"):
            model_params = {
                'freq': self.dataset.freq,
                'prediction_length': self.dataset.prediction_length,
            }
        elif model_config._target_.endswith("PatchTSTEstimator"):
            model_params = {
                # 'freq': self.dataset.freq,
                'prediction_length': self.dataset.prediction_length,
            }
        elif model_config._target_.endswith("DLinearEstimator"):
            model_params = {
                'prediction_length': self.dataset.prediction_length,
            }
        elif model_config._target_.endswith("TemporalFusionTransformerEstimator"):

            model_params = {
                'freq': self.dataset.freq,
                'prediction_length': self.dataset.prediction_length,
            }
        elif model_config._target_.endswith("TiDEEstimator"):
            model_params = {
                'freq': self.dataset.freq,
                'prediction_length': self.dataset.prediction_length,
            }
        elif model_config._target_.endswith("NBEATSEstimator"):
            model_params = {
                'freq': self.dataset.freq,
                'prediction_length': self.dataset.prediction_length,
            }
        elif model_config._target_.endswith("NBeatsEstimator"):
            model_params = {
                'freq': self.dataset.freq,
                'prediction_length': self.dataset.prediction_length,
                'eval_batch_size': self.cfg.eval_batch_size,  
            }
        elif model_config._target_.endswith("ITransformerEstimator"):
            model_params = {
                'freq': self.dataset.freq,
                'prediction_length': self.dataset.prediction_length,
            }
            if self.dataset.target_dim > 1 and not self.cfg.dataset.to_univariate:
                model_params["univariate"] = False
        elif model_config._target_.endswith("CrossformerEstimator"):
            model_params = {
                'prediction_length': self.dataset.prediction_length,
                'data_dim': self.dataset.target_dim,
            }
            if self.dataset.target_dim > 1 and not self.cfg.dataset.to_univariate:
                model_params["univariate"] = False
        else:
            raise ValueError(f"Model type {model_config._target_} not supported")      
        return model_params 
    
    def get_callbacks(self,trial) -> dict:
        callbacks = {
            "early_stopping": EarlyStopping(
                monitor="val_loss",
                min_delta=0.0,
                patience=5,      
                mode="min",
                strict=False,
                verbose=True
            ),
            "learning_rate_monitor": LearningRateMonitor(
                logging_interval="epoch"
            )
        }
        return callbacks

    def __call__(self, trial: optuna.Trial) -> float:
        selected_hparams = self.get_params(trial)
        model_params = self.get_model_params(selected_hparams)

        logs_dir = os.path.join(os.getenv("LIGHTNING_LOGS"),self.cfg.model.name,self.cfg.dataset.granularity,self.dataset.dataset_name,str(self.cfg.dataset.prediction_length))
        
        callbacks = self.get_callbacks(trial)
        # callbacks = {** {callback_name:hydra.utils.instantiate(callback_cfg) for callback_name,callback_cfg in self.cfg.callbacks.items()},
                # **{"pruning": LightningPytorchPruningCallback(trial, monitor="val_loss")}}

        estimator = instantiate(self.cfg.model.estimator, _convert_="all", _partial_=True)(
            **model_params,
            trainer_kwargs={
                "callbacks": list(callbacks.values()),
                "max_epochs" : 100,
                "devices" : 1,
                "default_root_dir": logs_dir
            },
            **selected_hparams,
        )

        try:
            estimator.train(
                self.dataset.training_dataset,
                validation_data=self.dataset.validation_dataset
                )
        except Exception as e:
            if isinstance(e, optuna.TrialPruned):
                raise  # re-raise the TrialPruned exception
            else:
                print(f"Error: {e}")
                # return NaN if the model fails to train except for pruning exception
                return float("nan")


        early_stopping = callbacks["early_stopping"]
        stopped_epoch = early_stopping.stopped_epoch or 100
        trial.set_user_attr("epochs", stopped_epoch - early_stopping.wait_count)
        return early_stopping.best_score
    
    def evaluate(self, trials: List[optuna.trial.FrozenTrial], output_dir):
        best_result = None
        for trial in trials:
            best_hparams = self.get_params(trial)
            model_params = self.get_model_params(best_hparams)
            
            callbacks = self.get_callbacks(trial)
            logs_dir = os.path.join(os.getenv("LIGHTNING_LOGS"),self.cfg.model.name,self.cfg.dataset.granularity,self.dataset.dataset_name,str(self.cfg.dataset.prediction_length))
            estimator = instantiate(self.cfg.model.estimator, _convert_="all", _partial_=True)(
                trainer_kwargs={
                    "callbacks": list(callbacks.values()),
                    "max_epochs" : 100,
                    "devices" : 1,
                    "default_root_dir": logs_dir
                },
                **model_params,
                **best_hparams,
            )
            for i in range(self.cfg.num_eval_repeats):
                predictor = estimator.train(self.dataset.training_dataset,validation_data=self.dataset.validation_dataset) # train on full data with best parameters
                season_length = get_seasonality(self.dataset.freq)
                metrics = instantiate(self.cfg.metrics, _convert_="all")
                res = evaluate_model(
                    predictor,
                    test_data=self.dataset.test_data,
                    metrics=metrics,
                    axis=None,
                    mask_invalid_label=True,
                    allow_nan_forecast=False,
                    seasonality=season_length,
                )
                if best_result is None or best_result["MASE[0.5]"][0] > res["MASE[0.5]"][0]:
                    best_result = res
                    print("New best result found")
        
        writer = SummaryWriter(log_dir=output_dir)
        for name, metric in best_result.to_dict("records")[0].items():
            writer.add_scalar(f"eval_metrics/{name}", metric)
        writer.close()

        csv_path = os.path.join(output_dir, "metrics.csv")
        best_result.to_csv(csv_path, index=False)
        print(f"Best result saved to {csv_path}")
    
    def get_predictor(self, trials: List[optuna.trial.FrozenTrial]):
        best_result = None
        best_predictor = None
        save_dir = os.path.join(os.getenv("DL_MODELS"),self.cfg.model.name,self.cfg.dataset.granularity,self.dataset.dataset_name,str(self.cfg.dataset.prediction_length))

        # Make sure the directory exists or create it:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # if .pt files exists inside the directory just load the predictor from file and return 
        if any([f.endswith(".pt") for f in os.listdir(save_dir)]):
            print("Predictor already exists.")
            predictor_deserialized = Predictor.deserialize(Path(save_dir))
            return predictor_deserialized
        for trial in trials:
            best_hparams = self.get_params(trial)
            model_params = self.get_model_params(best_hparams)

            callbacks = self.get_callbacks(trial)
            logs_dir = os.path.join(os.getenv("LIGHTNING_LOGS"),self.cfg.model.name,self.cfg.dataset.granularity,self.dataset.dataset_name,str(self.cfg.dataset.prediction_length))
            estimator = instantiate(self.cfg.model.estimator, _convert_="all", _partial_=True)(
                trainer_kwargs={
                    "callbacks": list(callbacks.values()),
                    "max_epochs" : 100,
                    "devices" : 1,
                    "default_root_dir": logs_dir
                },
                **model_params,
                **best_hparams,
            )
            predictor = estimator.train(self.dataset.training_dataset,validation_data=self.dataset.validation_dataset) # train on full data with best parameters
            
            season_length = get_seasonality(self.dataset.freq)
            metrics = instantiate(self.cfg.metrics, _convert_="all")
            res = evaluate_model(
                predictor,
                test_data=self.dataset.test_data,
                metrics=metrics,
                axis=None,
                mask_invalid_label=True,
                allow_nan_forecast=False,
                seasonality=season_length,
            )
            if best_result is None or best_result["MASE[0.5]"][0] > res["MASE[0.5]"][0]:
                best_result = res
                best_predictor = predictor
                print("New best result found")
        
        best_predictor.serialize(Path(save_dir))

        return best_predictor


def get_topk_trials(study, k=1):
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    completed_trials.sort(key=lambda x: x.value if x.value is not None else float('inf'))
    return completed_trials[:k] 


@ray.remote(num_gpus=GPU_SHARE_PER_TASK)
def run_with_ray(cfg, max_trials,storage,study_name):
        # Apply PyTorch fix in Ray worker process
        import torch
        original_torch_load = torch.load
        def patched_torch_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        torch.load = patched_torch_load
        
        experiment = Experiment(cfg)
        study = optuna.study.create_study(
            storage=storage,
            study_name=study_name,
            load_if_exists=True,
            direction="minimize",
        )
   
        states = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING)
        trials = study.get_trials(deepcopy=False, states=states)

        if len(trials) >= max_trials:
            return "Done"
    
        study.optimize(
            experiment,
            n_trials=cfg.n_trials,
            callbacks=[MaxTrialsCallback(max_trials, states=states)],
            gc_after_trial=True,
        )
        return "Done"

@hydra.main(version_base="1.3", config_path="conf_deep", config_name="default")
def main(cfg: DictConfig):
    ray.init()

    output_dir = HydraConfig.get().runtime.output_dir
    experiment = Experiment(cfg)
    study_name = f"{cfg.model.name}_{cfg.dataset.granularity}_{cfg.dataset.dataset_name}_{cfg.dataset.prediction_length}"
    max_trials = cfg.max_trials
    test = cfg.test
    infer = cfg.infer
    # TODO: 
    storage = JournalStorage(JournalFileStorage(f"outputs/deepmodel_optuna/hparam-tuning-journal-{cfg.model.name}.log")) 
    study = optuna.study.create_study(
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
        direction="minimize",
    )

    if test:
        top3_trials = get_topk_trials(study)
        experiment.evaluate(top3_trials,output_dir)
        # Clear checkpoints after evaluation
        lightning_logs = os.path.join(os.getenv("LIGHTNING_LOGS"),cfg.model.name,cfg.dataset.granularity,cfg.dataset.dataset_name,str(cfg.dataset.prediction_length))
        if lightning_logs:
            if os.path.isfile(lightning_logs):
                os.remove(lightning_logs)
                print("Lightning Logs File removed successfully.")
            elif os.path.isdir(lightning_logs):
                shutil.rmtree(lightning_logs)
                print("Lightning Logs Directory removed successfully.")
            else:
                print("The path does not exist.")
        else:
            print("Environment variable 'LIGHTNING_LOGS' is not set.")
    
    elif infer:
        top3_trials = get_topk_trials(study)
        experiment.get_predictor(top3_trials)
        # Clear checkpoints after evaluation
        lightning_logs = os.path.join(os.getenv("LIGHTNING_LOGS"),cfg.model.name,cfg.dataset.granularity,cfg.dataset.dataset_name,str(cfg.dataset.prediction_length))
        if lightning_logs:
            if os.path.isfile(lightning_logs):
                os.remove(lightning_logs)
                print("Lightning Logs File removed successfully.")
            elif os.path.isdir(lightning_logs):
                shutil.rmtree(lightning_logs)
                print("Lightning Logs Directory removed successfully.")
            else:
                print("The path does not exist.")
        else:
            print("Environment variable 'LIGHTNING_LOGS' is not set.")

    else:
        print("Running with Ray")    
        futures = ray.get([run_with_ray.remote(cfg, max_trials,storage,study_name) for _ in range(TOTAL_AGENTS)])
        
        # states = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING)
        # trials = study.get_trials(deepcopy=False, states=states)

        # if len(trials) >= max_trials:
        #     exit()

        # study.optimize(
        #     experiment,
        #     n_trials=cfg.n_trials,
        #     callbacks=[MaxTrialsCallback(max_trials, states=states)],
        #     gc_after_trial=True,
        # )

if __name__ == "__main__":
    main()
