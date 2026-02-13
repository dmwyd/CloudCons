from typing import NamedTuple
from pathlib import Path
import os
import numpy as np
import pandas as pd
import datasets
from typing import Union, Optional
from uni2ts.common.env import env
from enum import Enum
from gluonts.transform import Transformation
from functools import partial
from typing import Iterable, Iterator
from gluonts.dataset.split import TestData, split,TrainingDataset
from gluonts.dataset.common import ProcessDataEntry
from gluonts.itertools import Map
from gluonts.dataset import DataEntry
from toolz import compose


class MetaInfo(NamedTuple):
    dataset_name: str
    freq: str

    target_dim: int
    past_feat_dynamic_real_dim: int
    test_split_offset: Union[pd.Period, int]
    prediction_length: int

    feat_dynamic_real_dim: int = 0

class Granularity(str, Enum):
    MINUTE = '5T'
    HALF_HOUR = '30T'
    HOUR = '1H'

def get_dataset_dim(dataset_name: str):
    if dataset_name == 'azure2019':
        return 1,2
    elif dataset_name == 'borg2019_d':
        return 2,2
    elif dataset_name == 'borg2019_e':
        return 2,2
    elif dataset_name == 'huawei2025':
        return 2,1
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

class MultivariateToFirstdim(Transformation):
    def __init__(self, field,dim):
        self.field = field
        self.dim=dim

    def __call__(
        self, data_it: Iterable[DataEntry], is_train: bool = False
    ) -> Iterator:
        for data_entry in data_it:
            item_id = data_entry["item_id"]
            val_ls = list(data_entry[self.field])
            univariate_entry = data_entry.copy()
            univariate_entry[self.field] = val_ls[self.dim]
            univariate_entry["item_id"] = item_id + "_dim" + str(self.dim)
            yield univariate_entry

def itemize_start(data_entry: DataEntry) -> DataEntry:
    data_entry["start"] = data_entry["start"].item()
    return data_entry

class Dataset:
    _decision_data_path = (
        getattr(env, "DECISION_DATA_PATH", None)
        or os.getenv("DECISION_DATA_PATH")
        or getattr(env, "CUSTOM_DATA_PATH", None)
        or os.getenv("CUSTOM_DATA_PATH")
    )
    storage_path: Path = Path(_decision_data_path) if _decision_data_path is not None else Path("data/decision")
    def __init__(
        self,
        dataset_name: str,
        prediction_length: int,
        test_split_offset: Union[pd.Period, int]=288,
        freq: str='1H',
        selected_dim: int = 0,
        to_univariate: Optional[bool] = None,
        granularity: Optional[Granularity] = None,
    ):
        target_dim, past_feat_dynamic_real_dim = get_dataset_dim(dataset_name)
        self.meta_info = MetaInfo(dataset_name, freq, target_dim, past_feat_dynamic_real_dim, test_split_offset, prediction_length)
        self.dataset = datasets.load_from_disk(self.storage_path / Path(dataset_name )).with_format("numpy")
        self.prediction_length = prediction_length
        self.dataset_name = dataset_name
        self.freq = freq
        process = ProcessDataEntry(
            freq,
            one_dim_target=target_dim == 1,
        )
        self.gluonts_dataset = Map(compose(process, itemize_start), self.dataset)
        if target_dim > 1:
            self.gluonts_dataset = MultivariateToFirstdim("target",selected_dim).apply(
                self.gluonts_dataset
            )

    @property
    def test_data(self) -> TestData:
        _, test_template = split(
            self.gluonts_dataset, offset=self.meta_info.test_split_offset
        )
        test_data = test_template.generate_instances(
            prediction_length=self.meta_info.prediction_length,
            windows=1,
            distance=self.meta_info.prediction_length,
        )
        return test_data

    @property
    def training_dataset(self) -> TrainingDataset:
        training_dataset, _ = split(
            self.gluonts_dataset, offset=self.meta_info.test_split_offset-self.prediction_length
        )
        return training_dataset

    @property
    def validation_dataset(self) -> TrainingDataset:
        validation_dataset, _ = split(
            self.gluonts_dataset, offset=self.meta_info.test_split_offset
        )
        return validation_dataset

    
