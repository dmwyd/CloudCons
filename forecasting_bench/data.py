import numpy as np
import pandas as pd
from gluonts.transform import Transformation
from gluonts.dataset import DataEntry
from typing import Iterable, Iterator
from typing import NamedTuple
from pathlib import Path
from uni2ts.common.env import env
from typing import Union
from toolz import compose
from gluonts.dataset import DataEntry
from gluonts.dataset.common import ProcessDataEntry
from gluonts.dataset.split import TestData, TrainingDataset, split
from gluonts.itertools import Map
from enum import Enum

import datasets
class MetaInfo(NamedTuple):
    freq: str
    target_dim: int
    past_feat_dynamic_real_dim: int

    test_split_offset: Union[pd.Period, int]
    prediction_length: int
    rolling_evaluations: int

class Granularity(str, Enum):
    MINUTE = '5T'
    HALF_HOUR = '30T'
    HOUR = '1H'

class DatasetInfo:
    def __init__(self):
        meta_config_5T = {
            'azure2019':{
                48 : MetaInfo(Granularity.MINUTE, 1, 2, 1800, 48, 9),
                96: MetaInfo(Granularity.MINUTE, 1, 2, 1800, 96, 4),
                192:MetaInfo(Granularity.MINUTE, 1, 2, 1800, 192, 2),
                336: MetaInfo(Granularity.MINUTE, 1, 2, 1800, 336, 1)
            },
            'borg2019_d_trace':{
                48 : MetaInfo(Granularity.MINUTE, 2, 2, 1800, 48, 9),
                96: MetaInfo(Granularity.MINUTE, 2, 2, 1800, 96, 4),
                192:MetaInfo(Granularity.MINUTE, 2, 2, 1800, 192, 2),
                336: MetaInfo(Granularity.MINUTE, 2, 2, 1800, 336, 1)
            },
            'borg2019_e_trace':{
                48 : MetaInfo(Granularity.MINUTE, 2, 2, 1800, 48, 9),
                96: MetaInfo(Granularity.MINUTE, 2, 2, 1800, 96, 4),
                192:MetaInfo(Granularity.MINUTE, 2, 2, 1800, 192, 2),
                336: MetaInfo(Granularity.MINUTE, 2, 2, 1800, 336, 1)
                },
            'huawei2025':{
                48 : MetaInfo(Granularity.MINUTE, 2, 1, 1800, 48, 9),
                96: MetaInfo(Granularity.MINUTE, 2, 1, 1800, 96, 4),
                192:MetaInfo(Granularity.MINUTE, 2, 1, 1800, 192, 2),
                336: MetaInfo(Granularity.MINUTE, 2, 1, 1800, 336, 1)
            }
        }
        meta_config_30T = {
            'azure2019':{
                12 : MetaInfo(Granularity.HALF_HOUR, 1, 2, 300, 12, 6),
                24: MetaInfo(Granularity.HALF_HOUR, 1, 2, 300, 24, 3),
                36: MetaInfo(Granularity.HALF_HOUR, 1, 2, 300, 36, 2),
                48:MetaInfo(Granularity.HALF_HOUR, 1, 2, 300, 48, 1),
            },
            'borg2019_d_trace':{
                12 : MetaInfo(Granularity.HALF_HOUR, 2, 2, 300, 12, 6),
                24: MetaInfo(Granularity.HALF_HOUR, 2, 2, 300, 24, 3),
                36: MetaInfo(Granularity.HALF_HOUR, 2, 2, 300, 36, 2),
                48: MetaInfo(Granularity.HALF_HOUR, 2, 2, 300, 48, 1),
            },
            'borg2019_e_trace':{
                12 : MetaInfo(Granularity.HALF_HOUR, 2, 2, 300, 12, 6),
                24: MetaInfo(Granularity.HALF_HOUR, 2, 2, 300, 24, 3),
                36: MetaInfo(Granularity.HALF_HOUR, 2, 2, 300, 36, 2),
                48: MetaInfo(Granularity.HALF_HOUR, 2, 2, 300, 48, 1),
            },
            'huawei2025':{
                12 : MetaInfo(Granularity.HALF_HOUR, 2, 1, 300, 12, 6),
                24: MetaInfo(Granularity.HALF_HOUR, 2, 1, 300, 24, 3),
                36: MetaInfo(Granularity.HALF_HOUR, 2, 1, 300, 36, 2),
                48: MetaInfo(Granularity.HALF_HOUR, 2, 1, 300, 48, 1),
            }
        }
        meta_config_60T = {
            'azure2019':{
                6 : MetaInfo(Granularity.HOUR, 1, 2, 150, 6, 6),
                8: MetaInfo(Granularity.HOUR, 1, 2, 150, 8, 4),
                10: MetaInfo(Granularity.HOUR, 1, 2, 150, 10, 3),
                12: MetaInfo(Granularity.HOUR, 1, 2, 150, 12, 3),
            },
            'borg2019_d_trace':{
                6 : MetaInfo(Granularity.HOUR, 2, 2, 150, 6, 6),
                8: MetaInfo(Granularity.HOUR, 2, 2, 150, 8, 4),
                10: MetaInfo(Granularity.HOUR, 2, 2, 150, 10, 3),
                12: MetaInfo(Granularity.HOUR, 2, 2, 150, 12, 3),
            },
            'borg2019_e_trace':{
                6 : MetaInfo(Granularity.HOUR, 2, 2, 150, 6, 6),
                8: MetaInfo(Granularity.HOUR, 2, 2, 150, 8, 4),
                10: MetaInfo(Granularity.HOUR, 2, 2, 150, 10, 3),
                12: MetaInfo(Granularity.HOUR, 2, 2, 150, 12, 3),
            },
            'huawei2025':{
                6 : MetaInfo(Granularity.HOUR, 2, 1, 150, 6, 6),
                8: MetaInfo(Granularity.HOUR, 2, 1, 150, 8, 4),
                10: MetaInfo(Granularity.HOUR, 2, 1, 150, 10, 3),
                12: MetaInfo(Granularity.HOUR, 2, 1, 150, 12, 3),
            }
        }
        self.meta_config = {
            Granularity.MINUTE: meta_config_5T,
            Granularity.HALF_HOUR: meta_config_30T,
            Granularity.HOUR: meta_config_60T
        }

class MultivariateToUnivariate(Transformation):
    def __init__(self, field):
        self.field = field

    def __call__(
        self, data_it: Iterable[DataEntry], is_train: bool = False
    ) -> Iterator:
        for data_entry in data_it:
            item_id = data_entry["item_id"]
            val_ls = list(data_entry[self.field])
            for id, val in enumerate(val_ls):
                univariate_entry = data_entry.copy()
                univariate_entry[self.field] = val
                univariate_entry["item_id"] = item_id + "_dim" + str(id)
                yield univariate_entry

def itemize_start(data_entry: DataEntry) -> DataEntry:
    data_entry["start"] = data_entry["start"].item()
    return data_entry


class Dataset:
    storage_path: Path = env.CUSTOM_DATA_PATH
    def __init__(
        self,
        dataset_name: str,
        granularity: Granularity,
        prediction_length: int,
        to_univariate: bool = False,
    ):
        self.prediction_length = prediction_length
        self.dataset_name = dataset_name
        self.granularity = granularity
        self.dataset_info = DatasetInfo()
        self.meta_info = self.dataset_info.meta_config[self.granularity][self.dataset_name][self.prediction_length]
        self.freq = self.meta_info.freq
        self.target_dim = self.meta_info.target_dim
        self.past_feat_dynamic_real_dim = self.meta_info.past_feat_dynamic_real_dim
        self.test_split_offset = self.meta_info.test_split_offset
        self.prediction_length = self.meta_info.prediction_length
        self.rolling_evaluations = self.meta_info.rolling_evaluations
        self.windows = self.meta_info.rolling_evaluations
        if isinstance(self.granularity, Granularity):
            self.hf_dataset = datasets.load_from_disk(self.storage_path /Path(self.granularity.value) / Path(self.dataset_name )).with_format(
                "numpy"
            )
        else:
            self.hf_dataset = datasets.load_from_disk(self.storage_path /Path(self.granularity) / Path(self.dataset_name )).with_format(
                "numpy"
            )
        process = ProcessDataEntry(
            self.freq,
            one_dim_target=self.target_dim == 1,
        )

        self.gluonts_dataset = Map(compose(process, itemize_start), self.hf_dataset)
        if to_univariate:
            self.gluonts_dataset = MultivariateToUnivariate("target").apply(
                self.gluonts_dataset
            )

    @property
    def training_dataset(self) -> TrainingDataset:
        training_dataset, _ = split(
            self.gluonts_dataset, offset=self.test_split_offset-self.prediction_length
        )
        return training_dataset

    @property
    def validation_dataset(self) -> TrainingDataset:
        validation_dataset, _ = split(
            self.gluonts_dataset, offset=self.test_split_offset
        )
        return validation_dataset
    
    @property
    def test_data(self) -> TestData:
        _, test_template = split(
            self.gluonts_dataset, offset=self.test_split_offset
        )
        test_data = test_template.generate_instances(
            prediction_length=self.prediction_length,
            windows=self.rolling_evaluations,
            distance=self.prediction_length,
        )
        return test_data

        