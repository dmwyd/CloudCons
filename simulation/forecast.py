import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from predictor.data import MetaInfo
from gluonts.model import Forecast as GluontsForecast
from tqdm import tqdm
from typing import List, Optional, Union


class Forecast:
    def __init__(self,
                inputs: Optional[List[Dict]] = None,
                labels: Optional[List[Dict]] = None,
                forecasts: Optional[List[GluontsForecast]] = None,
                model_info: Optional[Dict] = None,
                metadata: Optional[Dict] = None):
        self.inputs=inputs or []
        self.labels = labels or []
        self.forecasts = forecasts or []
        self.model_info = model_info or {}
        self.metadata = metadata or {}
             
    def save(self, parent_dir: Union[str, Path]) -> None:
        path=parent_dir+"/"+self.metadata['dataset_name']+"/"+str(self.metadata['test_split_offset'])
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        meta_info = {
            'model_info': self.model_info,
            'metadata': self.metadata,
            'num_forecasts': len(self.forecasts)
        }
        
        with open(path / 'inputs.pkl', 'wb') as f:
            pickle.dump(self.inputs, f)
        with open(path / 'labels.pkl', 'wb') as f:
            pickle.dump(self.labels, f)
        with open(path / 'forecasts.pkl', 'wb') as f:
            pickle.dump(self.forecasts, f)
        with open(path / 'meta_info.json', 'w') as f:
            json.dump(meta_info, f, indent=2)
                
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'Forecast':
        path = Path(path)
        
        with open(path / 'meta_info.json', 'r') as f:
            meta_info = json.load(f)
        
        forecast_obj = cls(
            inputs=[],
            labels=[],
            forecasts=[],
            model_info=meta_info.get('model_info', {}),
            metadata=meta_info.get('metadata', {}),
        )
            
        if (path / 'inputs.pkl').exists():
            with open(path / 'inputs.pkl', 'rb') as f:
                forecast_obj.inputs = pickle.load(f)
        if (path / 'labels.pkl').exists():
            with open(path / 'labels.pkl', 'rb') as f:
                forecast_obj.labels = pickle.load(f)
        if (path / 'forecasts.pkl').exists():
            with open(path / 'forecasts.pkl', 'rb') as f:
                forecast_obj.forecasts = pickle.load(f)
        
        return forecast_obj
    

    @classmethod
    def predict(cls,
                test_data,
                metadata: MetaInfo ,
                model:str,
                predictor:Any,
                context_length: int = 120,) -> 'Forecast':
        predictor = predictor
        
        inputs=test_data.input
        forecasts = list(predictor.predict(tqdm(inputs,desc='predicting')))
        labels=list(test_data.label)
        inputs=list(test_data.input)
        
        model_info = {
            'model_name': model,
            'context_length': context_length,
        }
        
        _metadata={
            'dataset_name': metadata.dataset_name,
            'freq': metadata.freq,
            'prediction_length': metadata.prediction_length,
            'target_dim': metadata.target_dim,
            'feat_dynamic_real_dim': metadata.feat_dynamic_real_dim,
            'past_feat_dynamic_real_dim': metadata.past_feat_dynamic_real_dim,
            'test_split_offset': metadata.test_split_offset,
        }
        return cls(forecasts=forecasts, model_info=model_info, metadata=_metadata, labels=labels,inputs=inputs)
    
    

    def __len__(self) -> int:
        return len(self.forecasts)
    
    def __getitem__(self, index: int) -> GluontsForecast:
        return self.inputs[index],self.labels[index],self.forecasts[index]
    
