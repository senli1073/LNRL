from .base import DatasetBase
import os 
from typing import Optional,Tuple
import pandas as pd
import numpy as np
from utils import logger,cal_snr
from ._factory import register_dataset



class SOS(DatasetBase):
    """Waveform from sos"""
    
    _name = "sos"
    _part_range = None
    _channels = ["z"]
    _sampling_rate = 500
    
    def __init__(
        self,
        seed:int,
        mode:str,
        data_dir:str,
        shuffle:bool=True,
        data_split:bool=False,
        train_size:float=0.8,
        val_size:float=0.1,
        **kwargs
        ):
        
        super().__init__(
            seed=seed,
            mode=mode,
            data_dir=data_dir,
            shuffle=shuffle,
            data_split=data_split,
            train_size=train_size,
            val_size=val_size,
        )
        
    
    def _load_meta_data(self,filename="_all_label.csv")->pd.DataFrame:
        meta_df = pd.read_csv(
            os.path.join(self._data_dir, filename),
            low_memory=False,
            dtype={"fname": str, "itp": int, "its": int}
        )
        
        for k in meta_df.columns:
            if meta_df[k].dtype in [np.dtype("float"),np.dtype("int")]:
                meta_df[k] = meta_df[k].fillna(0)
            elif meta_df[k].dtype in [object, np.object_, "object", "O"]:
                meta_df[k] = meta_df[k].str.replace(" ", "")
                meta_df[k] = meta_df[k].fillna("")

        if self._shuffle:
            meta_df = meta_df.sample(frac=1, replace=False, random_state=self._seed)

        meta_df.reset_index(drop=True, inplace=True)

        if self._data_split:
            irange = {}
            irange["train"] = [0, int(self._train_size * meta_df.shape[0])]
            irange["val"] = [
                irange["train"][1],
                irange["train"][1] + int(self._val_size * meta_df.shape[0]),
            ]
            irange["test"] = [irange["val"][1], meta_df.shape[0]]

            r = irange[self._mode]
            meta_df = meta_df.iloc[r[0] : r[1], :]
            logger.info(f"Data Split: {self._mode}: {r[0]}-{r[1]}")
        
        return meta_df
    
    
    def _load_event_data(self,idx:int) -> Tuple[dict,dict]:
        """Load event data

        Args:
            idx (int): Index of target row.

        Returns:
            dict: Data of event.
        """  
        target_event = self._meta_data.iloc[idx]
        
        fname = target_event["fname"]
        ppk = target_event["itp"]
        spk = target_event["its"]

        fpath = os.path.join(self._data_dir,fname)

        npz = np.load(fpath)

        data = npz["data"].astype(np.float32)
        
        data = np.stack(data, axis=1)

        snr = np.array([cal_snr(data=data,pat=ppk) if ppk > 0 else 0.])
        event = {
            "data": data,
            "ppks": [ppk] if ppk > 0 else [],
            "spks": [spk] if spk > 0 else [],
            "snr": snr
        }
        
        return event,target_event.to_dict()
    
@register_dataset
def sos(**kwargs):
    dataset = SOS(**kwargs)
    return dataset