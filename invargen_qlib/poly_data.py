from typing import List, Union, Optional, Tuple
from enum import IntEnum
import numpy as np
import pandas as pd
import torch


class FeatureType(IntEnum):
    A = 0
    B = 1
    C = 2


class PolyData:

    def __init__(self,
                 device: torch.device = torch.device('cuda:0')) -> None:

        self.device = device
        self._features = list(FeatureType)
        self.data, self._poly_ids = self._get_data()


    def _load_exprs(self) -> pd.DataFrame:
        # This evaluates an expression on the data and returns the dataframe
        # It might throw on illegal expressions like "Ref(constant, dtime)"
        return pd.read_csv('csv_data/' + 'bqf_list_fund_disc_m3_m10000' + '.csv') 

    def _get_data(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        df = self._load_exprs()
        df = df.stack().unstack(level=0)
        poly_ids = df.columns
        values = df.values
        values = values.reshape((-1, values.shape[0], values.shape[-1])) # type: ignore

        return torch.tensor(values, dtype=torch.float, device=self.device), poly_ids

    @property
    def n_features(self) -> int:
        return len(self._features)

    @property
    def n_polys(self) -> int:
        return self.data.shape[-1]
    
    def make_dataframe(
        self,
        data: Union[torch.Tensor, List[torch.Tensor]],
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
            Parameters:
            - `data`: a tensor of size `(n_days, n_stocks[, n_columns])`, or
            a list of tensors of size `(n_days, n_stocks)`
            - `columns`: an optional list of column names
            """
        if isinstance(data, list):
            data = torch.stack(data, dim=2)
        if len(data.shape) == 2:
            data = data.unsqueeze(2)
        if columns is None:
            columns = [str(i) for i in range(data.shape[2])]
        n_days, n_stocks, n_columns = data.shape
        if self.n_days != n_days:
            raise ValueError(f"number of days in the provided tensor ({n_days}) doesn't "
                             f"match that of the current StockData ({self.n_days})")
        if self.n_stocks != n_stocks:
            raise ValueError(f"number of stocks in the provided tensor ({n_stocks}) doesn't "
                             f"match that of the current StockData ({self.n_stocks})")
        if len(columns) != n_columns:
            raise ValueError(f"size of columns ({len(columns)}) doesn't match with "
                             f"tensor feature count ({data.shape[2]})")
        date_index = self._dates[self.max_backtrack_days:-self.max_future_days]
        index = pd.MultiIndex.from_product([date_index, self._stock_ids])
        data = data.reshape(-1, n_columns)
        return pd.DataFrame(data.detach().cpu().numpy(), index=index, columns=columns)
