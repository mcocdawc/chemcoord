import pandas as pd
import numpy as np
from chemcoord._generic_classes._pandas_wrapper import _pandas_wrapper

df_data = pd.DataFrame(np.random.rand(3, 2), columns=['A', 'B'])
data = _pandas_wrapper(df_data)


def assert_loc():
    assert df_data.loc[:, :] == data.loc[:, :]
