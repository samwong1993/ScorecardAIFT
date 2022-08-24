import pandas as pd
import numpy as np
from tqdm import tqdm

data = pd.read_csv('./data1.csv')
sell = pd.read_csv('./newoldsellerID.csv')
data.sort_values(by='asin',inplace = True)
sell.sort_values(by='id',inplace = True)
data.reset_index(inplace=True,drop=True)
sell.reset_index(inplace=True,drop=True)

data['label'] = 0
label = []
data_asin_list = list(data['asin'])
sell_asin_list = list(sell['id'])
pbar = tqdm(total=len(data))
lis = sell.loc[:,'new_ava']
i = 0
j = 0
while i < len(data):
    pbar.update(1)
    asin = data_asin_list[i]
    if asin in sell_asin_list:
        idx = sell_asin_list.index(asin)
        if pd.isna(lis[idx]):
            label.append(1)
            # data.loc[i, 'label'] = 1
        else:
            label.append(0)
            # print(f'Find->i/j:{i}/{len(data)}')
    else:
        # data.loc[i, 'label'] = -1
        label.append(-1)
        # print(f'Not Find->i/j:{i}/{len(data)}')
    i += 1
pbar.close()
data.loc[:len(label)-1,'label'] = label
data.loc[:len(label),:].to_csv('./seller.csv')


