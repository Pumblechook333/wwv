import numpy as np
import pandas as pd
import dask.dataframe as dd
import datetime as dt
import h5py

def all_data(file_path, freq_band):
    f_min = 7; f_max = 10
    # Load the dataset into a Dask DataFrame
    ddf = dd.read_hdf(file_path, '/Data/Table Layout',\
                      chunksize=1000000, mode='r')
    # Filter the dataset where 'tfreq' is between f_min and f_max
    ddf['tfreq'] = ddf['tfreq'] / 1e6  # Divide 'tfreq' by 1e6 to convert to MHz
    ddf['tfreq'] = ddf['tfreq'].round(1)
    ddf_freq = ddf[(ddf['tfreq'] >= f_min) & (ddf['tfreq'] <= f_max)]
    # Select only the required columns: 'year', 'month', 'call_sign_tx'
    ddf_data = ddf_freq#[['year', 'month', 'tfreq']]
    # Compute the result to bring it into memory (Dask is lazy, so computation happens here)
    df = ddf_data.compute()
    df = df.dropna()
    # Create 'datetime' column by combining 'year', 'month', 'day', 'hour', 'min', 'sec'
    df['ut'] = pd.to_datetime(df['year'] + df['month'] + df['day'] + ' ' +
                                    df['hour'] + df['min'] + df['sec'],
                                    format='%Y%m%d %H%M%S')
    # delete columns
    df = df.drop(['year', 'month', 'day', 'hour', 'min', 'sec'], axis=1)
    # rename columns
    df = df.rename(columns={
                'call_sign_tx':'tx_call_sign', 'txlat':'tx_lat', 'txlon':'tx_lon',
                'call_sign_rx':'rx_call_sign', 'rxlat':'rx_lat', 'rxlon':'rx_lon',
                'pthlen':'dist', 'tfreq':'freq',
                'latcen':'mid_lat', 'loncen':'mid_lon'})
    # define column data types
    df = df.astype({'tx_call_sign':'string',  'rx_call_sign':'string',\
                    'tx_lat' :'float32', 'tx_lon' :'float32',\
                    'rx_lat' :'float32', 'rx_lon' :'float32',\
                    'mid_lat':'float32', 'mid_lon':'float32',\
                    'freq'  :'float32',  'dist'   :'float32',\
                    'sn':'int', 'smode':'str', 'ssrc':'str'})
    # round numbers
    df['tx_lat'] = df['tx_lat'].round(2)
    df['tx_lon'] = df['tx_lon'].round(2)
    df['rx_lat'] = df['rx_lat'].round(2)
    df['rx_lon'] = df['rx_lon'].round(2)
    df['mid_lat'] = df['mid_lat'].round(2)
    df['mid_lon'] = df['mid_lon'].round(2)
    df['dist'] = df['dist'].round(1)
    # reset row index of the file
    df = df.reset_index(drop=True)
    # delete or retain extra columns
    df = df.drop(['recno', 'kindat', 'kinst', 'ut1_unix', 'ut2_unix', 'smode'], axis=1)
    return df 

def main(file_path):
    # Load the dataset into a Dask DataFrame
    # ddf = dd.read_hdf(file_path, '/tecs', start=[0, 0], stop=[9, 9])

    with h5py.File(file_path, 'r') as f:
            data = f['tecs'][1:100]  # Read data as a NumPy array
            df = pd.DataFrame(data)

    return df

if __name__ == '__main__':
    datafile = "E:/Sabastian/Perry_Lab/dTEC_data/tid_182_2021_w121_n01_e15.h5"
    out = main(datafile)
    print(out)
