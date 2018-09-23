# -*- coding: utf-8 -*-

# ===== module ===== #
from librosa import load,stft,istft,display,output
import glob,os,argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# ===== config ===== #
N_FFT = 512
HOP_LEN = int(N_FFT/2)
LOWCUT_HZ = 10
HIGHCUT_HZ = 30
# ===== utils ===== #
def separate_wav(filename,result):
    y,sr = load(filename)
    spec = stft(y=y, n_fft=N_FFT, hop_length=HOP_LEN)
    spec_low = spec.copy()
    spec_high = spec.copy()

    spec_low[LOWCUT_HZ:] = 0
    spec_high[0:HIGHCUT_HZ] = 0
    spec[:LOWCUT_HZ]=0;spec[HIGHCUT_HZ:]=0
    low = istft(spec_low,hop_length=HOP_LEN,win_length=N_FFT)
    middle = istft(spec,hop_length=HOP_LEN,win_length=N_FFT)
    high = istft(spec_high,hop_length=HOP_LEN,win_length=N_FFT)
    output.write_wav(os.path.join(result,'low',filename.rsplit('/',1)[1]),low,sr=sr)
    output.write_wav(os.path.join(result,'middle',filename.rsplit('/',1)[1]),middle,sr=sr)
    output.write_wav(os.path.join(result,'high',filename.rsplit('/',1)[1]),high,sr=sr)

    # if want to watch the spectrogram, please remove '''
    '''
    fig,axes = plt.subplots(ncols=3,nrows=1,figsize=(12,9))
    plt.subplot(3,1,1)
    display.specshow(spec_high,hop_length=HOP_LEN,x_axis='time',y_axis='hz')
    plt.subplot(3,1,2)
    display.specshow(spec,hop_length=HOP_LEN,x_axis='time',y_axis='hz')
    plt.subplot(3,1,3)
    display.specshow(spec_low,hop_length=HOP_LEN,x_axis='time',y_axis='hz')
    plt.show()
    '''

# ===== main ===== #
def main():
    # set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data','-d',type=str,default='../data/',
                        help='Relative path to data directory')
    parser.add_argument('--result','-r',type=str,default='../data/separate/',
                        help='Relative path to save directory')
    args = parser.parse_args()
    # if not exist directory make file
    if not os.path.exists(args.result):
        os.mkdir(args.result)
        os.mkdir(os.path.join(args.result,'low'))
        os.mkdir(os.path.join(args.result,'middle'))
        os.mkdir(os.path.join(args.result,'high'))

    # get data paths
    d = glob.glob(os.path.join(args.data,'*.wav'))

    # make 3 different Hz wav file
    print('Start separate wav low middle high...')
    for i in tqdm(d):
        separate_wav(i,args.result)

if __name__ == '__main__':
    main()
