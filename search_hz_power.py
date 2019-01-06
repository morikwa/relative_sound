# ===== module ===== #
from librosa import load, stft
import numpy as np
import glob, os, argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ===== config ===== #
N_FFT = 512
HOP_LEN = 256
SR = 16000

# ===== utils ===== #
def search_Hz_power(filename):
    y, _ = load(path=filename, sr=SR)
    spec = np.abs(
        stft(y=y, n_fft=N_FFT, hop_length=HOP_LEN)
    )
    spec_mean = spec.mean(axis=1).flatten()
 
    return spec_mean[:256].reshape(32,8).mean(axis=1)[:,np.newaxis]

# ===== main ===== #
def main():
    # set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='../data/ok/LA',
                        help='Relative path to data directory')
    parser.add_argument('--result', type=str, default='../assets',
                        help='Relative path to saving directory')
    args = parser.parse_args()
    
    # get data paths
    wav_names = glob.glob( os.path.join(args.data, '*.wav') )

    # calculate Hz powers
    res = np.concatenate([search_Hz_power(filename=i) for i in tqdm( wav_names )], axis=1)
    res = np.flipud(res)
    
    # save results
    plt.figure(figsize=(9,9))
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    sns.heatmap(data=res, cbar=False)
    plt.xlabel('Wav file')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(args.result, 'hz-power-{0}.png'.format(os.path.basename(args.data))))
    plt.close()
    np.savez(os.path.join(args.result, 'hz-power-{0}.npz'.format(os.path.basename(args.data))),
             path=wav_names, hz_power=res)

if __name__ == '__main__':
    main()