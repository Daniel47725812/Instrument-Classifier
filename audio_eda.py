import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc
from python_speech_features import logfbank

# 繪製時域信號
# 顯示每個音訊類別的波形
# 每行顯示 5 個子圖
def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, 
                         sharey=True, figsize=(20, 5))
    fig.suptitle('Time Series', size=16, y=1.02)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

# 繪製頻域信號 (FFT 結果)
# 每行顯示 5 個子圖
def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, 
                         sharey=True, figsize=(20, 5))
    fig.suptitle('Fourier Transforms', size=16, y=1.02)
    i = 0
    for x in range(2):
        for y in range(5):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

# 繪製 MFCC 特徵
# 每行顯示 5 個子圖
def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, 
                         sharey=True, figsize=(20, 5))
    fig.suptitle('Mel Cepstrum Coefficients', size=16, y=1.02)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],
                        cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

# 繪製濾波器組係數 (Filter Bank Coefficients)
# 每行顯示 5 個子圖
def plot_fbank(fbanks):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, 
                         sharey=True, figsize=(20, 5))
    fig.suptitle('Filter Bank Coefficients', size=16, y=1.02)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(fbanks.keys())[i])
            axes[x,y].imshow(list(fbanks.values())[i],
                        cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

# 計算 FFT (快速傅立葉變換)
# 將時域信號轉換為頻域信號
# 輸出為頻率和對應的幅值
def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y = abs(np.fft.rfft(y)/n)
    return (Y, freq)

# 包絡檢測 (Envelope Detection)
# 過濾低於指定閾值 (threshold) 的信號，用於去除背景噪音
# 使用移動平均計算信號幅值的趨勢
# 返回布林遮罩，用於保留有效信號
def envelope(y, rate, threshold):
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    mask = y_mean > threshold
    return mask

# 讀取音訊檔案資訊
# instruments.csv 包含音訊檔案名稱和對應的樂器類別
# 計算每個音訊檔案的時長，並分析類別分佈
df = pd.read_csv('instruments.csv')
df.set_index('fname', inplace=True)

# 計算每個音訊檔案的長度 (秒)
for f in df.index:
    rate, signal = wavfile.read('wavfiles/'+f)
    df.at[f, 'length'] = signal.shape[0] / rate

# 獲取所有樂器類別
classes = list(np.unique(df.label))

# 計算每個類別的平均音訊長度
class_dist = df.groupby(['label'])['length'].mean()

# 繪製類別分佈圓餅圖
fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
       shadow=False, startangle=90)
ax.axis('equal')
plt.show()

df.reset_index(inplace=True)

# 初始化存放特徵的字典
signals = {}  # 時域信號
fft = {}      # 頻域信號
mfccs = {}    # MFCC 特徵
fbank = {}    # 濾波器組係數

# 對每個類別進行音訊處理
for c in classes:
    # 讀取該類別的第一個音訊檔案
    wav_file = df[df.label == c].iloc[0, 0]
    rate, signal = wavfile.read('wavfiles/' + wav_file)

    # 應用包絡檢測，去除背景噪音
    mask = envelope(signal, rate, 20)
    signal = signal[mask]

    # 將清理後的音訊寫入 samples 資料夾
    wavfile.write('samples/' + c + '.wav', rate, signal)

    # 提取時域信號
    signals[c] = signal

    # 提取頻域信號 (FFT)
    fft[c] = calc_fft(signal, rate)

    # 計算 MFCC 特徵
    signal = signal.astype(float)
    mel = mfcc(signal[:rate], samplerate=rate,
               numcep=64, nfilt=64, nfft=1103).T
    mfccs[c] = mel

    # 計算濾波器組係數
    bank = logfbank(signal[:rate], rate,
                    nfilt=64, nfft=1103).T
    fbank[c] = bank

# 繪製時域信號
plot_signals(signals)
plt.show()

# 繪製頻域信號
plot_fft(fft)
plt.show()

# 繪製濾波器組係數
plot_fbank(fbank)
plt.show()

# 繪製 MFCC 特徵
plot_mfccs(mfccs)
plt.show()

# 如果 clean 資料夾為空，則清理所有音訊檔案並存入 clean 資料夾
if len(os.listdir('clean')) == 0:
    for f in tqdm(df.fname):
        rate, signal = wavfile.read('wavfiles/' + f)
        mask = envelope(signal, rate, 20)
        wavfile.write('clean/' + f, rate, signal[mask])
