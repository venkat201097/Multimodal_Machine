import os
import sys
import subprocess
import pandas as pd
import pickle
import numpy as np
import random
import tensorflow as tf
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO

def get_free_gpu():
    id2gpu = {'0':'1','1':'2','2':'0'}
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]).decode()
    gpu_df = pd.read_csv(StringIO(gpu_stats),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    print('GPU usage:\n{}'.format(gpu_df))
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: float(x.rstrip(' [MiB]')))
    gpu_df['memory.used'] = gpu_df['memory.used'].map(lambda x: float(x.rstrip(' [MiB]')))
    gpu_df['memory.free_frac'] = gpu_df['memory.free']/(gpu_df['memory.free']+gpu_df['memory.used'])
    print(gpu_df)
    idx = gpu_df['memory.free_frac'].idxmax()
    print('Returning GPU {} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
    return id2gpu[str(idx)]

def set_gpu(free_gpu=None):
    free_gpu = free_gpu if free_gpu else get_free_gpu()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(free_gpu)
    if tf.__version__[0]=='2':
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.keras.backend.set_session(tf.Session(graph=tf.get_default_graph(), config=config))

def set_random_seed(random_seed=0):
    print('Seed: ',random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    tf.random.set_seed(random_seed)
    return random_seed

def get_data(path, data_list):
    data = []
    for i in data_list:
        data.append(pickle.load(open(path+i+'.pkl','rb')))
    return data

def bucket_data(img, aud, img2lab, buckets, bucket=0):
    img_bucket={}
    aud_bucket={}
    for i in buckets[bucket]:
        img_bucket[i] = img[i]
        aud_bucket[img2lab[i]] = aud[img2lab[i]]
    return img_bucket,aud_bucket

def createdf(img, aud, samples, mode):
    df = []
    for i in sorted(list(img)):
        for j in random.sample(list(img[i]),min(samples,len(img[i]))):
            df.append([j,img_lab2idx[i],0, 1])
            # df.append([j,lab2idx[i],0])

    for i in sorted(list(aud)):
        for j in random.sample(list(aud[i]),min(samples,len(aud[i]))):
            df.append([j,aud_lab2idx[i],1, 1])
            # df.append([j,lab2idx[i],0])

    df = np.array(df)
    np.random.shuffle(df)
    # df1 = np.array(df1)
    # np.random.shuffle(df1)
    return df

def createdf2(data, lab2id, samples, mode, mask):
    df = []
    for i in sorted(list(data)):
        np.random.shuffle(data[i])
        for j in data[i][:min(samples, len(data[i]))]:
            df.append([j, lab2id[i], mode, mask])
    return np.array(df)

def create_train_n(data,lab2id,samples,mode):
    df = []
    df_ = []
    for i in sorted(list(data)):
        np.random.shuffle(data[i])
        for j in data[i][:samples]:
            df.append([j,lab2id[i],mode])
        for j in data[i][samples:]:
            df_.append([j,lab2id[i],mode])
    np.random.shuffle(df)
    np.random.shuffle(df_)
    return df,df_

def createtest(data, classes, samples):
    df = []
    for i in classes:
        for j in data[i][:min(samples, len(data[i]))]:
            df.append(j)
    return np.array(df)

def createdata(img_train, aud_train, img_test, aud_test, img2lab, lab2img, img_lab2idx, aud_lab2idx, samples):
    df = []
    df_test_img = []
    df_test_aud = []
    for i in sorted(list(img_train)):
        # print(i)
        np.random.shuffle(img_train[i])
        for j in img_train[i][:samples]:
            df.append([j,img_lab2idx[i],0,1])
        for j in img_train[i][samples:]:
            df_test_img.append(j)
        for j in img_test[i]:
            df_test_img.append(j)

    for i in [img2lab[i] for i in sorted([lab2img[i] for i in list(aud_train)])]:
        # print(lab2img[i])
        np.random.shuffle(aud_train[i])
        for j in aud_train[i][:samples]:
            df.append([j,aud_lab2idx[i],1,1])
        for j in aud_train[i][samples:]:
            df_test_aud.append(j)
        for j in aud_test[i]:
            df_test_aud.append(j)

    df = np.array(df)
    # np.random.shuffle(df)
    df_test_img = np.array(df_test_img)
    df_test_aud = np.array(df_test_aud)
    return df,df_test_img,df_test_aud


def createdata2(img_train, aud_train, img_test, aud_test, img2lab, lab2img, img_lab2idx, aud_lab2idx, samples):
    df = []
    df_test_img = []
    df_test_aud = []

    for i in sorted(list(img_train)):
        i_ = img2lab[i]
        np.random.shuffle(img_train[i])
        np.random.shuffle(aud_train[i_])
        
        for j in img_train[i][samples:]:
            df_test_img.append(j)
        for j in img_test[i]:
            df_test_img.append(j)
        
        for j in aud_train[i_][samples:]:
            df_test_aud.append(j)
        for j in aud_test[i_]:
            df_test_aud.append(j)

    for sample in range(samples):
        for i in sorted(list(img_train)):
            i_ = img2lab[i]
            df.append([img_train[i][sample], img_lab2idx[i], 0, 0])
            df.append([aud_train[i_][sample], aud_lab2idx[i_], 1, 0])


    df = np.array(df)
    df_test_img = np.array(df_test_img)
    df_test_aud = np.array(df_test_aud)
    return df,df_test_img,df_test_aud