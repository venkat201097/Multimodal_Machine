import sys
import numpy as np
import pickle
import os
from utils import get_free_gpu
from model_triplet import JointNet
import tensorflow as tf
from utils import get_free_gpu, get_data, bucket_data, createdf2, createdf, createdata

def get_model(filepath):
    jointnet = JointNet(False,online = True)
    model = jointnet.model
    aud_transform = jointnet.audio_submodel
    img_transform = jointnet.image_submodel

    model.load_weights(filepath)
    return img_transform, aud_transform
    
def get_confusion(aud, img, classes, path, folder, num_conf=15, random_seed=None, ):
    if random_seed!=None:
        print('Np seed: ',random_seed)
        np.random.seed(random_seed)
    
    if not os.path.isdir(path+'Confusions'):
        os.system('mkdir '+path+'Confusions')

    # Get audio anchor confusions
    num_audios = len(aud)//len(classes)
    num_images = len(img)//len(classes)
    print(num_audios, num_images)
    NUM_CLASSES = len(classes)
    for i in range(num_conf):
        cmat = np.zeros((NUM_CLASSES, NUM_CLASSES))
        
        for x, ca in enumerate(classes):
            aud_ind = np.random.randint(num_audios)
            v1 = aud[x*num_audios + aud_ind]
            v1 = v1 / (np.linalg.norm(v1) + 1e-16)
            
            for y, ci in enumerate(classes):
                img_ind = np.random.randint(num_images)
                v2 = img[y*num_images + img_ind]                
                v2 = v2 / (np.linalg.norm(v2) + 1e-16)
                
                cmat[x][y] = np.dot(v1, v2)
        
        aud_folder = path+'Confusions/{}_audio_anchor/'.format(folder)
        if not os.path.isdir(aud_folder):
            os.system('mkdir '+aud_folder)
        with open(aud_folder+'confusion_mat_'+str(i)+'.pkl','wb') as fp:
            pickle.dump(cmat, fp)


    # Get image anchor confusions
    num_audios = len(aud)//len(classes)
    num_images = len(img)//len(classes)
    for i in range(num_conf):
        cmat = np.zeros((NUM_CLASSES, NUM_CLASSES))
        
        for x, ci in enumerate(classes):
            img_ind = np.random.randint(num_images)
            v1 = img[x*num_images + img_ind]
            v1 = v1 / (np.linalg.norm(v1) + 1e-16)
            
            for y, ca in enumerate(classes):
                aud_ind = np.random.randint(num_audios)
                v2 = aud[y*num_audios + aud_ind]
                v2 = v2 / (np.linalg.norm(v2) + 1e-16)
                
                cmat[x][y] = np.dot(v1, v2)

        img_folder = path+'Confusions/{}_image_anchor/'.format(folder)
        if not os.path.isdir(img_folder):
            os.system('mkdir '+img_folder)
        with open(img_folder+'confusion_mat_'+str(i)+'.pkl','wb') as fp:
            pickle.dump(cmat, fp)

    print('Confusions stored in: ', aud_folder,'\t',img_folder)
    return aud_folder, img_folder

def get_confusion_imgnet(speech_data, img_data_test, img_transform, aud_transform):

    aud_features = []
    for c in sorted(list(speech_data)):
        # print(c)
        for s in speech_data[c]:
            aud_features.append(s)

    aud_latent = aud_transform.predict(np.array(aud_features))
   
    img_features = []
    for c in sorted(list(img_data_test)):
        # print(c)
        for s in range(len(img_data_test[c]))[:16]:
            img_features.append(img_data_test[c][s])
    # exit()
    img_latent = img_transform.predict(np.array(img_features))

    # Get audio anchor confusions
    num_speakers = len(speech_data[list(speech_data)[0]])
    for i in range(NUM_CONF):
        cmat = np.zeros((NUM_CLASSES, NUM_CLASSES))

        for x, ca in enumerate(classes):
            spk_ind = np.random.randint(num_speakers)
            v1 = aud_latent[x*num_speakers + spk_ind]
            v1 = v1 / (np.linalg.norm(v1) + 1e-16)
            ind_done = 0

            for y, ci in enumerate(classes):
                num_images = 16#len(img_data_test[ci])
                img_ind = np.random.randint(num_images)
                v2 = img_latent[ind_done + img_ind]            
                v2 = v2 / (np.linalg.norm(v2) + 1e-16)
                ind_done+=num_images
                
                cmat[x][y] = np.dot(v1, v2)
        
        aud_folder = 'Confusions/{}_audio_anchor/'.format(folder)
        if not os.path.isdir(aud_folder):
            os.system('mkdir '+aud_folder)
        with open(aud_folder+'confusion_mat_'+str(i)+'.pkl','wb') as fp:
            pickle.dump(cmat, fp)

    
    # Get image anchor confusions
    for i in range(NUM_CONF):
        cmat = np.zeros((NUM_CLASSES, NUM_CLASSES))
        ind_done = 0

        for x, ci in enumerate(classes):
            num_images = 16#len(img_data_test[ci])
            img_ind = np.random.randint(num_images)
            v1 = img_latent[ind_done+ img_ind]
            v1 = v1 / (np.linalg.norm(v1) + 1e-16)
            ind_done+=num_images

            for y, ca in enumerate(classes):
                spk_ind = np.random.randint(num_speakers)
                v2 = aud_latent[y*num_speakers + spk_ind]
                v2 = v2 / (np.linalg.norm(v2) + 1e-16)
                
                cmat[x][y] = np.dot(v1, v2)

        img_folder = 'Confusions/{}_image_anchor/'.format(folder)
        if not os.path.isdir(img_folder):
            os.system('mkdir '+img_folder)
        with open(img_folder+'confusion_mat_'+str(i)+'.pkl','wb') as fp:
            pickle.dump(cmat, fp)

    return aud_folder, img_folder

def top_k(cmat, k):
    NUM_CLASSES = len(cmat)
    # print('num classes: ', NUM_CLASSES)
    acc = 0
    for x in range(NUM_CLASSES):
        topk_inds = np.argsort(cmat[x,:])[-k:]
        if x in topk_inds:
            acc += 1

    acc = acc/NUM_CLASSES
    return acc
    
def accuracy(folder, num_conf=15):  
    acc_top1 = 0
    acc_top5 = 0
    acc_top10 = 0
    for i in range(num_conf):
        with open(folder + 'confusion_mat_' + str(i) + '.pkl', 'rb') as fp:
            cmat = pickle.load(fp)
        # print(cmat)
        
        acc_top1 += top_k(cmat, 1)
        acc_top5 += top_k(cmat, 5)
        acc_top10 +=top_k(cmat, 10)
        
    acc_top1 = acc_top1/num_conf
    acc_top5 = acc_top5/num_conf
    acc_top10 = acc_top10/num_conf
    
    return acc_top1, acc_top5, acc_top10

# if __name__=='__main__':
    
#     set_gpu()
#     random_seed = set_random_seed(int(sys.argv[1]) if len(sys.argv)>1 else 0)

#     NUM_CONF = 15
#     NUM_CLASSES = 60
    
#     basepath = './Data/Features/'
#     # basepath = '/home/venkatk/Experiments/Audio-Visual-Deep-Multimodal-Networks-master/Data/'

#     # classes = open(basepath+'../classes.txt').read().split('\n')
#     # classes = classes[:-1]

#     img2lab = pickle.load(open('img2lab.pkl','rb'))
    
#     audio_test, img_test = get_data(path, ['audio_features_test', 'image_features_test'])
    
#     buckets = pickle.load(open('buckets.pkl','rb'))
#     filepath = 'Saved_models/noun_model_proxy_50/saved-proxy-audionetwork.hdf5'      # choose model to load
#     for filepath in ['mymodelk3.hdf5']:
#         img_transform, aud_transform = get_model(filepath)
#         img_latent = img_transform.predict(createdf2(img_test, img_lab2idx, samples=10, mode=0, noun=1)[:1])
#         aud_latent = aud_transform.predict(createdf2(audio_test, aud_lab2idx, samples=10, mode=1, noun=1)[:1])
#         # img_data_test,speech_data = bucket_data(img_data_test,speech_data,0)
#         get_confusion(aud_latent, aud_latent, sorted(list(img2lab)), num_conf=15, folder='confusions_noun', random_seed=random_seed)
#         print('\nImage retrieval accuracy:')
#         top1, top5, top10 = accuracy('Confusions/confusion_proxy_audio_anchor/')
#         print('Top 1: %.5f\tTop 5: %.5f\tTop10: %.5f'%(top1, top5, top10))
#         print('\nAudio retrieval accuracy:')
#         top1, top5, top10 = accuracy('Confusions/confusion_proxy_image_anchor/')
#         print('Top 1: %.5f\tTop 5: %.5f\tTop10: %.5f'%(top1, top5, top10))

