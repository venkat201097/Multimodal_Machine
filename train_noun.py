import sys
import os
import logging
import random
import pickle
import numpy as np
import tensorflow as tf
from test_final import get_confusion, accuracy
from keras.callbacks import ModelCheckpoint
import model_triplet as models
from utils import get_free_gpu, set_gpu, set_random_seed, get_data, bucket_data, createdf2, createdf, createdata, createdata2, createtest

log = logging.getLogger('tensorflow')
log.setLevel(logging.ERROR)
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'

def generator_imgnet_neganchor(unseen_data, seen_data):
	num_unseen_data = len(unseen_data)
	num_seen_data = len(seen_data)
	unseen_data = np.array(unseen_data)
	seen_data = np.array(seen_data)

	j = 0
	while True:
		unseen_data_batch = unseen_data
		seen_data_batch = seen_data[j:j+BATCH_SIZE-num_unseen_data]

		# if len(seen_data_batch)<BATCH_SIZE-num_unseen_data:
			# j = 0
		if j%num_seen_data>=0:
			np.random.shuffle(seen_data)
			seen_data_batch = seen_data[j:j+BATCH_SIZE-num_unseen_data]

		df = np.append(unseen_data_batch, seen_data_batch, axis=0)
		np.random.shuffle(df)
		anchor_labels = df[:, 1]

		if len(anchor_labels)<BATCH_SIZE:
			CURR_BATCH_SIZE = len(anchor_labels)
		else:
			CURR_BATCH_SIZE = BATCH_SIZE

		groundings = df[:, 2]
		seen_mask = df[:, 3]

		anchor_aud = np.zeros((CURR_BATCH_SIZE, MAX_LEN, 80))
		anchor_img = np.zeros((CURR_BATCH_SIZE, 2048))
		for s,x in enumerate(df[:]):                
			if x[2]:
				anchor_aud[s, :len(x[0]), :] = x[0]
			else:
				anchor_img[s] = x[0]
		
		yield [groundings, 1-groundings, seen_mask, np.array(anchor_aud), np.array(anchor_img), np.array(anchor_labels)], np.zeros(CURR_BATCH_SIZE)
		j += BATCH_SIZE//2


if __name__=='__main__':
	
	set_gpu()
	random_seed = set_random_seed(int(sys.argv[1]) if len(sys.argv)>1 else 0)

	BATCH_SIZE = 512
	NUM_EPOCHS = 3
	MARGIN = 0.8
	
	path = '/home/venkatk/Experiments/New-Experiment/'#Data/Features/'
	imgnet_path = '/home/venkatk/Experiments/Audio-Visual-Deep-Multimodal-Networks-master/'

	# Load Data -------------------------------------------------------------------------
	imgnet_audio_train, imgnet_audio_val, imgnet_img_train, imgnet_img_val = get_data(imgnet_path+'Data/', ['audio_features_train', 'audio_features_val', 
		'image_features_train', 'image_features_val'])
	MAX_LEN = imgnet_audio_train[list(imgnet_audio_train)[0]].shape[1]

	audio_train, audio_val, audio_test, img_train, img_val, img_test = get_data(path+'Data/Features/', ['audio_features_train', 'audio_features_val', 'audio_features_test',
		'image_features_train', 'image_features_val', 'image_features_test'])

	buckets = pickle.load(open(path+'buckets.pkl','rb'))
	shots = [1,2,3,1,2,3]
	shot2num_steps = [5, 2, 1]
	img2lab = pickle.load(open(path+'img2lab.pkl','rb'))
	lab2img = {j:i for i,j in img2lab.items()}
	img_lab2idx = {x:i for i,x in enumerate(list(img2lab))}
	aud_lab2idx = {img2lab[x]:i for i,x in enumerate(list(img2lab))}

	imgnet_classes = open(imgnet_path+'classes.txt','r').read().split('\n')[:-1]
	imgnet_lab2idx = {c:i for i,c in enumerate(imgnet_classes)}
	
	# Preprocessing ------------------------------------------------------------------------------
	imgnet_train = np.append(createdf2(imgnet_img_train, imgnet_lab2idx, samples=3, mode=0, mask=1), createdf2(imgnet_audio_train, imgnet_lab2idx, samples=3, mode=1, mask=1), axis=0)
	
	imgnet_val = np.append(createdf2(imgnet_img_val, imgnet_lab2idx, samples=3, mode=0, mask=1), createdf2(imgnet_audio_val, imgnet_lab2idx, samples=3, mode=1, mask=1), axis=0)

	for bucket, shots in zip(range(6), [1,2,3,1,2,3]):
		
		# Get bucket data
		img_train_bucket, audio_train_bucket = bucket_data(img_train, audio_train, img2lab, buckets, bucket=bucket)
		img_test_bucket, audio_test_bucket = bucket_data(img_test, audio_test, img2lab, buckets, bucket=bucket)
		img_val_bucket, audio_val_bucket = bucket_data(img_val, audio_val, img2lab, buckets, bucket=bucket)

		# Preprocess data - 
		# Train: [features, class, mode, mask], mode=0 if image else 1, mask=1 if seen else 0 
		bucket_train, test_img, test_aud = createdata2(img_train_bucket, audio_train_bucket, img_test_bucket, audio_test_bucket, img2lab, lab2img, img_lab2idx, aud_lab2idx, samples=shots)
		print('Train: ',bucket_train.shape, ' Test (Img, Aud): ', test_img.shape, test_aud.shape)
		'''test[bucket] = (test_img,test_aud)
		val = createdf(img_val_bucket, audio_val_bucket, samples=10, mode=1)
		val = np.append(createdf2(img_val_bucket, img_lab2idx, samples=3, mode=0, mask=0), createdf2(audio_val_bucket, aud_lab2idx, samples=3, mode=1, mask=0), axis=0)'''
		seen_train = np.empty((0,4))
		
		folder = path+"Saved_models2/train_noun"
		if not os.path.isdir(folder):
			os.system('mkdir '+folder)
		savefile = folder+"/nountf_bucket-{}_tf-{}_triplet-{}_epochs-{}_batch-{}_margin-{}.hdf5".format(bucket, 'yes', 'neganchor', NUM_EPOCHS, BATCH_SIZE, MARGIN)#-{epoch:02d}.hdf5"
		
		'''savefile = 'mymodelk9.hdf5'
		checkpoint = ModelCheckpoint(savefile, verbose=1, save_best_only=True, save_weights_only=True, period=1)
		callbacks_list = [checkpoint]'''

		jointnet = models.JointNet()
		model = jointnet.model

		# Load pretrained weights
		model.load_weights(path+'Saved_models/imagenet_model_proxy_audionoisy_100/saved-proxy-audionetwork.hdf5', by_name=True)

		for shot in range(shots):
			shot_train = bucket_train[20*shot:20*shot+20]

			# Seen data: Imagenet + First 10 samples of (n-1)th shot, n=2,3
			seen_train = np.append(seen_train, imgnet_train, axis=0)

			for samples in range(10):

				# Unseen: nth set of 10 samples, n=1,2,3
				unseen_train = shot_train[2*samples:2*samples+2]
				print('Unseen train: ', unseen_train.shape, 'Seen train: ', seen_train.shape)

				# ---- Train ------------------------------------------------
				history = model.fit_generator(generator_imgnet_neganchor(unseen_train, seen_train), epochs=1, steps_per_epoch = shot2num_steps[shot])#len(seen_train)//BATCH_SIZE)#, 
					# callbacks=callbacks_list, initial_epoch=0, validation_data = generator_imgnet_neganchor(val),validation_steps = len(val)//BATCH_SIZE)

			# Add nth set to seen data - mask=1
			seen_train = bucket_train[:20*shot+20]
			seen_train[:,3] = [1 for i in range(len(seen_train))]
			# print('unseen to seen',seen_train[:,3])

		# ---- Test ------------------------------------------------
		# model.load_weights(savefile)

		# Extract latent vectors of test data
		img_transform, aud_transform = jointnet.image_submodel, jointnet.audio_submodel
		img_latent = img_transform.predict(test_img)#createtest(data=img_test, classes=sorted(list(img_train_bucket)), samples=10))
		aud_latent = aud_transform.predict(test_aud)#createtest(data=audio_test, classes=[img2lab[i] for i in sorted(list(img_train_bucket))], samples=10))

		# Get confusion matrices and score
		aud_folder, img_folder = get_confusion(aud_latent, img_latent, sorted(list(img_train_bucket)), path=path, 
			folder='confusions_noun_bucket-{}'.format(bucket), num_conf=15, random_seed=random_seed)

		with open(path+'testresults_noun/fresh1.csv','a+') as fp:
			top1, top5, top10 = accuracy(aud_folder)
			print('\nImage retrieval accuracy:\tTop 1: %.5f \tTop 5: %.5f\tTop10: %.5f'%(top1, top5, top10))
			print(str(bucket)+',i,%.5f,%.5f,%.5f'%(top1, top5, top10), file=fp)

			top1, top5, top10 = accuracy(img_folder)
			print('\nAudio retrieval accuracy:\tTop 1: %.5f \tTop 5: %.5f \tTop10: %.5f'%(top1, top5, top10))
			print(str(bucket)+',a,%.5f,%.5f,%.5f'%(top1, top5, top10), file=fp)

		


