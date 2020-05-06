# import tensorflow as tf
# import tensorflow.keras as keras
# import tensorflow.keras.backend as K
# from tensorflow.keras.models import Model, load_model, Sequential
# from tensorflow.keras.layers import Lambda, Layer, Dense, Input, Dropout, concatenate, Multiply, Add, Masking, LSTM, BatchNormalization, Activation, TimeDistributed

import tensorflow as tf
import keras as keras
import keras.backend as K
from keras.models import Model, load_model, Sequential
from keras.layers import Lambda, Layer, Dense, Input, Dropout, concatenate, Multiply, Add, Masking, LSTM, BatchNormalization, Activation, TimeDistributed

import pickle
import triplet_loss_tf2 as TL
class JointNet():
    def __init__(self):
        self.audio_submodel = self.audio_submodel()
        self.image_submodel = self.image_submodel()
        self.model = self.joint_model_online()

    def identity_loss(self, y_true, y_pred):
        return K.mean(y_pred)

    def triplet_loss(self, X, alpha=0.4):
        anchor,positive,negative = X

        pos_dist = K.sum(K.square(anchor-positive),axis=1)
        neg_dist = K.sum(K.square(anchor-negative),axis=1)
        
        loss = K.maximum(pos_dist-neg_dist+alpha,0.0)
        return loss

    def audio_submodel(self):
        input_size = 80
        hidden_size = 128
        
        model = Sequential()
        model.add(Masking(mask_value=0.0, input_shape=(None, input_size), name='masking_1'))        
        model.add(LSTM(hidden_size, return_sequences=True, input_shape=(None, input_size), name='lstm_1', trainable=True))
        model.add(LSTM(hidden_size, return_sequences=False, input_shape=(None, hidden_size), name='lstm_2', trainable=True))
        model.add(Dense(2048, name='dense_2048', trainable=True))
        model.add(BatchNormalization(name='batch_normalization_2048', trainable=True))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # model.load_weights('Saved_models/imagenet_model_audio_triplet_subnetwork.h5', by_name=True)

        inp = Input((None, 80))
        op1 = model(inp)
        op2 = Dense(576, activation='tanh', input_shape=(2048, ), name='dense_aud1', trainable=True)(op1)
 
        model = Model(inputs=inp, outputs=op2, name='sequential_1')
        return model
    
    def image_submodel(self):
        inp = Input((2048, ))
        op1 = Dropout(0.5)(inp)
        op2 = Dense(576, activation='tanh', name='dense_img1', trainable=True)(op1)

        model = Model(inputs=inp, outputs=op2, name='sequential_2')
        return model

    def joint_model_offline(self):
        NUM_CLASSES = 10

        grounding = Input((1, ), name='grounding')
        grounding_bar = Input((1, ), name='grounding_bar')
        anchor_aud = Input((None, 80), name='anchor_aud')
        anchor_img = Input((2048, ), name='anchor_img')
        pos_aud = Input((None, 80), name='pos_aud')
        pos_img = Input((2048, ), name='pos_img')
        neg_aud = Input((None, 80), name='neg_aud')
        neg_img = Input((2048, ), name='neg_img')

        # class_mask = Input((NUM_CLASSES, ), name='class_mask')
        # class_mask_bar = Input((NUM_CLASSES, ), name='class_mask_bar')

        anchor_aud_latent = self.audio_submodel(anchor_aud)
        anchor_img_latent = self.image_submodel(anchor_img)
        pos_aud_latent = self.audio_submodel(pos_aud)
        pos_img_latent = self.image_submodel(pos_img)
        neg_aud_latent = self.audio_submodel(neg_aud)
        neg_img_latent = self.image_submodel(neg_img)
     
        anchor = Add()([Multiply()([grounding_bar, anchor_img_latent]), Multiply()([grounding, anchor_aud_latent])])     
        anchor_norm = Lambda(lambda x: K.l2_normalize(x, axis=-1))(anchor)

        pos = Add()([Multiply()([grounding, pos_img_latent]), Multiply()([grounding_bar, pos_aud_latent])])     
        pos_norm = Lambda(lambda x: K.l2_normalize(x, axis=-1))(pos)

        neg = Add()([Multiply()([grounding, neg_img_latent]), Multiply()([grounding_bar, neg_aud_latent])])     
        neg_norm = Lambda(lambda x: K.l2_normalize(x, axis=-1))(neg)

        loss = Lambda(self.triplet_loss,output_shape=(1,),name='loss')([anchor_norm, pos_norm, neg_norm])
        
        model = Model(input=[grounding, grounding_bar, anchor_aud, anchor_img, pos_aud, pos_img, neg_aud, neg_img],output=loss)
        model.compile(loss=self.identity_loss, optimizer=keras.optimizers.Adam(lr=0.001, decay=1e-5))
        return model

    def joint_model_online(self):
        NUM_CLASSES = 60

        grounding = Input((1, ), name='grounding')
        grounding_bar = Input((1, ), name='grounding_bar')
        seen_mask = Input((1,), name='seen_mask')
        # noun_bar = Input((1,), name='noun_bar')
        anchor_aud = Input((None, 80), name='anchor_aud')
        anchor_img = Input((2048, ), name='anchor_img')
        anchor_proxy = Input((576,), name='anchor_proxy')
        class_mask = Input((1, ), name='class_mask')
        
        anchor_aud_latent = self.audio_submodel(anchor_aud)
        anchor_img_latent = self.image_submodel(anchor_img)
     
        anchor = Add()([Multiply()([grounding_bar, anchor_img_latent]), Multiply()([grounding, anchor_aud_latent])])    
        # anchor = Add()([Multiply()([noun_bar, anchor_proxy]), Multiply()([noun, anchor])]) 
        anchor_norm = Lambda(lambda x: K.l2_normalize(x, axis=-1))(anchor)

        # loss = Lambda(TL.batch_all_triplet_loss,output_shape=(1,),name='loss')([class_mask, grounding, anchor_norm])
        loss = Lambda(TL.batch_all_triplet_loss,output_shape=(1,),name='loss')([class_mask, seen_mask, anchor_norm])
        
        # model = Model(inputs=[grounding, grounding_bar, anchor_aud, anchor_img, class_mask],outputs=loss)
        model = Model(inputs=[grounding, grounding_bar, seen_mask, anchor_aud, anchor_img, class_mask],outputs=loss)
        # model = Model(inputs=[grounding, grounding_bar, noun, noun_bar, anchor_aud, anchor_img, anchor_proxy, class_mask],outputs=loss)
        model.compile(loss=self.identity_loss, optimizer=keras.optimizers.Adam(lr=0.001, decay=1e-5))
        return model
