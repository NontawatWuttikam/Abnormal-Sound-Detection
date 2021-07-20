import pyaudio
import numpy as np
# from matplotlib import pyplot as plt
import librosa as lr
import cv2
import tensorflow as tf
import requests
url = 'https://notify-api.line.me/api/notify'
token = 'PUT YOUR LINE NOTIFY TOKEN HERE AND UNCOMMENT LINE 89'
headers = {'content-type':'application/x-www-form-urlencoded','Authorization':'Bearer '+token}

CHUNKSIZE = 2048

model_barking = tf.keras.models.load_model('./binary_model/Barking/DenseNet201_9_epoches_val_loss_ 0.1270 - val_accuracy_ 0.9591.h5')
model_car_alarm = tf.keras.models.load_model('./binary_model/Car Alarm/DenseNet201_16_epoches_val_loss_ 0.2043 - val_accuracy_ 0.9282.h5')
model_crying = tf.keras.models.load_model('./binary_model/Crying/DenseNet201_18_epoches_val_loss_ 0.1261 - val_accuracy_ 0.9627.h5')
model_explosion = tf.keras.models.load_model('./binary_model/Explosion/DenseNet201_11_epoches_val_loss_ 0.2408 - val_accuracy_ 0.8909.h5')
model_gunshot = tf.keras.models.load_model('./binary_model/Gunshot/DenseNet201_16_epoches_val_loss_ 0.3448 - val_accuracy_ 0.8629.h5')
model_interior_alarm = tf.keras.models.load_model('./binary_model/Interior_Alarm/DenseNet201_17_epoches_val_loss_ 0.1480 - val_accuracy_ 0.9555.h5')
model_screaming = tf.keras.models.load_model('./binary_model/Screaming/DenseNet201_8_epoches_val_loss_ 0.2941 - val_accuracy_ 0.8930.h5')
model_siren = tf.keras.models.load_model('./binary_model/Siren/DenseNet201_8_epoches_val_loss_ 0.2118 - val_accuracy_ 0.9232.h5')

class_list = ['Gunshot', 'Explosion','Car Alarm','Interior_Alarm','Siren', 'Screaming', 'Crying','Barking','..Other..']

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=22050, input=True, frames_per_buffer=CHUNKSIZE)

# _, ax = plt.subplots(1, 1)
ls = []
epsilon =1e-9
counter = 1
n = 2

data = stream.read(CHUNKSIZE)
numpydata = np.frombuffer(data, dtype=np.int16)
numpydata = numpydata/32768

S = lr.feature.melspectrogram(numpydata, sr = 22050)
S = np.log(S + epsilon)
S = lr.util.normalize(S)
S = np.kron(S, np.ones((n,n)))
Ss = S

temp = numpydata
while True:
    data = stream.read(CHUNKSIZE)
    numpydata = np.frombuffer(data, dtype=np.int16)
    numpydata = numpydata/32768
    temp = np.concatenate((temp,numpydata),axis=0)
    if counter == 108:
        S = lr.feature.melspectrogram(np.array(temp), sr = 44100)
        S = np.log(S + epsilon)
        S = lr.util.normalize(S)
        # plt.imsave('meltest.png',S)
        ximg = (S+1)/2
        ximg = np.expand_dims(ximg,-1)
        ximg = np.concatenate((ximg,ximg,ximg),-1)

        pred_barking = model_barking.predict(np.array([ximg]))[0]
        pred_car_alarm = model_car_alarm.predict(np.array([ximg]))[0]
        pred_crying = model_crying.predict(np.array([ximg]))[0]
        pred_explosion = model_explosion.predict(np.array([ximg]))[0]
        pred_gunshot = model_gunshot.predict(np.array([ximg]))[0]
        pred_interior_alarm = model_interior_alarm.predict(np.array([ximg]))[0]
        pred_screaming = model_screaming.predict(np.array([ximg]))[0]
        pred_siren = model_siren.predict(np.array([ximg]))[0]

        ypred = []

        if pred_crying[1] > 0.85:
            ypred.append(['crying',pred_crying[1]])
        if pred_car_alarm[1] > 0.85:
            ypred.append(['car_alarm',pred_car_alarm[1]])
        if pred_barking[1] > 0.85:
            ypred.append(['barking',pred_barking[1]])
        if pred_explosion[1] > 0.85:
            ypred.append(['explosion',pred_explosion[1]])
        if pred_gunshot[1] > 0.85:
            ypred.append(['gunshot',pred_gunshot[1]])
        if pred_interior_alarm[1] > 0.85:
            ypred.append(['interior_alarm',pred_interior_alarm[1]])
        if pred_screaming[1] > 0.85:
            ypred.append(['screaming',pred_screaming[1]])
        if pred_siren[1] > 0.85:
            ypred.append(['siren',pred_siren[1]])
        
        if len(ypred) == 0: ypred.append(['..other..',0.0])

        # if ypred != '..Other..' : requests.post(url, headers=headers, data = {'message':ypred+' prob : '+str(pred[agmx])})
        S = np.kron(S, np.ones((n,n)))
        S = (S+1)/2
        temp = []

        it = 0
        for yp in ypred:
            cv2.putText(S,yp[0]+' prob : '+str(yp[1]),(50, it+50),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2,cv2.LINE_AA)
            it += 30
        
        cv2.imshow('spec',S)
        cv2.waitKey(1)
        counter = 0
    counter += 1

# close stream
stream.stop_stream()
stream.close()
p.terminate()