import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing import image 
import time

tic = time.time()

cap = cv2.imread('testing_image/sepi/sepi10.jpg')
car_img = cap.copy()

model = tf.keras.models.load_model('h5/epoch_40.h5')

car_cascade = cv2.CascadeClassifier('xml/mycardetectorfix.xml')
car_rects = car_cascade.detectMultiScale(car_img,scaleFactor=1.5,minNeighbors=5)

count=0

for (x,y,w,h) in car_rects:
    cv2.imwrite('temp.jpg',car_img[y:y+h,x:x+w])
    uji_img = image.load_img('temp.jpg',target_size=(150,150))
    uji_img = image.img_to_array(uji_img)
    uji_img = np.expand_dims(uji_img,axis=0)
    uji_img = uji_img/255
    b = np.amax(model.predict(uji_img)[0])
    p = b*100
    print(b)

    data_uji = (model.predict_classes(uji_img)[0])
    print(data_uji)
    if data_uji == 1:
        count = count+1
        cv2.rectangle(car_img, (x,y), (x+w, y+h), (0,0,255), 3)
        cv2.putText(car_img, 'Mobil ' + str(int(p)) + '%',(x+2,y-10),cv2.FONT_HERSHEY_DUPLEX,0.7,(0,0,255),2) 
        print('Haarcascade count :'+ str(car_rects.shape[0]))
        print('Haarcascade and cnn count :'+ str(count))

toc = time.time() - tic
print("Computation time is "+str(toc)+ " seconds")

cv2.putText(car_img, str(count), (0, car_img.shape[0] -10), cv2.FONT_HERSHEY_DUPLEX, 4, (0,0,255), 2)
cv2.imwrite("resimg.jpg", car_img)
