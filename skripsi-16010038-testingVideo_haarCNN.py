import cv2
import numpy as np
import time
import tensorflow as tf
from keras.preprocessing import image 

cap = cv2.VideoCapture('testing_video/night.mp4')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

res = cv2.VideoWriter('res1.mp4', cv2.VideoWriter_fourcc('M','J','P','G'),30,(frame_width,frame_height))

model = tf.keras.models.load_model('h5/epoch_40.h5')
car_cascade = cv2.CascadeClassifier('xml/mycardetectorfix.xml')

start = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
            break

    tic = time.time()
    car_rects = car_cascade.detectMultiScale(frame,scaleFactor=1.5,minNeighbors=5)

    count = 0
    for (x, y, w, h) in car_rects:
        cv2.imwrite('temp.jpg',frame[y:y+h,x:x+w])
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
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
            cv2.putText(frame, 'Mobil ' + str(int(p)) + '%',(x+2,y-10),cv2.FONT_HERSHEY_DUPLEX,0.7,(0,0,255),2) 
            print('Haarcascade count :'+ str(car_rects.shape[0]))
            print('Haarcascade and cnn count :'+ str(count))
    toc = time.time() - tic
    print("Frame Computation time is "+ str(toc)+ " seconds")

    cv2.putText(frame, str(count), (0, frame.shape[0] -10), cv2.FONT_HERSHEY_DUPLEX, 4, (0,0,255), 2)

    cv2.imshow('res', frame)
    res.write(frame)

    if cv2.waitKey(1)& 0xFF == ord('q'):
	    break

end = time.time() - start
print("Final Computation time is "+str(end)+ " seconds")

cap.release()
res.release()
cv2.destroyAllWindows()
	
