#importing all the libraries
import os
import csv
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Lambda, Cropping2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


samples = []
batch_size=64
correction = 0.2 # this is a parameter to tune

with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        samples.append(line)

        
def generator(samples, batch_size):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
                #for center image
                name_c = 'data/IMG/'+ batch_sample[0].split('/')[-1]
                center_image = mpimg.imread(name_c)
                center_angle = float(batch_sample[3])             
                images.append(center_image)
                angles.append(center_angle)
                
                #for left image
                name_l = 'data/IMG/'+ batch_sample[1].split('/')[-1]
                left_image = mpimg.imread(name_l)
                images.append(left_image)
                left_angle = center_angle + correction
                angles.append(left_angle)
                
                #for right image
                name_r = 'data/IMG/'+ batch_sample[2].split('/')[-1]
                right_image = mpimg.imread(name_r)
                images.append(right_image)
                right_angle = center_angle - correction
                angles.append(right_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

            
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


#model
model = Sequential()

#normalizing and cropping 
model.add(Lambda(lambda x:x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))

#layer 1
model.add(Conv2D(32, (5,5), strides=2, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(rate=0.3))

#layer 2
model.add(Conv2D(32, (5,5), strides=2, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(rate=0.3))

model.add(Flatten())

#fc 1
model.add(Dense(32, activation='relu'))
model.add(Dropout(rate=0.3))
#fc 2
model.add(Dense(32, activation='relu'))
#fc 3
model.add(Dense(1))

model.summary()

#checkpoint 
checkpoint = ModelCheckpoint("model.h5", monitor="val_loss", mode="min", save_best_only = True, verbose=1)

#for earlystopping : if the val_loss is not improving
earlystop = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 3, verbose = 1, restore_best_weights = True)

#for reducing the Learning Rate : if the val_loss remains constant or is not improving
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 3, verbose = 1, min_delta = 0.0001)

callbacks = [earlystop, checkpoint, reduce_lr]

model.compile(loss='mse', optimizer='adam')
hh = model.fit_generator(train_generator, steps_per_epoch=np.ceil(len(train_samples)/batch_size),
                    validation_data=validation_generator,validation_steps=np.ceil(len(validation_samples)/batch_size),
                    callbacks=callbacks, epochs=10, verbose=1)



