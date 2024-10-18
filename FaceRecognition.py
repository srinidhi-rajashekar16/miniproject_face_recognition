from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd 
from tkinter import simpledialog
import numpy as np
import cv2
import os
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, load_model
from keras.models import model_from_json
import pickle
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint 
from keras.applications import ResNet50

main = tkinter.Tk()
main.title("Face Detection & Recognition") #designing main screen
main.geometry("800x700")

global filename
global model
names = ['Gia Mantegna', 'Nicole Gale Anderson', 'Samantha Barks', 'Sarah Hyland', 'Alexis Knapp', 'Alia Shawkat', 'Carly Schroeder', 'Diego Boneta',
         'Ella Rae Peck', 'Emma Watson', 'Jane Levy', 'Jennifer Lawrence', 'Kat Graham', 'Kristen Stewart', 'Liam Hemsworth', 'Shenae Grimes', 'Aaron Johnson',
         'Adelaide Clemens', 'Anton Yelchin', 'Claire Holt', 'Haley Joel Osment', 'Leah Pipes', 'Luke Kleintank', 'Mackenzie Rosman', 'Matthew Lewis',
         'Nikki Blonsky', 'Scout Taylor-Compton', 'Alex D. Linz', 'Allison Williams', 'AnnaLynne McCord', 'Daniel Radcliffe', 'Hayden Panettiere',
         'Hunter Parrish', 'Jesse McCartney', 'Keegan Allen', 'Kirsten Prout', 'Michael B. Jordan', 'Rosie Huntington-Whiteley',
         'Steven R. McQueen', 'Tori Black', 'Charlie McDermott', 'Chris Brown', 'Cody Horn', 'Cody Linley', 'Dakota Johnson', 'Danielle Panabaker']

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
global X, Y, X_train, y_train, X_test, y_test

def getID(name):
    index = -1
    name = name.split("_")
    names = ""
    for m in range(1, len(name)-1):
        names+=name[m]+" "
    name = names.strip()
    for i in range(len(names)):
        if names[i] == name:
            index = i
            break
    return index        

def uploadDataset():
    filename = filedialog.askdirectory(initialdir=".")
    textarea.delete('1.0', END)
    textarea.insert(END,filename+" loaded\n\n");
    textarea.insert(END,"Celebrities found in dataset are\n\n")
    for i in range(len(names)):
        textarea.insert(END,names[i]+"\n")
    
def processDataset():
    textarea.delete('1.0', END)
    global X, Y, X_train, y_train, X_test, y_test
    global filename
    if os.path.exists("model/X.txt.npy"):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])
                    img = cv2.resize(img,(300, 300), interpolation=cv2.INTER_CUBIC)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
                    print("Found {0} faces!".format(len(faces)))
                    if len(faces) > 0:
                        for (x, y, w, h) in faces:
                            sub_face = img[y:y+h, x:x+w]
                        img = cv2.resize(sub_face, (128, 128))
                    else:
                        img = cv2.resize(img, (128, 128))
                    label =  getID(directory[j])
                    for m in range(0, 4):
                        X.append(img)
                        Y.append(label)
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)
    X = X.astype('float32')
    X = X/255
    textarea.insert(END,"Dataset Preprocessing Completed\n")
    textarea.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n\n")
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
    textarea.insert(END,"80% images are used to train ResNet34 : "+str(X_train.shape[0])+"\n")
    textarea.insert(END,"20% images are used to train ResNet34 : "+str(X_test.shape[0])+"\n")        
    
def trainResnet():
    global X, Y, X_train, y_train, X_test, y_test
    textarea.delete('1.0', END)
    global model

    # Use ResNet50 from Keras applications
    resnet = ResNet50(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), 
                      weights='imagenet', include_top=False)

    for layer in resnet.layers:
        layer.trainable = False

    model = Sequential()
    model.add(resnet)
    model.add(Convolution2D(32, (1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32, (1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=y_train.shape[1], activation='softmax'))

    print(model.summary())
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    if not os.path.exists("model/model_weights.hdf5"):
        model_check_point = ModelCheckpoint(filepath='model/model_weights.hdf5', 
                                             verbose=1, save_best_only=True)
        hist = model.fit(X_train, y_train, batch_size=16, epochs=20, validation_data=(X_test, y_test),
                         callbacks=[model_check_point], verbose=1)
        with open('model/history.pckl', 'wb') as f:
            pickle.dump(hist.history, f)
    else:
        model = load_model("model/model_weights.hdf5")

    predict = model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    p = precision_score(testY, predict, average='macro') * 100
    r = recall_score(testY, predict, average='macro') * 100
    f = f1_score(testY, predict, average='macro') * 100
    a = accuracy_score(testY, predict) * 100

    textarea.insert(END, "HaarCascade & ResNet50 Accuracy  : " + str(a) + "\n")
    textarea.insert(END, "HaarCascade & ResNet50 Precision : " + str(p) + "\n")
    textarea.insert(END, "HaarCascade & ResNet50 Recall    : " + str(r) + "\n")
    textarea.insert(END, "HaarCascade & ResNet50 FSCORE    : " + str(f) + "\n\n")
       

def graph():
    f = open('model/history.pckl', 'rb')
    graph = pickle.load(f)
    f.close()
    accuracy = graph['accuracy']
    error = graph['loss']

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('EPOCH')
    plt.ylabel('Accuracy/Loss')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.plot(error, 'ro-', color = 'red')
    plt.legend(['ResNet34 Accuracy', 'ResNet34 Loss'], loc='upper left')
    plt.title('ResNet34 Training Accuracy & Loss Graph')
    plt.show()    

def faceRecognition():
    global model, names
    x1 = 0
    y1 = 0
    height = 0
    width = 0
    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image,(300, 300), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    print("Found {0} faces!".format(len(faces)))
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            sub_face = img[y:y+h, x:x+w]
            x1 = x
            y1 = y
            height = h
            width = w
        img = cv2.resize(sub_face, (128, 128))
        img = img.reshape(1,128,128,3)
        img = np.asarray(img)
        img = img.astype('float32')
        img = img/255
        preds = model.predict(img)
        predict = np.argmax(preds)
        img = cv2.imread(filename)
        cv2.rectangle(img, (x1, y1), (x1+width, y1+height), (0, 255, 0), 2)
        cv2.putText(img, names[predict], (x1, y1-10),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
        img = cv2.resize(img, (700,400))
        cv2.imshow('Face Recognized as : '+names[predict], img)
        cv2.waitKey(0)
    else:
        img = cv2.imread(filename)
        cv2.putText(img, "Unknown Face", (10,50),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
        img = cv2.resize(img, (700,400))
        cv2.imshow("Unknown Face", img)
        cv2.waitKey(0)       


font = ('times', 16, 'bold')
title = Label(main, text='Face Detection & Recognition in Organic Video: A Comparative Study for Sport Celebrities Database', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Celebrities Dataset", command=uploadDataset)
uploadButton.place(x=200,y=100)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=100,y=150)
processButton.config(font=font1)

trainButton = Button(main, text="Train HaarCascade & ResNet34", command=trainResnet)
trainButton.place(x=450,y=150)
trainButton.config(font=font1)

graphButton = Button(main, text="ResNet34 Training Graph", command=graph)
graphButton.place(x=200,y=200)
graphButton.config(font=font1)

predictButton = Button(main, text="Detect & Recognized Face using Test Images", command=faceRecognition)
predictButton.place(x=200,y=250)
predictButton.config(font=font1)

font1 = ('times', 12, 'bold')
textarea=Text(main,height=18,width=120)
scroll=Scrollbar(textarea)
textarea.configure(yscrollcommand=scroll.set)
textarea.place(x=10,y=300)
textarea.config(font=font1)

main.config(bg='light coral')
main.mainloop()
