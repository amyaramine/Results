#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import glob
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from sklearn.utils import shuffle


np.random.seed(1337)  # for reproducibility
batch_size = 256
nb_classes = 2
nb_epoch =  1000

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

split = 0.3

img_rows = 100
img_cols = 100


List_patient_sizes = []


Prediction = pd.read_csv("../2017/Prediction.csv", sep=";")

book = open("ResultsTanh.txt", "w")


Y_train_Prediction = open("Prediction.txt", "r").read()
Y_train_Prediction = Y_train_Prediction.split("\r\n")

print len(Y_train_Prediction),Y_train_Prediction

os.chdir('../2017/ImagesUpdatedAndAugmented100X100Center')
liste = os.listdir('../ImagesUpdatedAndAugmented100X100Center')

#Sauvegarder un model
def save_model(model, index, cross=''):
    json_string = model.to_json()
    if not os.path.isdir('Models'):
        os.mkdir('Models')
    json_name = 'architecture' + str(index) + cross + '.json'
    weight_name = 'model_weights' + str(index) + cross + '.h5'
    open(os.path.join('Models', json_name), 'w').write(json_string)
    model.save_weights(os.path.join('Models', weight_name), overwrite=True)


#Lire un Model
def read_model(index, cross=''):
    json_name = 'architecture' + str(index) + cross + '.json'
    weight_name = 'model_weights' + str(index) + cross + '.h5'
    model = model_from_json(open(os.path.join('Models', json_name)).read())
    model.load_weights(os.path.join('Models', weight_name))
    return model

def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


def merge_several_folds_geom(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a *= np.array(data[i])
    a = np.power(a, 1/nfolds)
    return a.tolist()

def CNN_Model4CouchesNew(img_rows, img_cols, color_type=1):
  model = Sequential()
  model.add(Convolution2D(16, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(color_type, img_rows, img_cols)))
  model.add(Activation('tanh'))
  model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
  model.add(Dropout(0.25))

  model.add(Convolution2D(32, nb_conv, nb_conv))
  model.add(Activation('tanh'))
  model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
  model.add(Dropout(0.25))

  model.add(Convolution2D(64, nb_conv, nb_conv))
  model.add(Activation('tanh'))
  model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
  model.add(Dropout(0.25))

  model.add(Convolution2D(128, nb_conv, nb_conv))
  model.add(Activation('tanh'))
  model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
  model.add(Dropout(0.25))

  model.add(Flatten())
#  model.add(Dense(256))
#  model.add(Activation('relu'))
#  model.add(Dropout(0.5))
  model.add(Dense(128))
  model.add(Activation('tanh'))
  model.add(Dropout(0.5))
  model.add(Dense(nb_classes))
  model.add(Activation('sigmoid')) 
 # model.load_weights('../larger_model.h5')
 # model.layers.pop()
 # model.add(Dense(nb_classes))
 # model.add(Activation('sigmoid'))
  model.summary()
  model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
  print "model compiled"
  return model

def little_Model(img_rows, imgcols, color_type=1):
        model = Sequential()
        model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(color_type, img_rows, img_cols)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        model.add(Dropout(0.25))

        model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        model.summary()
        model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
        print "model compiled"
        return model


def CNN_Model4Couches(img_rows, img_cols, color_type=1):
  model = Sequential()
  model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(color_type, img_rows, img_cols)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
  model.add(Dropout(0.25))

  model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
  model.add(Dropout(0.25))

  model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
  model.add(Dropout(0.25))

  model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(128))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(nb_classes))
  model.add(Activation('sigmoid'))

  model.summary()
  model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
  print "model compiled"
  return model

def CNN_Model2(img_rows, img_cols, color_type=1):
    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(color_type, img_rows, img_cols)))


    model.add(Activation('relu'))
    model.add(Convolution2D(64, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(256, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(Convolution2D(256, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(Convolution2D(256, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(512, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(Convolution2D(512, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(Convolution2D(512, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
    print "Model Compiled"
    return model


def vgg_std16_model(img_rows, img_cols, color_type=1):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(1,
                                                 img_rows, img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    # model.load_weights('vgg16_weights.h5')
    model.summary()
    # Code above loads pre-trained data and
    model.layers.pop()
    model.add(Dense(nb_classes, activation='sigmoid'))
    # Learning rate is changed to 0.001
    # sgd = SGD(lr=1e-3i, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer='sgd', loss='binary_crossentropy')
    return model


i = 0
for element in Prediction:
    book.write("Test numero = "+ str(i+1))	
    print "Test numero : ", i+1
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    Y_result = []

    listeIndexApprentissage = []
    listeIndexTest = []
    index = 0
    indicePatient = 0
    apprentissage = 0
    test = 0
    listeApprentissage = []
    listeTest = []


    for Patient in Prediction.ix[:,i]:

        if  (Patient  == 1) :
            apprentissage += 1
            listeApprentissage.append(liste[indicePatient])
            listeIndexApprentissage.append(index)
            index += 1

        else:
            test += 1
            listeTest.append(liste[indicePatient])
            listeIndexTest.append(index)
            index += 1
        indicePatient += 1

    print "Element = ",element
    print "Données pour apprentissage = ", len(listeApprentissage)
    print "Données pour test = ", len(listeTest)
    print "\n"

    index = 0
    print "Charger les donnees d'apprentissage"
    for eletA in listeApprentissage:

        liste2 = glob.glob("../ImagesUpdatedAndAugmented100X100Center/" + eletA + '/*')[0]

        img = glob.glob(liste2 + "/*")

        for tum in img:
            tumeur = np.load(tum)
            X_train.append(tumeur)
            Y_train.append(Y_train_Prediction[listeIndexApprentissage[index]])

        liste2 = glob.glob("../ImagesUpdatedAndAugmented100X100Center/" + eletA + '/*')[1]
        img = glob.glob(liste2+"/*/*")
        for tum in img:
            tumeur = np.load(tum)
            X_train.append(tumeur)
            Y_train.append(Y_train_Prediction[listeIndexApprentissage[index]])


        index += 1


    print "Donnees chargees"
    print "X_train = ",len(X_train)
    print "Y_train =",len(Y_train)
    print "\n"

    index = 0
    print "Charger les donnees de test"
    for eletT in listeTest:
        liste2 = glob.glob("../ImagesUpdatedAndAugmented100X100Center/" + eletT + '/*')[0]
        img = glob.glob(liste2 + "/*")


        for tum in img:
            tumeur = np.load(tum)
            X_test.append(tumeur)
            Y_test.append(Y_train_Prediction[listeIndexTest[index]])
        Y_result.append(Y_train_Prediction[listeIndexTest[index]])
        index += 1

    print "Donnees chargees"
    print "X_test= ",len(X_test)
    print "Y_test =",len(Y_test)
    print "\n"
    print "Nombre total de coupes = ", len(X_train) + len(X_test)*6
    print "Nombre total de coupes = ", len(Y_train) + len(Y_test)*6

    print "**********************\n"
    X_train = np.asarray(X_train)
    X_train = shuffle(X_train, random_state = 0)
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    print X_train.shape

    X_test = np.asarray(X_test)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    
    Y_train = np.asarray(Y_train, dtype=np.int32)	
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_train = shuffle(Y_train, random_state = 0)

    Y_test = np.asarray(Y_test, dtype=np.int32)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
    #Y_result = np_utils.to_categorical(Y_result, nb_classes)

    print len(listeTest), listeTest
    print X_test.shape

   # X_train = X_train[0:100]
   # Y_train = Y_train[0:100] 

    #print len(Y_result), Y_result

    model = CNN_Model4CouchesNew(img_rows, img_cols, 1)
    model.fit(X_train, Y_train, batch_size=batch_size,
              nb_epoch=nb_epoch,
              verbose=1,
              validation_split=split, shuffle=True)

    score = model.evaluate(X_test, Y_test, verbose=0)
    book.write("\nTest score for slices: " + str(score[0]))	
    print('Test score for slices:', score[0])
    print('Test accuracy for slices:', score[1])
    book.write("\nTest accuracy for slices: "+ str(score[1]))
	
    results = model.predict_classes(X_test)


    k = 0

    listePrediction = []
    for elet in listeTest:
        liste2 = glob.glob("../ImagesUpdatedAndAugmented100X100Center/" + elet + '/*')[0]
        liste2 = glob.glob(liste2 + "/*")
        j = 0
        PredictionReponse = 0
        while(j < len(liste2)):
            PredictionReponse += results[k]
            k += 1
            j += 1
        PredictionReponse = float(PredictionReponse) / float(j)
        if(PredictionReponse > 0.5):
            listePrediction.append(1)
        else:
            listePrediction.append(0)

    print len(listeTest), listeTest
    print len(listePrediction), listePrediction



    accuracy = 0
    indexPrediction = 0
    while (indexPrediction < len(Y_result)):
        if (listePrediction[indexPrediction] == int(Y_result[indexPrediction])):
            accuracy += 1
        indexPrediction += 1

    accuracy = float(accuracy) / float(len(listePrediction))
    print "Accuracy :", accuracy
    book.write("\nAccuracy : " + str(accuracy))

    VP = 0
    FN = 0
    FP = 0
    VN = 0
    indexPrediction = 0
    while (indexPrediction < len(listePrediction)):
	#print "listePrediction = ", listePrediction[indexPrediction]
	#print "Y_result = ", Y_result[indexPrediction]
        if ((listePrediction[indexPrediction] == 1) & (int(Y_result[indexPrediction]) == 1)):
            VP += 1
        elif ((listePrediction[indexPrediction] == 0) & (int(Y_result[indexPrediction]) == 1)):
            FN += 1
        elif ((listePrediction[indexPrediction] == 0) & (int(Y_result[indexPrediction]) == 0)):
            VN += 1
        else:
            FP += 1
        indexPrediction += 1

    sens = float(VP) / float(float(VP) + float(FN))
    spec = float(VN) / float(float(VN) + float(FP))
    	
    print "Confusion Matrix"
    print "            Malade      Non Malade "
    print "TestP     ", VP, "        ", FP
    print "TestN     ", FN, "        ", VN
    print "**************************************", VP + FP + FN + VN
    print "Sensibilite = ", sens
    print "Specificite = ", spec
    
    book.write("\n Sens = "+str(sens))
    book.write("\n Spec = "+str(spec))
    book.write("\n Vp = "+str(VP))
    book.write("\n FP = "+str(FP))
    book.write("\n FN = "+str(FN))
    book.write("\n VN = "+str(VN))
    book.write("\n\n****************\n")
    # Compute ROC curve and ROC area for each class
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()

    # for i in range(nb_classes):
    #     fpr[i], tpr[i], _ = roc_curve(Y_result[:, i], Y_result[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    #
    # # Compute micro-average ROC curve and ROC area
    # fpr["micro"], tpr["micro"], _ = roc_curve(Y_result.ravel(), Y_result.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    #
    # plt.figure()
    # lw = 2
    # plt.plot(fpr[1], tpr[1], color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()
    #
    # # Compute macro-average ROC curve and ROC area
    #
    # # First aggregate all false positive rates
    # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))
    #
    # # Then interpolate all ROC curves at this points
    # mean_tpr = np.zeros_like(all_fpr)
    # for i in range(nb_classes):
    #     mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    #
    # # Finally average it and compute AUC
    # mean_tpr /= nb_classes
    #
    # fpr["macro"] = all_fpr
    # tpr["macro"] = mean_tpr
    # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    #
    # # Plot all ROC curves
    # plt.figure()
    # plt.plot(fpr["micro"], tpr["micro"],
    #          label='micro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["micro"]),
    #          color='deeppink', linestyle=':', linewidth=4)
    #
    # plt.plot(fpr["macro"], tpr["macro"],
    #          label='macro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["macro"]),
    #          color='navy', linestyle=':', linewidth=4)
    #
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # for i, color in zip(range(nb_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #              label='ROC curve of class {0} (area = {1:0.2f})'
    #                    ''.format(i, roc_auc[i]))
    #
    # plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Some extension of Receiver operating characteristic to multi-class')
    # plt.legend(loc="lower right")
    # plt.show()

    i += 1
book.close()
