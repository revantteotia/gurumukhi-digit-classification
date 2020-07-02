import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn


def loadAndpreprocessTestData(test_dataset_path):
    '''
    To load test data from disk and scale the images between 0-1

    Takes path of test dataset as input and returns batches of tensor image data

    The directory structure of the test dataset should be like :
    
        testdataset/
            0/
                img0
                img1
                ...
            1/
                img0
                img1
                ...
            .
            .
            .

            9/
                img0
                ...
    '''

    IMAGE_SIZE = 32
    test_image_generator = ImageDataGenerator( rescale=1./255 ) # rescaling the test images 

    test_data_gen = test_image_generator.flow_from_directory(   # to load image data from disk and batch them
        test_dataset_path, 
        target_size=(IMAGE_SIZE, IMAGE_SIZE), 
        color_mode='grayscale', # for 1 channel images
        class_mode='sparse', # labels will be integers
        batch_size=32, 
        shuffle=True, 
        seed=None,
        )
    
    return test_data_gen

def loadTestData(test_dataset_path):
    '''
    Takes the path to the test dataset and returns :
        X_test : array of image arrays
        y_test : list of labels of the images

    '''
    
    test_data_gen = loadAndpreprocessTestData(test_dataset_path)
    X_test, y_test = test_data_gen[0]


    for idx in range(1, len(test_data_gen)):
        x, y = test_data_gen[idx]
        print(x.shape, y.shape)
        X_test = np.concatenate((X_test, x), axis=0)
        y_test = np.concatenate((y_test, y), axis=0)

    return X_test, y_test    

def plotConfusionMatrix(y_true, y_pred):

    confusionMatrix = confusion_matrix(y_true, y_pred)
    
    df_cm = pd.DataFrame(confusionMatrix, range(10), range(10))

    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

    # plt.matshow(confusionMatrix)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()

def evaluateTestData(model, X_test, y_test):
    '''
    Takes trained model, X_test, y_test and calculates loss, accuracy other metrics

    '''
    print("\n================== Evaluating Test Data ===================\n")

    evaluation = model.evaluate(X_test, y_test, verbose = 0)
    print("Evaluation on test data :")
    print("loss = {}, accuracy = {}".format(evaluation[0], evaluation[1]))

    prediction_softmax = model.predict(X_test)
    predictedDigits = np.argmax(prediction_softmax, axis=1)
    # print ("prediction =" )
    # print(predictedDigits)

    plotConfusionMatrix(y_test, predictedDigits)

if __name__ == "__main__":

    test_dataset_path = "gurumukhi_digits_dataset/val"

    X_test, y_test = loadTestData(test_dataset_path)

    # loading trained model
    model = tf.keras.models.load_model('multi_hidden_layer_NN.h5')
    
    evaluateTestData(model, X_test, y_test)