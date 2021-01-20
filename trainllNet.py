import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from model import llModel
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


def load_data(pathfile):
    train = pd.read_excel(pathfile, sheet_name="train")
    test = pd.read_excel(pathfile, sheet_name="test")

    train.drop(["Unnamed: 0"], axis=1, inplace=True)
    test.drop(["Unnamed: 0"], axis=1, inplace=True)

    trainX = train.iloc[:,:-1]
    trainY = train.iloc[:,-1]
    testX = test.iloc[:,:-1]
    testY = test.iloc[:,-1]

    assert len(trainX) == len(trainY), "len(trainX) should be equal len(trainY)"
    assert len(testX) == len(testY), "len(testX) should be equal len(testY)"
    assert len(trainX.columns) == len(testX.columns)

    return trainX, trainY, testX, testY

def preprocessing(trainX, trainY, testX, testY):

    ss = StandardScaler()
    lb = LabelEncoder()

    trainX = ss.fit_transform(trainX)
    testX = ss.transform(testX)
    trainY = to_categorical(lb.fit_transform(trainY))
    testY = to_categorical(lb.transform(testY))

    return trainX, trainY, testX, testY, ss, lb

def plot_history(history):
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    # Set figure size.
    plt.figure(figsize=(12, 8))

    # Generate line plot of training, testing loss over epochs.
    plt.plot(train_accuracy, label='Training Accuracy', color='#185fad')
    plt.plot(val_accuracy, label='Validation Accuracy', color='orange')

    # Set title
    plt.title('Training and Validation Accuracy by Epoch', fontsize = 25)
    plt.xlabel('Epoch', fontsize = 18)
    plt.ylabel('Categorical Crossentropy', fontsize = 18)
    plt.xticks(range(0,100,5), range(0,100,5))

    plt.legend(fontsize = 18)
    # plt.savefig("accuracy_global_feature.png")
    plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--filenamellFeature', required=True, type=str, help="Path of file data")

    args = vars(ap.parse_args())
    print(args)

    trainX, trainY, testX, testY = load_data(args["filenamellFeature"])
    trainX, trainY, testX, testY, _, _ = preprocessing(trainX, trainY, testX, testY)

    model = llModel.llNet()

    model.summary()

    # model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')

    history = model.fit(trainX, trainY, batch_size=256, epochs=100,\
                        validation_data=(testX, testY), shuffle=True, callbacks=[early_stop])
    model.save("./pretrained/llNet.h5")

    plot_history(history)
    