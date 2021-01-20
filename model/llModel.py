from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2

def llNet(inputshape=(200,), regularizers=0.001):
    inputsignal = Input(inputshape)

    outputsignal = Dense(256)(inputsignal)
    outputsignal = Activation('relu')(outputsignal)
    outputsignal = Dropout(0.25)(outputsignal)

    outputsignal = Dense(128)(outputsignal)
    outputsignal = Activation('relu')(outputsignal)
    outputsignal = Dropout(0.5)(outputsignal)

    outputsignal = Dense(128)(outputsignal)
    outputsignal = Activation('relu')(outputsignal)
    # outputsignal = Dropout(0.5)(outputsignal)

    outputsignal = Dense(6)(outputsignal)
    outputsignal = Activation('softmax')(outputsignal)

    model = Model(inputsignal, outputsignal)

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    return model

if __name__ == "__main__":
    model = llNet()
    model.summary()


