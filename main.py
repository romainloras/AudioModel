import numpy
import matplotlib.pyplot as plt
import pandas as pd
import math
import librosa.display
import librosa
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pickle


def ping():
    print("Pong")
    return "Pong"


def loading_file(file_dir="audio1.wav"):
    data, sampling_rate = librosa.load(file_dir,duration=30,sr=8096)
    df = pd.DataFrame(data=data)
    export = df.to_csv("audio.csv")
    return [df, sampling_rate]


def creating_df_train(df, row_nbr=10, col_nbr=10):
    # Random position in df
    final = []
    final2 = []
    for i in range(row_nbr):
        pos = random.randint(2, int(len(df)-col_nbr))
        # Generation col
        this_row = []
        for y in range(col_nbr):
            this_row.append(df['value'][pos+y])
        final.append(this_row)
        final2.append(df['value'][pos-1])
    return [pd.DataFrame(final, columns=[i+1 for i in range(len(final[0]))]), pd.DataFrame(data=final2)]
    # return [final, final2]


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


def LSTM_func(file_name="audio.csv",loadmodel=False):
    numpy.random.seed(7)
    dataset = pd.read_csv(file_name,usecols=[1])
    print(dataset.head())
    dataset = dataset.values
    dataset = dataset.astype("float32")
    dataset = dataset[int(len(dataset)*0.1):int(len(dataset) * 0.5),]
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    print(len(train), len(test))
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    filenamemodel = "LSTMModel1.sav"
    if(loadmodel == False):
        model = Sequential()
        model.add(LSTM(4, input_shape=(1,look_back)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
        # make predictions
        pickle.dump(model,open(filenamemodel,"wb"))
    else:
        model = pickle.load(open(filenamemodel, 'rb'))
    #loaded_model = pickle.load(open(filename, 'rb'))
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))

    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(look_back*2) +
                    1:len(dataset)-1, :] = testPredict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()
    return 1


#plt.plot(df['value'])
#plt.show()
#loading_file()
LSTM_func(loadmodel=False)

#LOADING MDOEL : 
#loaded_model = pickle.load(open("LSTMModel1.sav", 'rb'))