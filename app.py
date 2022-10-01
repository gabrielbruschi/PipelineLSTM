from flask import Flask

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import StandardScaler #normalizar os dados
from sklearn.model_selection import train_test_split #treinamento e teste
import yfinance as yf

app = Flask(__name__)
# route -> hashtagtreinamentos.com/
# função -> o que você quer exibir naquela página
# template

@app.route('/')
def homepage():
    
    #codigo
    obj = yf.Ticker('petr4.sa')
    data = obj.history(start='2018-01-01')
    df = data.dropna()

    df_acao_fec = df[['Close']]

    #verificar a quantidade de linhas
    qtd_linhas = len(df_acao_fec)

    qtd_linhas_treino = round(.70 * qtd_linhas)
    qtd_linhas_teste = qtd_linhas - qtd_linhas_treino

    info = (
        f"linhas treino = 0{qtd_linhas_treino}"
        f"linhas teste = {qtd_linhas_treino}:{qtd_linhas_treino + qtd_linhas_teste}"
    )

    #Normalizar os dados
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_acao_fec)

    train = df_scaled[:qtd_linhas_treino]
    test = df_scaled[qtd_linhas_treino: qtd_linhas_treino + qtd_linhas_teste] 

    #Convert an array of values into a df matrix
    #Rede LSTM
    def create_df(df, steps=1):
        dataX, dataY = [], []
        for i in range(len(df) - steps - 1):
            a = df[i:(i+steps), 0]
            dataX.append(a)
            dataY.append(df[i + steps, 0])
        return np.array(dataX), np.array(dataY)

    #Gerando dados de treino e teste
    steps = 15 
    X_train, Y_train = create_df(train, steps)
    X_test, Y_test = create_df(test, steps)


     # Gerando os dados que o modelo espera
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1) #1 'e a qntd de features
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    #Montando a rede
    model = Sequential()
    model.add(LSTM(35, return_sequences=True, input_shape=(steps, 1))) #35 neuronios, o return_seq = true : pega a informacao q sai e reinsere, dados com memoria
    model.add(LSTM(35, return_sequences=True))
    model.add(LSTM(35))
    model.add(Dropout(0.05)) #ANTES ERA 0.2 #nao causar um overfit (rede muito treinada), dou uma penalizada na feature
    model.add(Dense(1)) #saida unica do preco que queremos prever

    model.compile(optimizer='adam', loss='mse') #adam mais usado, loss: minimun sequer error, ver o quanto minha rede esta performando
    model.summary()

    #Treinamento do modelo
    validation = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30, batch_size=15, verbose=2)

    #Fazendo previsao
    prev = model.predict(X_test)
    prev = scaler.inverse_transform(prev) #removo a normalizacao

    #previsão para os proximos 10 dias
    lenght_test = len(test)

    #pegar os ultimos dias que são o tamanho do meu step - 15 dias
    days_input_steps = lenght_test - steps

    #transforma em array
    input_steps = test[days_input_steps:]
    input_steps = np.array(input_steps).reshape(1,-1)

    #transformar em lista
    list_output_steps = list(input_steps)
    list_output_steps = list_output_steps[0].tolist()

    #loop para prever os proximos 10 dias

    pred_output = []
    i = 0
    n_future = 10

    while (i < n_future):
    
        if (len(list_output_steps) > steps):
            input_steps = np.array(list_output_steps[1:])
            #print("{} dia. Valores de entrada -> {}".format(i,input_steps))
            input_steps = input_steps.reshape(1,-1)
            input_steps = input_steps.reshape((1, steps, 1))
            #print(input_steps)
            pred = model.predict(input_steps, verbose = 0)
            #print("{} dia. Valor previsto -> {}".format(i, pred))
            list_output_steps.extend(pred[0].tolist())
            list_output_steps = list_output_steps[1:]
            #print(list_output_steps)
            pred_output.extend(pred.tolist())
            i = i + 1
        else:
            input_steps = input_steps.reshape((1, steps, 1))
            pred = model.predict(input_steps, verbose = 0)
            #print(pred[0])
            list_output_steps.extend(pred[0].tolist())
            #print(len(list_output_steps))
            pred_output.extend(pred.tolist())
            i = i + 1
    

    #transformar a saida
    prev = scaler.inverse_transform(pred_output)
    prev = np.array(prev).reshape(1, -1)
    list_output_prev = list(prev)
    list_output_prev = prev[0].tolist()

    df1 = df.reset_index()
    #pegar as datas de previsão 
    dates = pd.to_datetime(df1['Date'])
    predict_dates = pd.date_range(list(dates)[-1] + pd.DateOffset(1), periods = 10, freq='b').tolist() #freq b = dias bussins


    #cria dataframe de previsao
    forecast_dates = []
    for i in predict_dates:
        forecast_dates.append(i.date())
    
    df_forecast = pd.DataFrame({'data_pregao': np.array(forecast_dates), 'preco_fechamento': list_output_prev})
    df_forecast['data_pregao'] = pd.to_datetime(df_forecast['data_pregao'])

    df_forecast = df_forecast.set_index(pd.DatetimeIndex(df_forecast['data_pregao'].values))
    df_forecast.drop('data_pregao', axis = 1, inplace = True)
    ##fim codigo
    
    return df_forecast.to_html()

@app.route('/usuario')
def user():
    return '<h1>Welcome user<h1>'
