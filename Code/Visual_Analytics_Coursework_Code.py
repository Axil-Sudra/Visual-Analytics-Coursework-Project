# Python libraries
import pandas_datareader as pdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime 
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.manifold import MDS
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error 
import warnings
import time
from statsmodels.tsa.stattools import acf, pacf

# DJIA30 equity names
DJIA30_Equities_Name = ['MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX',
                   'CSCO', 'KO', 'DIS', 'DWDP', 'XOM', 'GS', 
                   'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 
                   'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'TRV', 
                   'UTX', 'UNH', 'VZ', 'V', 'WMT', 'WBA']

# Function to retrieve equities data from Yahoo Finance API
def Equity_Data(Equity_Name):
     return pdr.get_data_yahoo(Equity_Name, 
               start = datetime.datetime(2015, 1, 1), 
               end = datetime.datetime(2017, 12, 31))
     
# Store data to dataframe
for Equity in DJIA30_Equities_Name:
     globals()[Equity] = Equity_Data(Equity)
del Equity

# Store DJIA30 index data to dataframe
DJIA30_Index = Equity_Data('^DJI')

DJIA30_Equities = [MMM, AXP, AAPL, BA, CAT, CVX,
                   CSCO, KO, DIS, DWDP, XOM, GS, 
                   HD, IBM, INTC, JNJ, JPM, MCD, 
                   MRK, MSFT, NKE, PFE, PG, TRV, 
                   UTX, UNH, VZ, V, WMT, WBA]

# Insert name column to each equity dataframe 
i = 0
for Equity in DJIA30_Equities:
     Equity['Ticker'] = DJIA30_Equities_Name[i]
     i = i + 1
del i, Equity
     
# Function to check equity dataframes for missing values
def Missing_Values(Equity_Data, Equity_Name):
     if Equity_Data.isnull().sum().sum() > 0:
          print(str(Equity_Name), 'dataframe has ', 
                Equity_Data.isnull().sum().sum(), 'missing values')
     else:
          print(str(Equity_Name), 'dataframe has no missing values')

# Check equity dataframes for missing values
i = 0
for Equity in DJIA30_Equities:
     Missing_Values(Equity, DJIA30_Equities_Name[i])
     i = i + 1
del i, Equity 

# Convert index to 'datetime' format
for Equity in DJIA30_Equities:
     pd.to_datetime(Equity.index)
del Equity

# Convert index of DJIA30 index to 'datetime' format
pd.to_datetime(DJIA30_Index.index)

# Pivot table for adjusted closing prices of equities
DJIA30_Equities_Table = pd.concat(DJIA30_Equities)
DJIA30_Equities_PTable = DJIA30_Equities_Table[['Adj Close', 'Ticker']]
DJIA30_Equities_PTable = DJIA30_Equities_PTable.pivot_table(
          index = DJIA30_Equities_Table.index, columns = 'Ticker',
          aggfunc = sum)
DJIA30_Equities_PTable.columns = DJIA30_Equities_PTable.columns.droplevel()
DJIA30_Equities_PTable.columns.name = None
DJIA30_Equities_PTable.index.name = None

# Save equities pivot table to csv file to plot plotly graphs
DJIA30_Equities_PTable.to_csv('DJIA30_Equities_Adj_Close.csv')

# Save DJIA30 index dataframe to csv file to plot plotly graph
DJIA30_Index.to_csv('DJIA30_Index.csv')

## Calculate daily returns of equities in pivot table 
DJIA30_Equities_PTable_DReturns = DJIA30_Equities_PTable.pct_change()
DJIA30_Equities_PTable_DReturns.fillna(0, inplace = True)

Correlation = DJIA30_Equities_PTable_DReturns.corr()

# Calculate daily returns and daily volatility of equities
def Daily_Returns(Equity_Data):
     Equity_Data['Daily Returns'] = Equity_Data['Adj Close'].pct_change()
     Equity_Data['Daily Returns'].fillna(0, inplace = True)

for Equity in DJIA30_Equities:
     Daily_Returns(Equity)
del Equity

# Save equities dataframes to csv files
i = 0
for Equity in DJIA30_Equities:
     Name = DJIA30_Equities_Name[i] + '.csv'
     Equity.to_csv(Name)
     i = i + 1
del i, Equity, Name

DJIA30_Equities_Info_Names = ['MMM_Info', 'AXP_Info', 'AAPL_Info', 'BA_Info', 'CAT_Info', 
                        'CVX_Info', 'CSCO_Info', 'KO_Info', 'DIS_Info', 'DWDP_Info', 
                        'XOM_Info', 'GS_Info', 'HD_Info', 'IBM_Info', 'INTC_Info', 
                        'JNJ_Info', 'JPM_Info', 'MCD_Info', 'MRK_Info', 'MSFT_Info', 
                        'NKE_Info', 'PFE_Info', 'PG_Info', 'TRV_Info', 'UTX_Info', 
                        'UNH_Info', 'VZ_Info', 'V_Info', 'WMT_Info', 'WBA_Info']

# Calculate annual return, annual volatility, average adj close price and average volume
def Average_Features(Equity_Data):
     Average_Annual_Return = ((1 + Equity_Data['Daily Returns'].mean()) ** 252) - 1
     Average_Annual_Volatility = Equity_Data['Daily Returns'].std() * np.sqrt(252)
     Average_Adj_Close_Price = np.log(Equity_Data['Adj Close'].mean())
     Average_Volume = np.log(np.round(Equity_Data['Volume'].mean(), 0))
     return [Average_Annual_Return, Average_Annual_Volatility, Average_Adj_Close_Price, Average_Volume]

i = 0
for Equity in DJIA30_Equities:
     globals()[DJIA30_Equities_Info_Names[i]] = Average_Features(Equity) 
     i = i + 1
del i, Equity

DJIA30_Equities_Info = [MMM_Info, AXP_Info, AAPL_Info, BA_Info, CAT_Info, 
                        CVX_Info, CSCO_Info, KO_Info, DIS_Info, DWDP_Info, 
                        XOM_Info, GS_Info, HD_Info, IBM_Info, INTC_Info, 
                        JNJ_Info, JPM_Info, MCD_Info, MRK_Info, MSFT_Info, 
                        NKE_Info, PFE_Info, PG_Info, TRV_Info, UTX_Info, 
                        UNH_Info, VZ_Info, V_Info, WMT_Info, WBA_Info]

Measurement_Features = []
for Equity in DJIA30_Equities_Info:
     Measurement_Features.append(Equity)
del Equity 

Measurement_Features_PD = pd.DataFrame(np.array(Measurement_Features).reshape(30, 4))    
Measurement_Features_PD.rename(columns = {0 : 'Average Annual Returns', 
                                          1 : 'Average Annual Volatility', 
                                          2 : 'Average Adj Close Price', 
                                          3 : 'Average Volume'}, inplace = True)
     
Equitiy_Characteristics = Measurement_Features_PD.copy()
     
# Compute euclidean distances between equities for the measurement of features
Similarities_Euclidean = euclidean_distances(Measurement_Features_PD)
Similarities_Euclidean_PD = pd.DataFrame(Similarities_Euclidean)
Similarities_Euclidean_PD.rename(columns = {0 : 'MMM', 1 : 'AXP', 2 : 'AAPL', 3 : 'BA', 
                                            4 : 'CAT', 5 : 'CVX', 6 : 'CSCO', 7 : 'KO', 
                                            8 : 'DIS', 9 : 'DWDP', 10 : 'XOM', 11 : 'GS', 
                                            12 : 'HD', 13 : 'IBM', 14 : 'INTC', 15 : 'JNJ', 
                                            16 : 'JPM', 17 : 'MCD', 18 : 'MRK', 19 : 'MSFT',
                                            20 : 'NKE', 21 : 'PFE', 22 : 'PG', 23 : 'TRV', 
                                            24 : 'UTX', 25 : 'UNH', 26 : 'VZ', 27 : 'V', 
                                            28 : 'WMT', 29 : 'WBA'}, inplace = True)
Similarities_Euclidean_PD['Equity'] = DJIA30_Equities_Name
Similarities_Euclidean_PD.set_index('Equity', inplace = True)

# Save euclidean distance matrix to csv file to plot in plotly
Similarities_Euclidean_PD.to_csv('Similarities_Euclidean_Distances.csv')

# MDS 3D model computation 
Model_3D = MDS(n_components  = 3, dissimilarity = 'precomputed', random_state = 1)
Coordinates_3D = Model_3D.fit_transform(Similarities_Euclidean)
Coordinates_3D_PD = pd.DataFrame(Coordinates_3D)
Coordinates_3D_PD['Equity'] = DJIA30_Equities_Name
Coordinates_3D_PD.set_index('Equity', inplace = True)

# Save MDS coordinates to csv file to plot in plotly
Coordinates_3D_PD.to_csv('DJIA30_Equities_MDS_3D_Euclidean_Coordinates.csv')

# Compute manhattan distances between equities for an alternative 
Similarities_Manhattan = manhattan_distances(Measurement_Features_PD)
Similarities_Manhattan_PD = pd.DataFrame(Similarities_Manhattan)
Similarities_Manhattan_PD.rename(columns = {0 : 'MMM', 1 : 'AXP', 2 : 'AAPL', 3 : 'BA', 
                                            4 : 'CAT', 5 : 'CVX', 6 : 'CSCO', 7 : 'KO', 
                                            8 : 'DIS', 9 : 'DWDP', 10 : 'XOM', 11 : 'GS', 
                                            12 : 'HD', 13 : 'IBM', 14 : 'INTC', 15 : 'JNJ', 
                                            16 : 'JPM', 17 : 'MCD', 18 : 'MRK', 19 : 'MSFT',
                                            20 : 'NKE', 21 : 'PFE', 22 : 'PG', 23 : 'TRV', 
                                            24 : 'UTX', 25 : 'UNH', 26 : 'VZ', 27 : 'V', 
                                            28 : 'WMT', 29 : 'WBA'}, inplace = True)
Similarities_Manhattan_PD['Equity'] = DJIA30_Equities_Name
Similarities_Manhattan_PD.set_index('Equity', inplace = True)

# Save manhattan distance matrix to csv file to plot in plotly
Similarities_Manhattan_PD.to_csv('Similarities_Manhattan_Distances.csv')

# MDS 3D model computation 
Model_3D_Manhattan = MDS(n_components  = 3, dissimilarity = 'precomputed', random_state = 1)
Coordinates_3D_Manhattan = Model_3D_Manhattan.fit_transform(Similarities_Manhattan)
Coordinates_3D_PD_Manhattan = pd.DataFrame(Coordinates_3D_Manhattan)
Coordinates_3D_PD_Manhattan['Equity'] = DJIA30_Equities_Name
Coordinates_3D_PD_Manhattan.set_index('Equity', inplace = True)

# Save MDS coordinates to csv file to plot in plotly
Coordinates_3D_PD_Manhattan.to_csv('DJIA30_Equities_MDS_3D_Manhattan_Coordinates.csv')

# Equities characteristics for further EDA
Equity_Characteristics = Measurement_Features_PD.copy()
Equity_Characteristics['Average Annual Returns'] = Equity_Characteristics['Average Annual Returns'] * 100
Equity_Characteristics['Average Annual Volatility'] = Equity_Characteristics['Average Annual Volatility'] * 100
Equity_Characteristics['Equity'] = DJIA30_Equities_Name
Equity_Characteristics.set_index('Equity', inplace = True)

# Save equities characteristics to csv file to plot using plotly
Equity_Characteristics.to_csv('DJIA30_Equities_Characteristics.csv')


# ARIMA model construction and evaluation
# Visa equity is to be used in this forecasting experiment

def ARIMA_Model(Equity_Data, Parameter_Order):
     # Apply 70% / 30% split from dataframe 
     Split = int(len(Equity_Data) * 0.7)
     # Split dataframe into training and test
     Training_Data = Equity_Data[0 : Split]
     Test_Data = Equity_Data[Split : len(Equity_Data)]
     # Set training and test target lists
     Training_Target = [X for X in Training_Data]
     Predictions = []
     for i in range(len(Test_Data)):
          Model = ARIMA(Training_Target, order = Parameter_Order)
          Model_Fit = Model.fit(disp = 0)
          Model_Output = Model_Fit.forecast()[0]
          Y_Hat = Model_Output[0]
          Predictions.append(Y_Hat)
          Training_Target.append(Test_Data[i])
          print('Predicted ', Predictions[i], 'Observed ', Test_Data[i])
     # Calculate MSE 
     Error = mean_squared_error(Test_Data, Predictions)
     print('MSE =', Error)
     return Error

def Model_Evaluation(Data, P_Values, D_Values, Q_Values):
     Data = Data.astype('float32')
     Best_MSE = float('inf')
     Best_Parameters = None
     # Set lists to record p, d, q and MSE values
     MSE_List = []
     P_List = []
     D_List = []
     Q_List = []
     for p in P_Values:
          for d in D_Values:
               for q in Q_Values:
                    Order = (p, d, q)
                    try:
                         MSE = ARIMA_Model(Data, Order)
                         MSE_List.append(MSE)
                         P_List.append(p)
                         D_List.append(d)
                         Q_List.append(q)
                         if MSE < Best_MSE:
                              Best_MSE = MSE
                              Best_Parameters = Order
                    except:
                         continue
     print('Best ARIMA parameters' , str(Best_Parameters), ' MSE =', str(Best_MSE))
     Results = pd.DataFrame({'P Values' : P_List, 'D Values' : D_List, 
                             'Q Values' : Q_List, 'MSE' : MSE_List})
     Results.to_csv('ARIMA Grid Search Results.csv')

P_Values = range(0, 4)
D_Values = range(0, 4)
Q_Values = range(0, 4)
warnings.filterwarnings('ignore')
Start = time.time()
Model_Evaluation(V['Adj Close'], P_Values, D_Values, Q_Values)
End = time.time()
print('Execution Time:', End - Start)

# Run ARIMA with best hyperparameters from random search
 # Apply 70% / 30% split from dataframe 
     Split_Final = int(len(V['Adj Close']) * 0.7)
     # Split dataframe into training and test
     Training_Data = V['Adj Close'][0 : Split_Final]
     Test_Data = V['Adj Close'][Split_Final : len(V['Adj Close'])]
     # Set training and test target lists
     Training_Target = [X for X in Training_Data]
     Predictions = []
     for i in range(len(Test_Data)):
          Model = ARIMA(Training_Target, order = (2, 1, 1))
          Model_Fit = Model.fit(disp = 0)
          Model_Output = Model_Fit.forecast()[0]
          Y_Hat = Model_Output[0]
          Predictions.append(Y_Hat)
          Training_Target.append(Test_Data[i])
          print('Predicted ', Predictions[i], 'Observed ', Test_Data[i])
Predicted = {'Predict' : Predictions}
P_DF = pd.DataFrame(Predicted)
P_DF.index = Test_Data.index

O_DF = pd.Series.to_frame(Test_Data)
New = pd.DataFrame.join(O_DF, P_DF)          
Train_DF = pd.Series.to_frame(Training_Data)

New.to_csv('Visa Forecast.csv')
Train_DF.to_csv('Visa Training Set.csv')

Difference = New['Adj Close'] - New['Predict']
Difference.to_csv('Visa Forecast Difference.csv')

ACF_V = acf(V['Adj Close'])
PACF_V = pacf(V['Adj Close'])

ACF_V_PD = pd.DataFrame({'ACF' : ACF_V})
PACF_V_PD = pd.DataFrame({'PACF': PACF_V})
