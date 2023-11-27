import pickle 
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score

print()
print('Process Started .....!!!')
print()

df = pd.read_csv('RFS_Cleaned_Dataset.csv')

# Split data 
x = df.drop('Weekly_Sales',axis=1)

y = df['Weekly_Sales']

print('Data Split Process Done ....')

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2)

print()
print('Model Training Process Starts...')
model = ExtraTreesRegressor().fit(xtrain,ytrain)
print()
print('Model Trained ...')

y_test_pred  = model.predict(xtest)

r2_testing = r2_score(ytest,y_test_pred)

acc = r2_testing*100

print()
print('Inferencing Starts...')

def process():
    print()
    print('Calling Function for Prediction...')
    print()
    print('Provide the values for inferencing...')
    print()

    store = float(input('Enter Store Number :'))
    dept = float(input('Enter Department Number :'))
    isholiday = float(input('Enter isholiday :'))
    temperature = float(input('Enter Temperature:'))
    fuel_price = float(input('Enter Fuel Price :'))
    day = float(input('Enter The Day:'))
    Month = float(input('Enter The Month:'))
    Year = float(input('Enter The Year:'))
    Type = float(input('Enter The Type:'))
    Size = float(input('Enter The Size:'))
    MarkDown1 = float(input('Enter The MarkDown1:'))
    MarkDown2 = float(input('Enter The MarkDown2:'))
    MarkDown3 = float(input('Enter The MarkDown3:'))
    MarkDown4 = float(input('Enter The MarkDown4:'))
    MarkDown5 = float(input('Enter The MarkDown5:'))
    cpi= float(input('Enter The CPI:'))
    Unemployment = float(input('Enter The Unemployment:'))

    print('Model Predicting the response variable value...')

    x = model.predict([[store, dept, isholiday, temperature, fuel_price, day,
        Month, Year, Type, Size, MarkDown1, MarkDown2, MarkDown3,
        MarkDown4, MarkDown5, cpi, Unemployment]])
    
    return x[0]

while True:   
    value = process()
    print()
    print()
    print('Predicted value :',value)
    print()
    Input = input('Type "stop" to terminate the execution :')
    if Input == 'stop' or 'Stop':
        break

