

import numpy as np 
from numpy import *
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import pandas as pd
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import optimizers
from sklearn import preprocessing

data = pd.read_csv("new.csv") #kaggle dataset
data.fillna(data.mean(), inplace=True)

#append each column of data frame to a list
list_values_of_features= []

feature_name_list = pd.read_csv("new.csv", nrows=1).columns.tolist()


for i in feature_name_list:
    list_values_of_features.append(data[i].values.tolist())

def code_value(a): #replace the string values features with 0,1,2
    extracted_unique_value = data[a].unique() #get unique value within column and put them into list
    
    code_list = list(range(0, len(extracted_unique_value))) #create the list from 0 to n E.g: if feature has 5 unique values => 0,1,2,3,4,5
    
    replace = data[a].replace(to_replace = extracted_unique_value, value = code_list) #replace unique value with 0,1,2,etc. 
    
    final_list = replace.tolist() #export them to a list
    
    return final_list

new_arr = []
count = 0
for i in data:
    if is_string_dtype(data[i]):
        new_arr.append(code_value(i)) #run code_value function and append the entire column to the new_arr
        print(i, "done")
        count += 1
    else:
        new_arr.append(list_values_of_features[count]) #append the entire column to new_arr
        count += 1

to_np_array = np.array(new_arr)
temp_arr = to_np_array[2:78] #Exclude first 2 feature (id) and the price features

X_train = temp_arr.T.astype(np.float32)
 

#Replace NAN value in column with average  
load_y_train = np.array(data.iloc[:,81:82]).astype(np.float32)
y_train = load_y_train.flatten()

print(X_train.shape)
print(y_train.shape)

model = Sequential([
    BatchNormalization(),
    Dense(76, activation = 'relu', name = 'first_layer'),
    BatchNormalization(),
    Dense(60, activation = 'relu', name = 'second_layer'),
    BatchNormalization(),
    Dense(50, activation = 'relu', name = 'fourth_layer'),   
    Dense(1, activation = 'linear', name = 'last_layer')
])

model.compile(
    loss=tf.keras.losses.MeanSquaredError(),  
    optimizer=tf.keras.optimizers.Adam(0.4), 
    metrics = ['mean_squared_error']
)

model.fit(
    X_train,y_train,
    epochs= 5
)

#import fomarted and cleaned testing set
df_test = pd.read_csv("testset.csv")
x_test = df.iloc[:,[0,75]]
y_test = df.iloc[:,[75]]
results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)


print("Generate predictions for 10 samples")
predictions = model.predict(x_test[:10])
print("predictions shape:", predictions.shape)
