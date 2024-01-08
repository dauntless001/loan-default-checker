import numpy as np
import pandas as pd
import keras
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
import warnings
from collections import Counter
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from sklearn.metrics import *
# import gradio as gr



keras.utils.set_random_seed(42)
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',100)



DATA_CSV_PATH = "./Code Ocean.csv"

df_train = pd.read_csv(DATA_CSV_PATH)


df=df_train.copy()

df[df.duplicated()==True]

    
data = df.drop(['phi_term_month', 'phi_loan_amnt', 'phi_emp_length',
       'phi_annual_inc', 'phi_dti', 'phi_delinq_2yrs', 'phi_revol_util',
       'phi_total_acc', 'phi_credit_length_in_years', 'phi_int_rate','train_flag'],axis = 1)


#Now checking for outliers

def find_outliers(data,ftrs):
    """
    A python function to detect outliers. It takes in the dataframe data and checks if features ftrs have outliers
    """
    outlier_list = []

    # iterating over the features(ftr)
    for column in ftrs:

        # calculating interquartile range (Q3-Q1)
        Q1 = np.percentile(df[column], 25)
        Q3 = np.percentile(df[column],75)

        IQR = Q3 - Q1

        # outlier step
        outlier_step = 1.5 * IQR

        # Getting the indices of outliers
        outlier_indices_list = df[(df[column] < Q1 - outlier_step) | (df[column] > Q3 + outlier_step )].index.tolist()

        # adding the outlier_indices_list to the empty list
        outlier_list.extend(outlier_indices_list)

        # Checking for multiple outliers
        #multi_outliers = list( p for p, q in Counter(outlier_list).items() if q > n )

        return outlier_list


# calculating interquartile range (Q3-Q1)
Q1 = np.percentile(data['annual_inc'], 25)
Q3 = np.percentile(data['annual_inc'],75)

IQR = Q3 - Q1

# outlier step
outlier_step = 1.5 * IQR
data[(data['annual_inc'] < Q1 - outlier_step) | (data['annual_inc'] > Q3 + outlier_step )]


#checking columns with a '' value
for column in data.columns.tolist():
    #if (dff[column] == np.NaN):
    data[column] = data[column].replace(r'^\s*$',np.nan,regex = True)



#checking columns with a '' value
for column in data.columns.tolist():
    data[column] = data[column].replace(np.nan, data[column].mode()[0])


data[['emp_length','dti','delinq_2yrs','revol_util','credit_length_in_years','remain','CRI']] = data[['emp_length','dti','delinq_2yrs','revol_util','credit_length_in_years','remain','CRI']].astype('float64')


# label_encoder object knows
# how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'species'.
data['home_ownership']= label_encoder.fit_transform(data['home_ownership'])
data['purpose']= label_encoder.fit_transform(data['purpose'])
data['addr_state']= label_encoder.fit_transform(data['addr_state'])
data['verification_status']= label_encoder.fit_transform(data['verification_status'])
data['application_type']= label_encoder.fit_transform(data['application_type'])

data['home_ownership'].unique(),data['purpose'].unique(),data['addr_state'].unique(),data['verification_status'].unique(),data['application_type'].unique()


train = data

X = train.drop('default_loan',axis=1)#.values
y = train.default_loan


# Feature Scaling first to reduce training time
sc = preprocessing.StandardScaler()
X_scaled = sc.fit_transform(X)


#spliting the X into train and test sets
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,train_size=0.8,random_state=42,shuffle=True,stratify=y)


def create_mlp_model(input_dim):
    model = Sequential()
    model.add(Dense(units=128, input_dim=input_dim, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(units=64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(units=32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))#sigmoid and not relu should be the activation funtion of the output layer since we're expectioning 0s and 1s. ReLU is not bound to [0, 1] unlike Sigmoid
    return model

model = create_mlp_model(18)
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.1), metrics=['accuracy'])


# Apply SMOTE-ENN
smote_enn = SMOTEENN(random_state=42)
X_resampled1, y_resampled1 = smote_enn.fit_resample(X[:100000], y[:100000])
X_resampled2, y_resampled2 = smote_enn.fit_resample(X[100000:200000], y[100000:200000])
X_resampled3, y_resampled3 = smote_enn.fit_resample(X[200000:300000], y[200000:300000])


X_resampled = pd.concat([X_resampled1,X_resampled2,X_resampled3],ignore_index=True)
y_resampled = pd.concat([y_resampled1,y_resampled2,y_resampled3],ignore_index=True)


y_resampled_df=pd.DataFrame(y_resampled,columns=['default_loan'])
data_resampled = pd.concat([X_resampled, y_resampled_df],axis=1)

#spliting the X into train and test sets
X_train2,X_test2,y_train2,y_test2=train_test_split(X_resampled,y_resampled,train_size=0.8,random_state=42,shuffle=True,stratify=y_resampled)



# Start the timer
# WOA hyperparameters
max_iter = 10  # Maximum number of iterations
pop_size = 10   # Population size
c1 = 2.0        # Constant for the spiral updating formula
c2 = 2.0        # Constant for the whale updating formula


# Initialize the population of models
population = [create_mlp_model(18) for _ in range(pop_size)]

loss_list = []

# Define the WOA algorithm
def whales_optimization_algorithm(X_train, y_train, population, max_iter, c1, c2):
    for iteration in range(max_iter):
        for i, model in enumerate(population):
            # Train the model with Adam optimizer
            model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.1), metrics=['accuracy'])
            history_woa = model.fit(X_train2, y_train2, validation_split=0.2, batch_size=512, epochs=1, verbose=True)

            # Calculate fitness based on accuracy
            y_pred = model.predict(X_test2)
            accuracy = accuracy_score(y_test2, (y_pred > 0.5).astype(int))

            # Update the model weights and biases using WOA
            a = 2.0 - iteration * ((2.0) / max_iter)  # Linearly decrease a from 2 to 0
            A = 2.0 * a * np.random.uniform() - a
            C = 2.0 * np.random.uniform()

            D = np.abs(C * np.array(model.get_weights()) - np.array(model.get_weights()))  # Difference between current weight and random whale
            new_weights = np.array(model.get_weights()) - A * D

            # Update the population with the improved model
            population[i].set_weights(new_weights)

        # Print the loss at the end of each iteration
        loss = model.evaluate(X_train2, y_train2, verbose=True)[0]
        loss_list.append(loss)
        print(f"Iteration {iteration + 1}/{max_iter}, Loss: {loss}")

# Apply the WOA algorithm
whales_optimization_algorithm(X_train2, y_train2, population, max_iter, c1, c2)

# Evaluate the best model from the population
best_model = max(population, key=lambda model: accuracy_score(y_test2, (model.predict(X_test2) > 0.5).astype(int)))

# Evaluate the best model on various metrics
predictions = best_model.predict(X_test2)


def classify(num):
    if num<0.5:
        return 'Non-Defaulter'
    
    else:
        return 'Defaulter'
     
classify(1)
      
      
def predict_Loan_Status(term_months,home_ownership,purpose,addr_state, verification_status , application_type, loan_amnt,emp_length,annual_inc,    
          
     dti, delinq_2yrs, revol_util,total_acc,credit_length_in_years,int_rate,remain,issue_year, CRI                       
     ):
    input_array=np.array([[term_months,home_ownership,purpose,addr_state, verification_status , application_type, loan_amnt,emp_length,annual_inc,    
          
     dti, delinq_2yrs, revol_util,total_acc,credit_length_in_years,int_rate,remain,issue_year, CRI]])
    pred=best_model.predict(input_array)
    output=classify(pred[0])
    print(output)
    if output=='Non-Defaulter':
      return True
    
    else:
      return False


