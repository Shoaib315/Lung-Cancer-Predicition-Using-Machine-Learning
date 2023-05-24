#Importing the Libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('survey lung cancer.csv')

df['GENDER'] = df['GENDER'].replace('M', 0)
df['GENDER'] = df['GENDER'].replace('F', 1)

df['LUNG_CANCER'] = df['LUNG_CANCER'].replace('NO', 0)
df['LUNG_CANCER'] = df['LUNG_CANCER'].replace('YES', 1)

X = df.drop(columns = 'LUNG_CANCER',axis=1)
Y = df['LUNG_CANCER']


scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

X = standardized_data

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)


classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, Y_train)
  
  
      factors = {
        'Smoke': 'Do you smoke? (yes/no) ',
        'Yellow Fingers': 'Do you have yellow fingers? (yes/no) ',
        'Anxiety': 'Do you experience anxiety? (yes/no) ',
        'Peer Pressure': 'Do you feel peer pressure? (yes/no) ',
        'Chronic Disease': 'Do you have any chronic diseases? (yes/no) ',
        'Fatigue': 'Do you experience fatigue? (yes/no) ',
        'Allergy': 'Do you have any allergies? (yes/no) ',
        'Wheezing': 'Do you experience wheezing? (yes/no) ',
        'Alcohol': 'Do you consume alcohol? (yes/no) ',
        'Cough': 'Do you have a cough? (yes/no) ',
        'Shortness of Breath': 'Do you experience shortness of breath? (yes/no) ',
        'Swallowing': 'Do you have any difficulties with swallowing? (yes/no) ',
        'Chest': 'Do you have any chest-related issues? (yes/no) '
    }

    input_data = []
    G = input("What is your Gender? (Male/Female) ")
    A = input("What is your Age? ")

    input_data.append(0 if G.lower() == 'male' else 1)
    input_data.append(int(A))

    for factor, question in factors.items():
        response = input(question)
        input_data.append(1 if response.lower() == 'yes' else 0)

    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    std_data = scaler.transform(input_data)
    prediction = classifier.predict(std_data)
    
    if prediction == 0:
        print('The symptoms show that you do not have lung cancer, but make sure to visit the doctor.')
    else:
        print('The symptoms indicate the possibility of lung cancer. Please consult a doctor for further evaluation.')
