import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from joblib import load

# Caricamento del modello e dello scaler
rfm: RandomForestClassifier = load('modelRF.joblib')
min_max_scaler: MinMaxScaler = load("minmax_scaler.joblib")

while True:
    # Input per costruire il dataset per la predizione
    Gender = int(input("Gender(1/0): "))
    Age = float(input("Age: "))
    Height = float(input("Height: "))
    Weight = float(input("Weight: "))
    family_history_with_overweight = int(input("family_history(1/0): "))
    FAVC = int(input("FAVC(1/0): "))
    FCVC = int(input("FCVC(1-3): "))
    NCP = int(input("NCP(1-4): "))
    SMOKE = int(input("SMOKE(1/0): "))
    CH2O = int(input("CH2O(1-3): "))
    FAF = int(input("FAF(0-3): "))
    TUE = int(input("TUE(0-2): "))
    CALC = int(input("CALC(0-3): "))
    MTRANS = int(input("MTRANS(0-4): "))

    # Creazione del dataset
    dataset = pd.DataFrame({
        "Gender": [Gender],
        "Age": [Age],
        "Height": [Height],
        "Weight": [Weight],
        "family_history_with_overweight": [family_history_with_overweight],
        "FAVC": [FAVC],
        "FCVC": [FCVC],
        "NCP": [NCP],
        "SMOKE": [SMOKE],
        "CH2O": [CH2O],
        "FAF": [FAF],
        "TUE": [TUE],
        "CALC": [CALC],
        "MTRANS": [MTRANS]
    })

    # Normalizzazione di Age e Weight
    dataset[['Age', 'Weight']] = min_max_scaler.transform(dataset[['Age', 'Weight']])

    predicted_class = rfm.predict(dataset)

    # Predizione
    if predicted_class[0] == 0:
        print('Your condition is: Insufficient Weight')
    elif predicted_class[0] == 1:
        print('Your condition is: Normal Weight')
    elif predicted_class[0] == 2:
        print('Your condition is: Overweight Weight Level 1')
    elif predicted_class[0] == 3:
        print('Your condition is: Overweight Weight Level 2')
    elif predicted_class[0] == 4:
        print('Your condition is: Obesity Level 1')
    elif predicted_class[0] == 5:
        print('Your condition is: Obesity Level 2')
    elif predicted_class[0] == 6:
        print('Your condition is: Obesity Level 3')
