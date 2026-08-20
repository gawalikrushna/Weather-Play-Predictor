# Marvellous Infosystems Play Predictor
# Machine Learning Assignment - KNN

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


Border = "-" * 60


# ---------------------------------------------------------------
# Function Name : LoadData
# Description   : Load data from CSV file
# Input         : File name
# Output        : DataFrame
# ---------------------------------------------------------------

def LoadData(filename):

    df = pd.read_csv(filename)

    return df


# ---------------------------------------------------------------
# Function Name : DisplayData
# Description   : Display dataset information
# Input         : DataFrame
# Output        : None
# ---------------------------------------------------------------

def DisplayData(df):

    print(Border)
    print("Dataset Loaded Successfully")
    print(Border)

    print(df)

    print(Border)


# ---------------------------------------------------------------
# Function Name : PrepareData
# Description   : Convert categorical data into numerical data
# Input         : DataFrame
# Output        : X, Y
# ---------------------------------------------------------------

def PrepareData(df):

    WeatherEncoder = LabelEncoder()
    TemperatureEncoder = LabelEncoder()
    PlayEncoder = LabelEncoder()

    # Convert Weather
    df["Wether"] = WeatherEncoder.fit_transform(
        df["Wether"]
    )

    # Convert Temperature
    df["Temperature"] = TemperatureEncoder.fit_transform(
        df["Temperature"]
    )

    # Convert Play
    df["Play"] = PlayEncoder.fit_transform(
        df["Play"]
    )

    # Features
    X = df[["Wether", "Temperature"]]

    # Target
    Y = df["Play"]

    return X, Y, WeatherEncoder, TemperatureEncoder, PlayEncoder


# ---------------------------------------------------------------
# Function Name : TrainModel
# Description   : Train KNN model
# Input         : X, Y, K
# Output        : Trained model
# ---------------------------------------------------------------

def TrainModel(X, Y, K):

    model = KNeighborsClassifier(n_neighbors=K)

    model.fit(X, Y)

    return model


# ---------------------------------------------------------------
# Function Name : PredictPlay
# Description   : Predict Play result for new data
# Input         : Model, Weather, Temperature
# Output        : Prediction
# ---------------------------------------------------------------

def PredictPlay(model, WetherEncoder, TemperatureEncoder,
                PlayEncoder, wether, temperature):

    # Convert input into numerical values

    weather_encoded = WetherEncoder.transform(
        [wether]
    )[0]

    temperature_encoded = TemperatureEncoder.transform(
        [temperature]
    )[0]

    # Create new input
    new_data = pd.DataFrame(
        [[weather_encoded, temperature_encoded]],
        columns=["Wether", "Temperature"]
    )

    # Prediction
    prediction = model.predict(new_data)

    # Convert numerical result back to original label
    result = PlayEncoder.inverse_transform(prediction)

    return result[0]


# ---------------------------------------------------------------
# Function Name : Accuracy
# Description   : Calculate accuracy for given K
# Input         : X, Y, K
# Output        : Accuracy
# ---------------------------------------------------------------

def Accuracy(X, Y, K):

    model = KNeighborsClassifier(
        n_neighbors=K
    )

    model.fit(X, Y)

    prediction = model.predict(X)

    accuracy = accuracy_score(
        Y,
        prediction
    )

    return accuracy


# ---------------------------------------------------------------
# Function Name : DisplayAccuracy
# Description   : Display accuracy for different K values
# Input         : X, Y
# Output        : None
# ---------------------------------------------------------------

def DisplayAccuracy(X, Y):

    print()
    print(Border)
    print("Accuracy for Different Values of K") 
    print(Border)

    for K in range(1, 10):

        accuracy = Accuracy(X, Y, K)

        print("K =", K,"Accuracy =", round(accuracy * 100, 2), "%")


# ---------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------
def main():

    # Step 1 : Get Data

    filename = "MarvellousInfosystems_PlayPredictor.csv"

    df = LoadData(filename)

    DisplayData(df)


    # Step 2 : Clean, Prepare and Manipulate Data

    X, Y, WeatherEncoder, TemperatureEncoder, PlayEncoder = \
        PrepareData(df)


    print("Features:")
    print(X)

    print()
    print("Target:")
    print(Y)


    # Step 3 : Train Data

    K = 3

    model = TrainModel(X,Y,K)

    print()
    print(Border)
    print("KNN Model Trained Successfully....")
    print("Value of K =", K)
    print(Border)


    # Step 4 : Test Data

    print()
    print("Enter Weather")
    print("Available :", list(WeatherEncoder.classes_))

    weather = input("Enter Weather: ")

    print()
    print("Enter Temperature")
    print("Available :",list(TemperatureEncoder.classes_))

    temperature = input("Enter Temperature: ")


    # Prediction

    result = PredictPlay(
        model,
        WeatherEncoder,
        TemperatureEncoder,
        PlayEncoder,
        weather,
        temperature
    )


    print()
    print(Border)
    print("Prediction Result")
    print(Border)

    print("Weather      :", weather)

    print("Temperature  :", temperature)

    print("Predicted Play:", result)

    print(Border)


    # Step 5 : Calculate Accuracy

    DisplayAccuracy(X,Y)


# ---------------------------------------------------------------
# Program Execution
# ---------------------------------------------------------------

if __name__ == "__main__":
    main()