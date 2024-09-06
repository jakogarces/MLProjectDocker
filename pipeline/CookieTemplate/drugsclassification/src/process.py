"""
This is the demo code that uses hydra to access the parameters in under the directory config.

Author: Khuyen Tran
"""

import hydra
from omegaconf import DictConfig

import pandas as pd
from hydra.utils import to_absolute_path as abspath
from patsy import dmatrices
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

def get_data(raw_path: str):
    data = pd.read_csv(raw_path)
    return data

# def get_features(target: str, features: list, data: pd.DataFrame):
#     feature_str = " + ".join(features)
#     y, X = dmatrices(
#         f"{target} ~ {feature_str} - 1", data=data, return_type="dataframe"
#     )
#     return y, X

# def rename_columns(X: pd.DataFrame):
#     X.columns = X.columns.str.replace(r'\[', "_", regex=True).str.replace(
#         r'\]', "", regex=True
#     )
#     return X

@hydra.main(config_path="../config", config_name="main", version_base="1.2")

#def processTestEntry(rowToProcess)
    

def process_data(config: DictConfig):
    """Function to process the data"""

    data = get_data(abspath(config.raw.path)) #Pasamos de CVS a Dataframe
    #data = get_data(abspath("../data/raw/Drug.csv")) #Pasamos de CVS a Dataframe
    #y, X = get_features(config.process.target, config.process.features, data)
    
    #X = rename_columns(X) #Cambiamos de nombre a las columnas

    #X_train, X_test, y_train, y_test = train_test_split(
    #    X, y, test_size=0.2, random_state=7
    #)

    # Defining the feature set and the target variable
    X = data.drop('Drug', axis=1)
    y = data['Drug']

    # Identifying numerical and categorical features
    numerical_features = ['Age', 'Na', 'K']
    categorical_features = ['Sex']
    ordinal_features = ['BP', 'Cholesterol']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        # Defining a column transformer with one-hot encoding for categorical features and scaling for numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features),
            ('ord', OrdinalEncoder(), ordinal_features)
        ])

    # Fitting the transformer on the training data and transforming both the training and test data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Guardamos los Datos de Entrenamiento y Prueba
    pd.DataFrame(X_train_processed).to_csv(abspath(config.processed.X_train.path), index=False)
    pd.DataFrame(X_test_processed).to_csv(abspath(config.processed.X_test.path), index=False)
    y_train.to_csv(abspath(config.processed.y_train.path), index=False)
    y_test.to_csv(abspath(config.processed.y_test.path), index=False)
    print("Preprocessing success")
    #print(y)

    X_test2= [51,'M','HIGH','NORMAL',0.832467,0.073392,'drugB']

if __name__ == "__main__":
    process_data()
