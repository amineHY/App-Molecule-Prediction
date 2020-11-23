import os
import argparse
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import servier.src.config as config
import servier.src.feature_extractor as fe


def LoadAndProcessData(path=config.path_single):

    print('[INFO] Load the data and extract features')
    isinstance(path, str)

    # load the csv file
    df = pd.read_csv(path)

    # extract feature
    df['smiles_features'] = df.smiles.apply(
        lambda x: np.array(fe.fingerprint_features(x)))

    # merge the extracted features to the dataframe
    df_feat = pd.DataFrame()
    for i in range(len(df.smiles_features)):
        df_feat['fe_'+str(i)] = df.smiles_features[i]
    df_feat = df_feat.T.reset_index(drop=True)

    # [fe_0, fe_2, ... fe_2047]
    df_feat.columns = ['fe_'+str(i) for i in range(len(df.smiles_features[0]))]

    X_all = pd.concat([df, df_feat], axis=1)
    X_all.drop(columns=['mol_id', 'smiles_features', 'smiles'], inplace=True)

    y = X_all['P1']
    X = X_all.drop(columns=['P1'], axis=1)
    return X, y


def SplitDataset(X, y):
    """
    Randomly Split the dataset into training and validation sets: 80%-20%
    """
    print('[INFO] Split the dataset: train - validation sets')
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=.2, random_state=0)
    return X_train, X_valid, y_train, y_valid


def SaveModelToDisk(model, path=config.MODEL_OUTPUT):
    """
    Save the pretrained machine learning model to disk
    """
    joblib_file = os.path.join(path)
    joblib.dump(model, joblib_file)


def LoadModelFromDisk(path):
    """
    Load a pretrained model from a (.bin) file
    """
    try:
        return joblib.load(path)
    except:
        raise ValueError('The file', path, 'can not be found')


def create_folds_stratefied(path_dataset):
    # Training data is in a CSV file called train.csv
    df = pd.read_csv(path_dataset)

    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1

    # the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.target.values

    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)

    # fill the new kfold column
    for fold, (trn_, val_) in enumerate(kf.split(X=df, y=y)):
        df.loc[val_, 'kfold'] = fold

    # save the new csv with kfold column
    df.to_csv(path_dataset[:-4]+'_folds.csv', index=False)


def Train(data_path=config.MODEL_OUTPUT['single']):
    """
    Train a machine learning model to perform prediction of basic preperties of a molecule from its fingerprint features
    Input: path to csv file of the dataset
    """

    isinstance(data_path, str)
    # Load and split  dataset
    X, y = LoadAndProcessData(data_path)
    X_train, X_valid, y_train, y_valid = SplitDataset(X, y)

    print('[INFO] Train a regression model')

    tic = time.time()

    # Defining the model
    model = RandomForestClassifier(
        n_estimators=config.n_est, max_depth=config.max_depth, n_jobs=-1, random_state=0)

    # Training the model
    model.fit(X_train, y_train)

    toc = time.time()
    print('Time:{0:.3f} seconds'.format(toc-tic))

    # Save the prediction model to disk
    n = X_train.shape[1]
    if n == 2048:
        SaveModelToDisk(model, config.MODEL_OUTPUT['single'])
    elif n == 2056:
        SaveModelToDisk(model, config.MODEL_OUTPUT['multi'])
    else:
        raise ValueError('Please make sure have the correct dataset')

    return X_valid, y_valid


def Predict(X_test=None):
    """
    Perform prediction of basic preperties of a molecule using a pretrained regresion model

    """
    if isinstance(X_test, str):
        X_test, y_test = LoadAndProcessData(X_test)
    elif isinstance(X_test, pd.DataFrame) or isinstance(X_test, np.array):
        pass
    else:
        raise ValueError(
            'Please specify data either by entering an array or a path to csv file')

    # Load pretrained model
    n = X_test.shape[1]
    if n == 2048:
        model = LoadModelFromDisk(config.MODEL_OUTPUT['single'])
    elif n == 2056:
        model = LoadModelFromDisk(config.MODEL_OUTPUT['multi'])
    else:
        raise ValueError('Please make sure you have a pretrained model')

    print('[INFO] Prediction')
    tic = time.time()
    y_pred = model.predict(X_test)
    toc = time.time()
    print('Predicted values:', y_pred)
    print('Time:{0:.3f} seconds'.format(toc-tic))
    return y_pred


def Evaluate(y_true, y_pred):
    '''
    Evaluate the prediction model by measuring accuracy
    - Accuracy
    '''

    print('[INFO] Model Evaluation')
    tic = time.time()
    accuracy = metrics.accuracy_score(y_true, y_pred)
    toc = time.time()
    print('Time:{0:.3f} seconds'.format(toc-tic))

    # Display metrics
    print('\tAccuracy={0:2.3f}%'.format(accuracy*100))
    return accuracy


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str,
                        default=config.path_single, help='Specify the path of the dataset')

    args = parser.parse_args()
    data_path = args.data

    print(80*'*')
    print('This application uses machine learning to predict basic properties of a molecule')
    print(80*'*')
    print('[INFO] Start processing ...')

    # Train machine learning model (and save pretrained model to disk)
    X_valid, y_valid = Train(data_path)

    # prediction
    y_pred = Predict(X_valid)

    # evaluation
    Evaluate(y_valid, y_pred)

    print('[INFO] DONE...')


if __name__ == "__main__":
    main()
