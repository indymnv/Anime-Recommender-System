import argparse
import os
import time

import config
import model_dispatcher

import joblib
import pandas as pd
from sklearn import metrics
#from sklearn import tree

def run(fold, model):

    #Read training data with folds
    df=pd.read_csv(config.TRAINING_FILE)
    
    
    #Training data is where kfold is not equal to given kfold
    df_train = df[df.kfold != fold].reset_index(drop=True)

    #Define validation with fold given
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    #Drop the label column and store y Y. Convert to numpy array
    x_train = df_train.drop(config.LABEL,axis=1).values
    y_train = df_train[config.LABEL].values

    #Same with validation dataset
    x_valid = df_valid.drop(config.LABEL,axis=1).values
    y_valid = df_valid[config.LABEL].values

    #Initialize the model given in function
    clf = model_dispatcher.models[model]
    
    start = time.time()
    #Fit model in train
    clf.fit(x_train, y_train)

    end = time.time()
    training_time = round(end-start)
    #Prediction in Validation
    preds = clf.predict(x_valid)

    #get accuracy with MAE and print it
    accuracy = metrics.mean_absolute_error(y_valid, preds)
    print(f"Fold = {fold}, Accuracy={accuracy}, Training time = {training_time}")
    


    #save the model
    joblib.dump(
        clf,
        os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin")
    )


if __name__== "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        type=int
    )
    parser.add_argument(
        "--model",
        type=str
    )

    args = parser.parse_args()

    run(
        fold=args.fold,
        model=args.model
    )
    
