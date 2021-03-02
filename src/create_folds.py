import pandas as pd
from sklearn import model_selection

if __name__== "__main__" :
    #Read train csv
    df = pd.read_csv('../data/train5.csv')
    #Create column kfold
    df["kfold"] = -1
    #Randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)
    # Initiate the kfold class from model_selection module
    kf = model_selection.KFold(n_splits=5)

    #Fill the new kfold column
    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold'] = fold

    #Save the new csv with kfold column
    df.to_csv("../data/train_folds5.csv", index=False)
