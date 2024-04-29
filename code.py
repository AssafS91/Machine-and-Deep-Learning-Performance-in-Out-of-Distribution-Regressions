import os
from google.colab import drive
drive.mount('/content/drive')
from google.colab import files

!pip install numpy==1.23.4
!pip install tpot
!pip install gplearn
!pip install tensorflow==2.8.0
!pip install autokeras==1.0.18
!pip install keras-tuner==1.1.0
!pip install feyn

import random
import pandas as pd
import numpy as np
from sympy import symbols, lambdify, simplify

import feyn
from gplearn.genetic import SymbolicRegressor
import tpot
from sklearn.model_selection import RepeatedKFold
from tpot import TPOTRegressor
import scipy.stats
from matplotlib.ticker import MaxNLocator, FuncFormatter
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
import time
from sympy import *
import sklearn
from sklearn.metrics import mean_squared_error
import autokeras as ak
from sklearn.linear_model import LogisticRegression
from gplearn.genetic import SymbolicClassifier
from tpot import TPOTClassifier
from sklearn.linear_model import LinearRegression

from sklearn.metrics import roc_auc_score

import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
import time
import os
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from scipy.stats import zscore
from scipy.stats import entropy
from sklearn.metrics import mean_absolute_error



def multivariate_zscore_split(df):
    X = df.drop(columns=['target'])  
    y = df['target']

    mean = np.mean(X, axis=0)
    cov_matrix = np.cov(X, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix + 1/1000000000 * np.eye(cov_matrix.shape[0]))
    diff = X - mean
    z_scores = np.sqrt(np.sum(np.dot(diff, inv_cov_matrix) * diff, axis=1))

    # Calculate the mean z-score for each row
    top_percent = 15
    num_rows_to_select = int(len(X) * (top_percent / 100))

    selected_rows_indices = np.argpartition(z_scores, -num_rows_to_select)[-num_rows_to_select:]
    X_out = X.iloc[selected_rows_indices]
    X_in = X.drop(selected_rows_indices)
    y_out = y.iloc[selected_rows_indices]
    y_in = y.drop(selected_rows_indices)

    X_train, X_test_in, y_train, y_test_in = train_test_split(X_in, y_in, test_size=0.15, random_state=np.random.randint(100))
    X_train, X_val_in, y_train, y_val_in = train_test_split(X_train, y_train, test_size=0.15, random_state=np.random.randint(100))
    X_test_all=pd.concat([X_test_in, X_out])
    diff = X_test_all - mean
    z_scores = np.sqrt(np.sum(np.dot(diff, inv_cov_matrix) * diff, axis=1))

    return [X_train, y_train, X_test_in, y_test_in, X_out, y_out,X_val_in,y_val_in,z_scores]

def random_feature_zscore_split(df):
    X = df.drop(columns=['target']) 
    y = df['target']  

    # Randomly choose one feature
    random_feature = np.random.choice(X.columns)

    # Calculate Z-scores for the randomly chosen feature
    mean = np.mean(X[random_feature])
    std_dev = np.std(X[random_feature])
    z_scores = (X[random_feature] - mean) / std_dev

    # Select rows with high Z-scores as outliers
    top_percent = 15
    num_rows_to_select = int(len(X) * (top_percent / 100))
    selected_rows_indices = np.argpartition(z_scores, -num_rows_to_select)[-num_rows_to_select:]

    # Split the data into training, testing, and outlier sets
    X_out = X.iloc[selected_rows_indices]
    X_in = X.drop(index=selected_rows_indices)
    y_out = y.iloc[selected_rows_indices]
    y_in = y.drop(index=selected_rows_indices)

    X_train, X_test_in, y_train, y_test_in = train_test_split(X_in, y_in, test_size=0.15, random_state=np.random.randint(100))
    X_train, X_val_in, y_train, y_val_in = train_test_split(X_train, y_train, test_size=0.15, random_state=np.random.randint(100))
    X_test_all=pd.concat([X_test_in, X_out])
    z_scores = (X_test_all[random_feature] - mean) / std_dev

    return [X_train, y_train, X_test_in, y_test_in, X_out, y_out, X_val_in, y_val_in,z_scores]


def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def multivariate_kl_divergence_split(df):
    X = df.drop(columns=['target'])  # Remove the 'target' column
    y = df['target']  # Store the 'target' column separately

    mean = np.mean(X, axis=0)
    kl_divergences = np.apply_along_axis(lambda row: kl_divergence(row, mean), axis=1, arr=X)
    top_percent = 15
    num_rows_to_select = int(len(X) * (top_percent / 100))

    selected_rows_indices = np.argpartition(kl_divergences, -num_rows_to_select)[-num_rows_to_select:]
    X_out = X.iloc[selected_rows_indices]
    X_in = X.drop(selected_rows_indices)
    y_out = y.iloc[selected_rows_indices]
    y_in = y.drop(selected_rows_indices)

    X_train, X_test_in, y_train, y_test_in = train_test_split(X_in, y_in, test_size=0.15, random_state=np.random.randint(100))
    X_train, X_val_in, y_train, y_val_in = train_test_split(X_train, y_train, test_size=0.15, random_state=np.random.randint(100))
    X_test_all=pd.concat([X_test_in, X_out])
    z_scores = np.apply_along_axis(lambda row: kl_divergence(row, mean), axis=1, arr=X_test_all)

    return [X_train, y_train, X_test_in, y_test_in, X_out, y_out, X_val_in, y_val_in,z_scores]


def fit_SR_gplearn(X,y,parsimony_coefficient,generations,function_set):

    converter = {
    'add': lambda x, y : x + y,
    'sub': lambda x, y : x - y,
    'mul': lambda x, y : x*y,
    'div': lambda x, y : x/y,
    'sqrt': lambda x : x**0.5,
    'log': lambda x : log(x),
    'abs': lambda x : abs(x),
    'neg': lambda x : -x,
    'inv': lambda x : 1/x,
    'max': lambda x, y : max(x, y),
    'min': lambda x, y : min(x, y),
    'sin': lambda x : sin(x),
    'cos': lambda x : cos(x),
    'pow': lambda x, y : x**y,
    }

    est_gp = SymbolicRegressor(function_set=function_set,
    generations=generations,parsimony_coefficient=parsimony_coefficient,feature_names=X.columns)

    t0 = time.time()
    est_gp.fit(X, y)
    train_time=time.time() - t0
    print('Time to fit:', time.time() - t0, 'seconds')
    return [est_gp,train_time]

def fit_SR_feyn(X,y,max_complexity):
    t0 = time.time()
    ql = feyn.QLattice()

    data=X.copy(deep=True)
    data['target']=y

    models = ql.auto_run(
        data=data,
        output_name='target',
        max_complexity=max_complexity
    )
    best = models[0]
    train_time=time.time() - t0

    return [best,train_time]

def train_TPOT(X_train, X_test_in,X_test_out, y_train, y_test_in,y_test_out,max_time,X_val,y_val):
    max_time_mins=max_time
    model = TPOTRegressor(max_time_mins=max_time_mins,n_jobs=1)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time=time.time() - t0
    model.export('tpot.py')
    y_pred_in = model.predict(X_test_in)
    mse=mean_squared_error(y_test_in, y_pred_in)
    TPOT_in=np.sqrt(mse)
    y_pred_out = model.predict(X_test_out)
    mse=mean_squared_error(y_test_out, y_pred_out)
    TPOT_out=np.sqrt(mse)
    y_pred_train = model.predict(X_train)
    train_mse=mean_squared_error(y_train, y_pred_train)
    train_rmse=np.sqrt(train_mse)
    X_test_all_array = np.concatenate((X_test_in, X_test_out), axis=0)
    X_test_all = pd.DataFrame(X_test_all_array, columns=X_test_in.columns)
    all_preds=model.predict(X_test_all)
    y_pred_val=model.predict(X_val)
    val_mse=mean_squared_error(y_val, y_pred_val)
    val_rmse=np.sqrt(val_mse)

    return [model,TPOT_in,TPOT_out,train_time,train_rmse,all_preds,val_rmse]

def train_autokeras(X_train, X_test_in,X_test_out, y_train, y_test_in,y_test_out,epochs,X_val,y_val):
    reg = ak.StructuredDataRegressor(max_trials=1, overwrite=True)
    t0 = time.time()
    reg.fit(X_train, y_train, epochs=epochs,verbose=1)
    train_time=time.time() - t0
    y_pred_in = reg.predict(X_test_in)
    mse=mean_squared_error(y_test_in, y_pred_in)
    AK_in=np.sqrt(mse)
    y_pred_out = reg.predict(X_test_out)
    mse=mean_squared_error(y_test_out, y_pred_out)
    AK_out=np.sqrt(mse)
    y_pred_train = reg.predict(X_train)
    train_mse=mean_squared_error(y_train, y_pred_train)
    train_rmse=np.sqrt(train_mse)
    X_test_all_array = np.concatenate((X_test_in, X_test_out), axis=0)
    X_test_all = pd.DataFrame(X_test_all_array, columns=X_test_in.columns)
    all_preds=reg.predict(X_test_all)
    y_pred_val=reg.predict(X_val)
    val_mse=mean_squared_error(y_val, y_pred_val)
    val_rmse=np.sqrt(val_mse)
    
    return [reg,AK_in,AK_out,train_time,train_rmse,all_preds,val_rmse]


def train_LR(X_train, X_test_in,X_test_out, y_train, y_test_in,y_test_out):

    gsearch1=LinearRegression()
    LR_reg=gsearch1.fit(X_train,y_train)
    y_pred_in = LR_reg.predict(X_test_in)
    mse=mean_squared_error(y_test_in, y_pred_in)
    rmse=np.sqrt(mse)
    LR_in=rmse
    y_pred_out = LR_reg.predict(X_test_out)
    mse=mean_squared_error(y_test_out, y_pred_out)
    rmse=np.sqrt(mse)
    LR_out=rmse
    X_test_all_array = np.concatenate((X_test_in, X_test_out), axis=0)
    X_test_all = pd.DataFrame(X_test_all_array, columns=X_test_in.columns)
    preds_LR=LR_reg.predict(X_test_all)
    return [LR_in,LR_out,LR_reg,preds_LR]



datasets_list0=['0-Huang1FS.csv',
'0-Huang1CS.csv',
'0-Su1.csv',
'0-Su2.csv',
'01-bachir.csv',
'0-koya2-poisson28nu.csv',
'0-matbench.csv',
'0-koya1-rup28.csv',
'0-koya1-cte.csv',
'0-koya2-comp28.csv',
'0-koya2-split28.csv',
'0-koya2-elast28.csv',
'0-guo1ys.csv',
'0-guo2ts.csv',
'0-guo3el.csv']

reg_datasets = []

file_list_reg = os.listdir(folder_path_reg)
good_file_names = []
for file_name in file_list_reg:
    if file_name in datasets_list0:
        file_path = os.path.join(folder_path_reg, file_name)
        df = pd.read_csv(file_path)
        df = df.dropna()
        for col in df.columns:
            numerical_threshold = 0.9
            numerical_count = pd.to_numeric(df[col], errors='coerce').count()
            total_count = df[col].count()
            percentage_numerical = numerical_count / total_count if total_count > 0 else 0
            if percentage_numerical > numerical_threshold:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            if pd.api.types.is_string_dtype(df[col]):
                unique_values = df[col].unique()
                if len(unique_values) == 2:  # If there are only two values, change them to 0 and 1
                    mapping = {val: idx for idx, val in enumerate(unique_values)}
                    df[col] = df[col].map(mapping)
                else:  
                    df=df.drop(columns=col)
        df = df.dropna()
        reg_datasets.append(df)
        good_file_names.append(file_name)

results=pd.DataFrame()
results2=pd.DataFrame()
function_set = ['add', 'sub', 'mul','div']
max_time=10
epochs=200

for round_number in range(50):
    for i in range(0,len(reg_datasets)):
        print(i)
        print(good_file_names[i])
        df=reg_datasets[i].copy(deep=True)
        X=df.drop(columns=['target'])
        y=df['target']
        #choose one OOD type
        X_train,y_train,X_test_in,y_test_in,X_test_out,y_test_out,X_val_in,y_val_in,z_scores=random_feature_zscore_split(df)
        # X_train,y_train,X_test_in,y_test_in,X_test_out,y_test_out,X_val_in,y_val_in,z_scores=multivariate_zscore_split(df)
        #X_train,y_train,X_test_in,y_test_in,X_test_out,y_test_out,X_val_in,y_val_in,z_scores=multivariate_kl_divergence_split(df)
        z_scores=np.asarray(z_scores) 

        parsimony_coefficient=[0.005,0.01,0.02,0.03,0.04,0.05] #for gplearn
        #parsimony_coefficient=[5,10,15,20,25,30] #max_complexity for feyn
        generations=[30] #for gp learn
        X_SR_train=X_train.copy(deep=True)
        X_SR_test_in=X_test_in.copy(deep=True)
        X_SR_test_out=X_test_out.copy(deep=True)
        SR_results=[]
        SR_times=[]

        best_rmse = float('inf')  # Initialize a variable to keep track of the best RMSE
        best_SR_model = None  # Initialize a variable to store the best SR model
        for pars_coef in parsimony_coefficient:
            for gen in generations:
                [SR_model, train_time] = fit_SR_gplearn(X_train, y_train, pars_coef, gen, function_set)
                #[SR_model, train_time] = fit_SR_feyn(X_train, y_train,pars_coef)
                X_SR_train_temp = X_train.copy()
                X_SR_train_temp['SR'] = SR_model.predict(X_train)
                X_SR_test_temp_in = X_test_in.copy()
                X_SR_test_temp_in['SR'] = SR_model.predict(X_test_in)
                X_SR_test_temp_out = X_test_out.copy()
                X_SR_test_temp_out['SR'] = SR_model.predict(X_test_out)

                y_pred = SR_model.predict(X_val_in)  # Predict on the training data
                mse = mean_squared_error(y_val_in, y_pred)  # Calculate MSE on training data
                rmse = np.sqrt(mse)  # Calculate RMSE

                # If the current model's RMSE is better than the previous best, update the best model and RMSE
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_SR_model = SR_model
                    X_SR_train = X_SR_train_temp
                    X_SR_test_in = X_SR_test_temp_in
                    X_SR_test_out = X_SR_test_temp_out
                # Append the RMSE and training time for the current model
                    y_pred_in = SR_model.predict(X_test_in)  # Predict on the training data
                    mse = mean_squared_error(y_test_in, y_pred_in)  # Calculate MSE on training data
                    test_rmse_in = np.sqrt(mse)  # Calculate RMSE
                    SR_in=test_rmse_in
                    y_pred_out = SR_model.predict(X_test_out)  # Predict on the training data
                    mse = mean_squared_error(y_test_out, y_pred_out)  # Calculate MSE on training data
                    test_rmse_out = np.sqrt(mse)  # Calculate RMSE
                    SR_out=test_rmse_out
                    SR_times=train_time

        X_SR_val_in=X_val_in.copy(deep=True)
        X_SR_val_in['SR']=best_SR_model.predict(X_val_in)

        [trained_model1,TPOT_in,TPOT_out,timeTPOT,train_rmse_TPOT,preds_TPOT,val_TPOT] = train_TPOT(X_train, X_test_in,X_test_out, y_train, y_test_in,y_test_out,max_time,X_val_in,y_val_in)
        print('TPOT time:'+str(timeTPOT))
        [trained_model2,TPOTSR_in,TPOTSR_out,timeTPOTSR,train_rmse_TPOTSR,preds_TPOTSR,val_TPOTSR] = train_TPOT(X_SR_train, X_SR_test_in,X_SR_test_out, y_train, y_test_in,y_test_out,max_time,X_SR_val_in,y_val_in)
        print('TPOTSR time:'+str(timeTPOTSR))
        [trained_model3,AK_in,AK_out,timeAK,train_rmse_AK,preds_AK,val_AK] = train_autokeras(X_train, X_test_in,X_test_out, y_train, y_test_in,y_test_out,epochs,X_val_in,y_val_in)
        print('AK time:'+str(timeAK))
        [trained_model4,AKSR_in,AKSR_out,timeAKSR,train_rmse_AKSR,preds_AKSR,val_AKSR] = train_autokeras(X_SR_train, X_SR_test_in,X_SR_test_out, y_train, y_test_in,y_test_out,epochs,X_SR_val_in,y_val_in)
        print('AKSR time:'+str(timeAKSR))
        [LR_in,LR_out,LR_reg,preds_LR] = train_LR(X_train, X_test_in,X_test_out, y_train, y_test_in,y_test_out)
        row_data = {
            'i': i,
            'dataset':good_file_names[i],
            'SR_in': SR_in,
            'SR_out': SR_out,
            'SR_time': SR_times,
            'TPOT_in': TPOT_in,
            'TPOT_out': TPOT_out,
            'TPOTSR_in': TPOTSR_in,
            'TPOTSR_out': TPOTSR_out,
            'AK_in': AK_in,
            'AK_out': AK_out,
            'AKSR_in': AKSR_in,
            'AKSR_out': AKSR_out,
            'LR_in': LR_in,
            'LR_out': LR_out,
            'timeTPOT': timeTPOT,
            'timeTPOTSR': timeTPOTSR,
            'timeAK': timeAK,
            'timeAKSR': timeAKSR,
            'generations': generations,
            'max_TPOT_time': max_time,
            'tpot_in':TPOT_in/TPOTSR_in,
            'tpot_out': TPOT_out/TPOTSR_out,
            'ak_in':AK_in/AKSR_in,
            'ak_out': AK_out/AKSR_out,
            'tpot_in_conditional': (TPOT_in / (TPOTSR_in if val_TPOTSR < val_TPOT else TPOT_in)),
            'ak_in__conditional': (AK_in / (AKSR_in if val_AKSR < val_AK else AK_in)),
            'tpot_out_conditional': (TPOT_out / (TPOTSR_out if TPOTSR_in < TPOT_in else TPOT_out)),
            'ak_out_conditional': (AK_out / (AKSR_out if AKSR_in < AK_in else AK_out)),
            }
        results = results.append(row_data, ignore_index=True)
        results.to_csv('/content/drive/My Drive/SRresults/results1.csv', index=False)
        print('round number: '+str(round_number))
        X_test_all_array = np.concatenate((X_test_in, X_test_out), axis=0)
        X_test_all = pd.DataFrame(X_test_all_array, columns=X_test_in.columns)
        X_SR_test_all_array = np.concatenate((X_SR_test_in,X_SR_test_out), axis=0)
        X_SR_test_all = pd.DataFrame(X_SR_test_all_array, columns=X_SR_test_in.columns)
        y_test_all = np.concatenate((y_test_in, y_test_out), axis=0)

        random_number=random.randint(1, 1000000)
        indices = X_test_all.index
        if len(indices) > 1000:
            indices = np.random.choice(indices, size=min(1000, len(indices)), replace=False)

        for j in range(len(indices)):
            currentInd=indices[j]
            for k in range(0,7):
                if k==0:
                    model_type = 'TPOT'
                    prediction = preds_TPOT[currentInd]
                    mse=mean_squared_error([y_test_all[currentInd]], [prediction])
                    rmse=np.sqrt(mse)
                elif k==1:
                    model_type = 'TPOTSR'
                    prediction = preds_TPOTSR[currentInd]
                    mse=mean_squared_error([y_test_all[currentInd]], [prediction])
                    rmse=np.sqrt(mse)
                elif k==2:
                    model_type = 'AK'
                    prediction = preds_AK[currentInd]
                    mse=mean_squared_error([y_test_all[currentInd]], [prediction])
                    rmse=np.sqrt(mse)
                elif k==3:
                    model_type = 'AKSR'
                    prediction = preds_AKSR[currentInd]
                    mse=mean_squared_error([y_test_all[currentInd]], [prediction])
                    rmse=np.sqrt(mse)
                elif k==4:
                    model_type = 'LR'
                    prediction = preds_LR[currentInd]
                    mse=mean_squared_error([y_test_all[currentInd]], [prediction])
                    rmse=np.sqrt(mse)
                elif k==5:
                    model_type = 'TPOT_combined'
                    if val_TPOTSR<val_TPOT:
                      prediction = preds_TPOTSR[currentInd]
                    else:
                      prediction = preds_TPOT[currentInd]
                    mse=mean_squared_error([y_test_all[currentInd]], [prediction])
                    rmse=np.sqrt(mse)
                elif k==6:
                    model_type = 'AK_combined'
                    if val_AKSR<val_AK:
                      prediction = preds_AKSR[currentInd]
                    else:
                      prediction = preds_AK[currentInd]
                    mse=mean_squared_error([y_test_all[currentInd]], [prediction])
                    rmse=np.sqrt(mse)

                row_data2 = {
                    'i': random_number,
                    'dataset':good_file_names[i],
                    'mean_y':np.mean(y),
                    'mean_abs_y':np.mean(np.abs(y)),
                    'std_y':np.std(y),
                    'ood':'z_score',
                    'sr':'gp',
                    'z_score': z_scores[currentInd],
                    'model': model_type,
                    'rmse':rmse,
                    'tpot_valgood': 1 if val_TPOTSR < val_TPOT else 0,
                    'ak_valgood': 1 if val_AKSR < val_AK else 0
                }
                row_df = pd.DataFrame([row_data2])
                results2 = pd.concat([results2, row_df], ignore_index=True)
                results2.to_csv('/content/drive/My Drive/SRresults/OOD1.csv', index=False)