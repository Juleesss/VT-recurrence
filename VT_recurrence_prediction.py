

### This code is used in this article:
# Machine Learning Based Prediction of 1-year Arrhythmia Re-currence after Ventricular Tachycardia Ablation in Patients with Structural Heart Disease
# Ferenc Komlósi M.D.1†, Patrik Tóth M.D. 1†, Gyula Bohus1, Péter Vámosi M.D. 1, Márton Tokodi M.D, Ph.D. 1,2 Nándor Szegedi M.D. 1, Ph.D., Zoltán Salló M.D. 1, Katalin Piros M.D. 1, Péter Perge M.D. Ph.D. 1, István Osztheimer M.D. Ph.D. 1, Pál Ábrahám M.D. Ph.D. 1, Gábor Széplaki M.D. 3, Ph.D., Béla Merkely M.D. 1, Ph.D., D.Sc., László Gellér M.D. , Ph.D., D.Sc. 1 ‡, Klaudia Vivien Nagy M.D., Ph.D. 1‡*



'''Copyright <2023> <Gyula Bohus>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.'''



import numpy as np
import pandas as pd
import time
import datetime
import os



from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline as skPipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import KNNImputer

from xgboost import XGBClassifier

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.pipeline import Pipeline as imbPipeline

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical






stratsplit_5fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng)
stratsplit_10fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=rng)



#create feature rankings:
def rank_features(
        X_data, # imputed dataframe
        y_data, # pd.Series containing the endpoints
        rs # random state for initialization and reproducibility
): # returns a list of lists, which contain the rank of each feature in a way that list[0][6] = 3 means that based on the first ranking method the 7th feature is the 4th best
    #find optimal model to use for rankings

    MLP_search_grid = [
    {
            'NN__hidden_layer_sizes':  [(5,), (10,), (10, 5), (5, 5) ],
            'NN__alpha': [ 1, 0.01, 0.001],
            'NN__max_iter': [200],
            'NN__activation': ['identity', 'tanh', 'relu'],
            'NN__solver' : ['adam', 'lbfgs'],
            'NN__random_state' : [rs]
        }
    ]


    RF_search_grid = [{'max_depth': [2, 3],
                            'max_features': [2, 3],
                                'min_samples_leaf' : [5, 10],
                                'criterion' : ['gini', 'log_loss'],
                                'n_estimators' : [500],
                                'n_jobs' : [-1],
            'random_state' : [rs]}]


    XGB_search_grid = [{
            'min_child_weight': [5, 10],
            'gamma': [0.5, 1, 1.5, 2],
            'subsample': [0.6, 0.8],
            'colsample_bytree': [0.6, 0.8],
            'max_depth': [2, 3],
            'random_state' : [rs]
            }]

    rfstratsplit = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
    fs_grid_MLP = pd.DataFrame(
        GridSearchCV(
        skPipeline([('scaler' , StandardScaler()), ('NN', MLPClassifier())]),
        # skPipeline([('scaler' , StandardScaler()), ('NN', MLPClassifier())]),
        MLP_search_grid,
        scoring={'AUC' : 'roc_auc', 'Accuracy' : make_scorer(accuracy_score), 'F1' : make_scorer(f1_score)},
        cv= rfstratsplit,
        n_jobs=6,
        verbose=3,
        refit=False).fit(X_data, y_data).cv_results_
    ).sort_values('mean_test_AUC', ascending=False)

    fs_grid_RF = pd.DataFrame(
    GridSearchCV(
        RandomForestClassifier(),
        RF_search_grid,
        scoring={'AUC' : 'roc_auc', 'Accuracy' : make_scorer(accuracy_score), 'F1' : make_scorer(f1_score)},
        cv= rfstratsplit,
        n_jobs=6,
        verbose=3,
        refit=False).fit(X_data, y_data).cv_results_
    ).sort_values('mean_test_AUC', ascending=False)

    fs_grid_XGB = pd.DataFrame(
    GridSearchCV(
        XGBClassifier(),
        XGB_search_grid,
        scoring={'AUC' : 'roc_auc', 'Accuracy' : make_scorer(accuracy_score), 'F1' : make_scorer(f1_score)},
        cv= rfstratsplit,
        n_jobs=6,
        verbose=3,
        refit=False).fit(X_data, y_data).cv_results_
    ).sort_values('mean_test_AUC', ascending=False)

    fs_XGB = XGBClassifier().set_params(**fs_grid_XGB.loc[0, 'params'])
    fs_RF =RandomForestClassifier().set_params(**fs_grid_RF.loc[0, 'params'])
    fs_MLP = skPipeline([('scaler' , StandardScaler()), ('NN', MLPClassifier())]).set_params(**fs_grid_MLP.loc[0, 'params'])


    global feature_ranking_methods
    feature_ranking_methods = ['XGB, RFE', 'RF, PI', 'XGB, PI', 'MLP, PI', 'SUM']

    feature_ranking = [
        RFE(fs_XGB, n_features_to_select=1).fit(X_data, y_data).ranking_,
        np.argsort(np.argsort(np.array([-imp for imp in permutation_importance(fs_RF.fit(X_data, y_data), X_data, y_data, scoring='roc_auc', n_repeats=10, n_jobs=6, random_state=rs).importances_mean]))),
        np.argsort(np.argsort(np.array([-imp for imp in permutation_importance(fs_XGB.fit(X_data, y_data), X_data, y_data, scoring='roc_auc', n_repeats=10, n_jobs=6, random_state=rs).importances_mean]))),
        np.argsort(np.argsort(np.array([-imp for imp in permutation_importance(fs_MLP.fit(X_data, y_data), X_data, y_data, scoring='roc_auc', n_repeats=10, n_jobs=6, random_state=rs).importances_mean])))
    ]

    feature_ranking = np.array(feature_ranking)

    #add sum rank
    sum_rank = np.argsort(np.argsort(feature_ranking.sum(axis=0)))
    feature_ranking = np.append(feature_ranking, np.array([sum_rank]), axis=0)

    feature_ranking_df = pd.DataFrame(columns=X_data.columns, index=feature_ranking_methods, data = feature_ranking)
    feature_ranking_df['selected model params'] = [
        fs_grid_XGB.loc[0, 'params'],
        fs_grid_RF.loc[0, 'params'],
        fs_grid_XGB.loc[0, 'params'],
        fs_grid_MLP.loc[0, 'params'],
        np.nan
    ]
    return feature_ranking_df


def serious_feature_ranking(
        number_of_repeats, # how many times should the ranking be repeated for more reliability of the end results
        X_dataframe, # imputed pd.Dataframe
        y_series # pd.Series containing the endpoints
):
    feature_ranking_dataframes = []
    for random_state in range(number_of_repeats):
        print(f'Feature selection round: {random_state}')
        feature_ranking = rank_features(X_dataframe, y_series, random_state)
        feature_ranking_dataframes.append(feature_ranking.reset_index())
    all_ranking_df = pd.concat(feature_ranking_dataframes, axis=0)


    sum_ranking_df = all_ranking_df.drop('selected model params', axis=1).groupby('index').sum()
    data_raw = sum_ranking_df.apply(lambda x: np.argsort(np.argsort(x.values)), axis=1).values
    data_fordf = np.array([arr for arr in data_raw]) # for some reason the shape of the array needs to be reset
    new_ranking_df = pd.DataFrame(index=sum_ranking_df.index, columns=X_dataframe.columns, data = data_fordf)

    return new_ranking_df, all_ranking_df


def create_feature_groups_from_rankings(
        feature_rankings_dataframe, # combined feature ranking produced by the serious_feature_rankng() funcion
        coldata = [], # columns names as list used at the start of the feature selection
        mincolnum = 3, # minimum number of variables present in a group
        maxcolnum = 9 # maximum number of variables present in a group
):
    feature_groups_list = []
    feature_groups_indices = []
    for index, row in feature_rankings_dataframe.iterrows():
        for i in range(mincolnum, maxcolnum+1):
            cols = [1 if x < i else 0 for x in row.values]
            feature_groups_list.append(cols)
            feature_groups_indices.append(index + '#' +str(i))

    feature_groups_df = pd.DataFrame(columns=coldata, data= feature_groups_list, index=feature_groups_indices)
    feature_groups_df.reset_index()

    feature_groups_df['selected_columns'] = feature_groups_df.apply(lambda x: '#'.join([colval for colval in list(feature_groups_df.columns.values[x.values == 1])]), axis=1)
    # feature_groups_df['endpoint'] = [ind.split('#')[2] for ind in feature_groups_df.index.values]

    return feature_groups_df


def find_best_with_oversampling(
        subX_df, # NOT imputed data
        suby_df, # endpoints as pd.Series
        seriousness = 6 # how many jobs should run in parallel (= sklearn's n_jobs)
):
    MLP_search_grid_ms = [
        {
            'NN__hidden_layer_sizes': [(20,), (5, 4, 3), (5,), (10,), (10, 5), (5, 5)],
            'NN__alpha': [10, 1, 0.01, 0.001, 0.0001],
            'NN__max_iter': [200, 150],
            'NN__activation': ['identity', 'logistic', 'tanh', 'relu'],
            'NN__solver': ['adam', 'lbfgs'],
            'NN__random_state': [0]
        }
    ]

    RF_search_grid_ms = [{'RF__max_depth': [2, 3, 5],
                          'RF__max_features': [2, 3, 5],
                          'RF__min_samples_leaf': [5, 10, 20],
                          'RF__criterion': ['gini', 'log_loss'],
                          'RF__n_estimators': [100, 500, 700],
                          'RF__n_jobs': [-1],
                          'RF__random_state': [0, 42]}]

    XGB_search_grid_ms = [{
        'xgb__min_child_weight': [1, 5, 10],
        'xgb__gamma': [0.5, 1, 1.5, 2, 5],
        'xgb__subsample': [0.6, 0.8, 1.0],
        'xgb__colsample_bytree': [0.6, 0.8, 1.0],
        'xgb__max_depth': [3, 4, 5],
        'xgb__random_state': [0, 42]
    }]

    ms_mlp = imbPipeline([('KNNimputer', KNNImputer(n_neighbors=6, weights='distance')), ('scaler' , StandardScaler()), ('oversample', RandomOverSampler(random_state=rng)), ('NN', MLPClassifier())])
    ms_grid_MLP = pd.DataFrame(
        GridSearchCV(
            ms_mlp,
            MLP_search_grid_ms,
            scoring={'AUC' : 'roc_auc', 'Accuracy' : make_scorer(accuracy_score), 'F1' : make_scorer(f1_score)},
            cv= StratifiedKFold(n_splits=5, shuffle=True, random_state=rng),
            n_jobs=seriousness,
            verbose=3,
            refit=False).fit(subX_df.values, suby_df.values).cv_results_)

    ms_xgb = imbPipeline([('KNNimputer', KNNImputer(n_neighbors=6, weights='distance')), ('ros', RandomOverSampler(random_state = rng)) , ('xgb', XGBClassifier())])
    ms_grid_XBG = pd.DataFrame(
        GridSearchCV(
            ms_xgb,
            XGB_search_grid_ms,
            scoring={'AUC' : 'roc_auc', 'Accuracy' : make_scorer(accuracy_score), 'F1' : make_scorer(f1_score)},
            cv= StratifiedKFold(n_splits=5, shuffle=True, random_state=rng),
            n_jobs=seriousness,
            verbose=3,
            refit=False).fit(subX_df.values, suby_df.values).cv_results_)

    ms_rf = imbPipeline([('KNNimputer', KNNImputer(n_neighbors=6, weights='distance')), ('ros', RandomOverSampler(random_state = rng)), ('RF', RandomForestClassifier())])
    ms_grid_RF = pd.DataFrame(
        GridSearchCV(
            ms_rf,
            RF_search_grid_ms,
            scoring={'AUC' : 'roc_auc', 'Accuracy' : make_scorer(accuracy_score), 'F1' : make_scorer(f1_score)},
            cv= StratifiedKFold(n_splits=5, shuffle=True, random_state=rng),
            n_jobs=seriousness,
            verbose=3,
            refit=False).fit(subX_df.values, suby_df.values).cv_results_)

    valued_columns = ['params', 'mean_test_AUC', 'std_test_AUC']

    fullres = pd.concat([ms_grid_RF[valued_columns], ms_grid_XBG[valued_columns], ms_grid_MLP[valued_columns]], axis=0)
    fullres.sort_values(by='mean_test_AUC', ascending=False, inplace=True)

    group_results_auc = fullres['mean_test_AUC'].values
    group_results_auc = group_results_auc[np.isfinite(group_results_auc)]
    result_auc = np.percentile(group_results_auc, 90)

    return result_auc, fullres['mean_test_AUC'].values[0], fullres['params'].values[0]


models = {}
model_param_grids = {}
model_param_spaces = {}
model_names = []


RF_search_gridspace_bayes = [{
    'RF__max_depth': Integer(1, 5, name='RF__max_depth'),
    'RF__max_features': Integer(1, 9, name='RF__max_features'),
    'RF__min_samples_leaf': Integer(1, 20, 'log-uniform', name='RF__min_samples_leaf'),
    'RF__criterion': Categorical(['gini', 'log_loss'], name='RF__criterion'),
    'RF__n_estimators': Integer(300, 1000, 'log-uniform', name='RF__n_estimators'),
    'RF__class_weight': Categorical(['balanced', 'balanced_subsample', None]),
    'RF__random_state': Integer(0, 100)
                            }]

model_param_spaces['RF'] = RF_search_gridspace_bayes
model_names.append('RF')
models['RF'] = RandomForestClassifier()

XGB_search_gridspace_bayes = [{
    'XGB__min_child_weight': Real(0, 50),
    'XGB__gamma': Real(0.1, 5),
    'XGB__subsample': Real(0.5, 0.7),
    'XGB__colsample_bytree': Real(0.5, 1),
    'XGB__max_depth': Integer(1, 3),
    'XGB__scale_pos_weight': Real(0.5, 2),
    'XGB__alpha' : Real(5, 30),
    'XGB__random_state': [0]
}]

model_param_spaces['XGB'] = XGB_search_gridspace_bayes
model_names.append('XGB')
models['XGB'] = XGBClassifier()

MLP_search_gridspace_bayes = [
    {
        # 'NN__hidden_layer_sizes':  Categorical([(20), (15), (25), (5, 4, 3), (10, 10, 5), (6, 5, 4), (5, 4, 3) , (5), (10,), (10, 5), (5, 5)]),
        'NN__hidden_layer_sizes': Categorical(
            [(20), (15), (10), (25), (9), (8), (7), (6), (5), (4), (3), (22), (24), (27), (30), (35), (40), (50)]),
        'NN__alpha': Real(0.0001, 10, 'log-uniform'),
        'NN__max_iter': Integer(50, 500),
        'NN__activation': Categorical(['identity', 'logistic', 'tanh', 'relu']),
        'NN__solver': Categorical(['adam', 'lbfgs']),
        'NN__random_state': [0]
    }
]

model_param_spaces['NN'] = MLP_search_gridspace_bayes
model_names.append('NN')
models['NN'] = MLPClassifier()

preprocessing_ways = [('SS', StandardScaler()), ('MM', MinMaxScaler())]

oversampling_ways = [('ROS', RandomOverSampler(random_state=rng)), ('SMOTE', SMOTE(random_state=rng)),
                     ('ADASYN', ADASYN(random_state=rng))]


def find_best_with_bayes(
        X_dataframe,
        y_dataframe,
        niter_for_bayes = 100, # how many iterations should the BayesSearchsCV function make
        seriousness = 6, # how many jobs should run in parallel (= sklearn's n_jobs)
        cross_val_splits = 10
):
    iternum = 0
    performance_evaluation_df_list = []

    for MLmethod in [model_names[1]]:
        stime = time.time()
        iternum +=1
        preformance_df = pd.DataFrame(
            BayesSearchCV(
                imbPipeline([('KNNimpute', KNNImputer(n_neighbors=5, weights='distance')),(MLmethod, models[MLmethod])]),
                model_param_spaces[MLmethod],
                scoring= 'roc_auc',
                cv = StratifiedKFold(n_splits=cross_val_splits, shuffle=True, random_state=rng),
                n_jobs=seriousness,
                verbose=0,
                refit=False,
                error_score='raise',
                return_train_score=True,
                n_iter=niter_for_bayes,
                random_state=rng
            ).fit(X_dataframe.values, y_dataframe.values).cv_results_
        )
        preformance_df.sort_values(by='mean_test_score', ascending=False, inplace=True)
        preformance_df['model'] = MLmethod
        preformance_df['preprocessing'] = 'Not used'
        preformance_df['oversampling'] = 'Not used'
        # preformance_df=preformance_df.drop_duplicates(subset=['model', 'preprocessing', 'oversampling', 'params', 'mean_test_score'], keep='first').head(30)

        preformance_df = preformance_df[['model', 'preprocessing', 'oversampling', 'params', 'mean_test_score', 'mean_train_score', 'std_test_score', 'std_train_score']]
        # display(preformance_df)
        performance_evaluation_df_list.append(preformance_df)
        print(f'Done with:', iternum, MLmethod, f' no preprocessing, it took: {round((time.time()-stime)/60, 2)}minutes')

        for preprocessing in preprocessing_ways:
            for oversampling in oversampling_ways:
                stime = time.time()
                iternum +=1
                preformance_df = pd.DataFrame(
                    BayesSearchCV(
                        imbPipeline([('KNNimpute', KNNImputer(n_neighbors=5, weights='distance')), preprocessing,  oversampling, (MLmethod, models[MLmethod])]),
                        model_param_spaces[MLmethod],
                        scoring= 'roc_auc',
                        cv = StratifiedKFold(n_splits=cross_val_splits, shuffle=True, random_state=rng),
                        n_jobs=seriousness,
                        verbose=0,
                        refit=False,
                        # error_score='raise',
                        return_train_score=True,
                        n_iter=niter_for_bayes,
                        random_state=rng
                    ).fit(X_dataframe.values, y_dataframe.values).cv_results_
                )
                preformance_df.sort_values(by='mean_test_score', ascending=False, inplace=True)
                
                preformance_df['model'] = MLmethod
                preformance_df['preprocessing'] = preprocessing[0]
                preformance_df['oversampling'] = oversampling[0]
                # preformance_df=preformance_df.drop_duplicates(subset=['model', 'preprocessing', 'oversampling', 'params', 'mean_test_score'], keep='first').head(30)
                preformance_df = preformance_df[['model', 'preprocessing', 'oversampling', 'params', 'mean_test_score', 'mean_train_score', 'std_test_score', 'std_train_score']]
                # display(preformance_df)
                performance_evaluation_df_list.append(preformance_df)
                print(f'Done with:', iternum, MLmethod, preprocessing, oversampling, f'it took: {round((time.time()-stime)/60, 2)}minutes')
    performance_df = pd.concat(performance_evaluation_df_list, axis=0)
    return performance_df


#the functions are used as shown below:

patient_data_df = pd.DataFrame() #data used for training 
columns_used_at_start = []
endpoint = '' #name of the columns used as endppoint


X_df = patient_data_df[columns_used_at_start]
knn_imputer = KNNImputer(n_neighbors=10, weights='distance')
X_imputed_df = pd.DataFrame(columns=X_df.columns, data = knn_imputer.fit_transform(X_df))

y_ser = patient_data_df[endpoint]


feature_ranking_dataframe, all_feature_ranking_dataframe = serious_feature_ranking(
    X_dataframe = X_imputed_df,
    y_series = y_ser,
    number_of_repeats= 100 #number_of_iterations_for_feature_ranking
)



feature_groups_df = create_feature_groups_from_rankings(
    feature_ranking_dataframe,
    coldata = X_df.columns,
    mincolnum=4, #minimum_number_of_columns
    maxcolnum=7 #maximum_number_of_columns
)




new_os_cols_to_groups = ['os_auc_score', 'os_best_auc', 'os_best_params']
prog = 1
for index, row in feature_groups_df.iterrows():
    startime = time.time()
    subcols = row['selected_columns'].split('#')

    if np.isnan(feature_groups_df.loc[index, 'os_auc_score']):
        os_score, os_best_auc, os_best_params = find_best_with_oversampling(
            X_df[subcols],
            y_ser,
            1 )#number_of_parallel_jobs
        feature_groups_df.loc[index, new_os_cols_to_groups] = os_score, os_best_auc, os_best_params

best_columns = feature_groups_df.sort_values(by='os_auc_score', ascending=False)['selected_columns'].values[0].split('#')

model_search_df = find_best_with_bayes(
    X_df[best_columns], 
    y_ser, 
    1  # number_of_parallel_jobs
    ) 


#description of the final model pipeline
describe_best = model_search_df.sort_values(by='mean_test_score', ascending=False).iloc[0, :].to_list()