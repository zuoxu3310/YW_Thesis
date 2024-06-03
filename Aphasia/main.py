import numpy as np
import pandas as pd
import yaml

import random
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

import optuna
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, make_scorer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import fbeta_score, make_scorer
from func import weighted_balanced_accuracy,fbeta_macro, remap_labels,optimize_model_parameters,compute_scores,process_results,plot_cv_results

# Baseline Models
from sklearn.dummy import DummyClassifier

# Bayesian Models
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Linear Models
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier

# Instance-Based Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier

# Kernel-Based Models
from sklearn.svm import SVC

# Tree Models
from sklearn.tree import DecisionTreeClassifier

# Ensemble Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
import lightgbm as lgb

# Deep Learning Models
from sklearn.neural_network import MLPClassifier

# Load configurations
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load data
df = pd.read_csv(config['data_path'])
demographic_features = df[['Gender', 'Age', 'Post onset']]
scan_feature_start = df.columns.get_loc('CinMid - GM')
scan_features = df.iloc[:, scan_feature_start:]
selected_features = list(demographic_features) + list(scan_features)
categorical_features = ['Gender']
numerical_features = ['Age', 'Post onset'] + list(scan_features)
X = df[selected_features].copy()
y = df['Aphasia_severity'].copy()

# Set configurations
EXP = config['experiment_name']  

combinations = config['combinations'][config['start']:config['end']]

seed = config['seed']

n_trials = config['number_of_trials']

all_models = config['model_selection']['all_models']

selected_models = config['model_selection']['selected_models']

weighted_ba_scorer = make_scorer(weighted_balanced_accuracy)
f_0_5_macro = make_scorer(fbeta_macro, beta=0.5)
f_2_macro = make_scorer(fbeta_macro, beta=2)

scoring = config['scoring']
scoring_metrics_yaml = config['scoring_metrics']
scoring_metrics_default = {
    'accuracy': 'accuracy',
    'balanced_accuracy': 'balanced_accuracy',
    'weighted_balanced_accuracy': weighted_ba_scorer,
    'precision_macro': 'precision_macro',
    'precision_micro': 'precision_micro',
    'recall_macro': 'recall_macro',
    'recall_micro': 'recall_miacro',
    'f1_macro': 'f1_macro',
    'f1_micro': 'f1_micro',
    'f0.5_macro': f_0_5_macro,
    'f2_macro': f_2_macro,
}
scoring_metrics = {}
for metric in scoring_metrics_yaml:
    if metric in scoring_metrics_default:
        scoring_metrics[metric] = scoring_metrics_default[metric]

cv = StratifiedKFold(
    n_splits=config['cross_validation']['n_splits'],
    shuffle=config['cross_validation']['shuffle'],
    random_state=config['cross_validation']['random_state']
)

if config['oversampling']:
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE
    smt = SMOTE(random_state=config['seed'])
    def create_pipeline(model, numerical_features, categorical_features):
        numerical_pipeline = Pipeline([
            ('scaler', StandardScaler())
        ])
        categorical_pipeline = Pipeline([
            ('encoder', OneHotEncoder())
        ])
        preprocessor = ColumnTransformer([
            ('numerical', numerical_pipeline, numerical_features),
            ('categorical', categorical_pipeline, categorical_features)
        ])
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('smote', smt),
            ('model', model)
        ])
        return pipeline
else:
    from sklearn.pipeline import Pipeline
    def create_pipeline(model, numerical_features, categorical_features):
        numerical_pipeline = Pipeline([
            ('scaler', StandardScaler())
        ])
        categorical_pipeline = Pipeline([
            ('encoder', OneHotEncoder())
        ])
        preprocessor = ColumnTransformer([
            ('numerical', numerical_pipeline, numerical_features),
            ('categorical', categorical_pipeline, categorical_features)
        ])
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        return pipeline



#-------------------------------------------------------------------------------#
#--------------------------------Experiment-------------------------------------#
#-------------------------------------------------------------------------------#

if __name__ == "__main__":    
    np.random.seed(seed)
    random.seed(seed)

    if all_models or 'RandomLabeling' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing RandomLabeling\n")
        print('='*80)

        def objective(trial, X, y, cv, random_state=42):

            strategy = trial.suggest_categorical('strategy', ['uniform'])

            #-------------------------------Model--------------------------------------#
            model = DummyClassifier(strategy=strategy, random_state=random_state)
            #--------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features)

            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)

            return np.mean(scores)

        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=1, seed=seed)

        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            #-------------------------------------------------------------------------#
            model = DummyClassifier(**params, random_state=seed)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)

        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        all_results.to_csv(f'{EXP}_RandomLabeling.csv')


    if all_models or 'MajorityVote' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing MajorityVote\n")
        print('='*80)
        
        def objective(trial, X, y, cv, random_state=42):

            strategy = trial.suggest_categorical('strategy', ['most_frequent','prior'])

            #-------------------------------Model--------------------------------------#
            model = DummyClassifier(strategy=strategy, random_state=random_state)
            #--------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features)

            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)

            return np.mean(scores)

        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=1, seed=seed)

        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            #-------------------------------------------------------------------------#
            model = DummyClassifier(**params, random_state=seed)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)
            
        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        all_results.to_csv(f'{EXP}_MajorityVote.csv')


    if all_models or 'DecisionStump' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing DecisionStump\n")
        print('='*80)

        def objective(trial, X, y, cv, random_state=42):

            max_depth = trial.suggest_int('max_depth', 1, 1)

            #-------------------------------Model--------------------------------------#
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
            #--------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)

            return np.mean(scores)

        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=1, seed=seed)

        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            #-------------------------------------------------------------------------#
            model = DecisionTreeClassifier(max_depth=1, random_state=seed)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)
        
        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        all_results.to_csv(f'{EXP}_DecisionStump.csv')


    if all_models or 'NaiveBayes' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing NaiveBayes\n")
        print('='*80)

        def objective(trial, X, y, cv, random_state=42):
            var_smoothing = trial.suggest_float('var_smoothing', 1e-12, 1e-1, log=True)

            #-------------------------------Model--------------------------------------#
            model = GaussianNB(
                var_smoothing=var_smoothing
            )
            #--------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)

            return np.mean(scores)

        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=n_trials, seed=seed)    
        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            #-------------------------------------------------------------------------#
            model = GaussianNB(**params)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)
        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        all_results.to_csv(f'{EXP}_{model_name}.csv')

    if all_models or 'QDA' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing QDA\n")
        print('='*80)

        def objective(trial, X, y, cv, random_state=42):

            reg_param = trial.suggest_float('reg_param', 1e-5, 1.0, log=True)

            #-------------------------------Model--------------------------------------#
            model = QuadraticDiscriminantAnalysis(reg_param=reg_param)
            #--------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)
            return np.mean(scores)

        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=n_trials, seed=seed)
        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            #-------------------------------------------------------------------------#
            model = QuadraticDiscriminantAnalysis(**params)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)
        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        all_results.to_csv(f'{EXP}_{model_name}.csv')


    if all_models or 'KNN' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing KNN\n")
        print('='*80)


        def objective(trial, X, y, cv, random_state=42):

            n_neighbors = trial.suggest_int('n_neighbors', 1, 100, log=True)
            weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
            p = trial.suggest_int('p', 1, 2)
            algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree','brute'])
            leaf_size = trial.suggest_int('leaf_size', 1, 100, log=True)

            #-------------------------------Model--------------------------------------#
            model = KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    algorithm=algorithm,
                    leaf_size=leaf_size,
                    p=p,
                )
            #--------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)

            return np.mean(scores)

        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=n_trials, seed=seed)
        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            #-------------------------------------------------------------------------#
            model = KNeighborsClassifier(**params)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)

        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        all_results.to_csv(f'{EXP}_{model_name}.csv')


    if all_models or 'RKNN' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing RKNN\n")
        print('='*80)

        def objective(trial, X, y, cv, random_state=42):
            radius = trial.suggest_float('radius', 1e-3, 10.0, log=True)
            weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
            algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
            leaf_size = trial.suggest_int('leaf_size', 1, 100, log=True)
            outlier_label=trial.suggest_categorical('outlier_label', ['most_frequent'])
            p = trial.suggest_int('p', 1, 2)

            #-------------------------------Model--------------------------------------#
            model = RadiusNeighborsClassifier(
                radius=radius,
                weights=weights,
                algorithm=algorithm,
                leaf_size=leaf_size,
                outlier_label=outlier_label,
                p=p
            )
            #--------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)

            return np.mean(scores)
        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=n_trials, seed=seed)
        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            #-------------------------------------------------------------------------#
            model = RadiusNeighborsClassifier(**params)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)


        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        all_results.to_csv(f'{EXP}_{model_name}.csv')



    if all_models or 'LR' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing LR\n")
        print('='*80)

        def objective(trial, X, y, cv, random_state=42):

            penalty = trial.suggest_categorical('penalty', ['l2'])
            C = trial.suggest_float('C', 1e-5, 10, log=True)
            max_iter = trial.suggest_int('max_iter', 1500, 2500)
            tol = trial.suggest_float('tol', 1e-5, 10.0, log=True)
            class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])

            #-------------------------------Model--------------------------------------#
            model = LogisticRegression(
                penalty=penalty,
                C=C,
                max_iter=max_iter,
                solver='lbfgs',
                tol=tol,
                class_weight=class_weight,
                multi_class='multinomial',
                random_state=random_state
            )
            #--------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)

            return np.mean(scores)
        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=n_trials, seed=seed)
        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            #-------------------------------------------------------------------------#
            model = LogisticRegression(**params, random_state=seed)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)
        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        all_results.to_csv(f'{EXP}_{model_name}.csv')

    if all_models or 'RR' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing RR\n")
        print('='*80)

        def objective(trial, X, y, cv, random_state=42):
            alpha = trial.suggest_float('alpha', 1e-3, 10.0, log=True)
            tol = trial.suggest_float('tol', 1e-5, 1e-1, log=True)
            class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])

            #-------------------------------Model--------------------------------------#
            model = RidgeClassifier(
                alpha=alpha,
                tol=tol,
                class_weight=class_weight,
                random_state=random_state
            )
            #--------------------------------------------------------------------------#


            pipeline = create_pipeline(model, numerical_features, categorical_features)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)

            return np.mean(scores)
        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=n_trials, seed=seed)
        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            #-------------------------------------------------------------------------#
            model = RidgeClassifier(**params, random_state=seed)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)
        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        all_results.to_csv(f'{EXP}_{model_name}.csv')

    if all_models or 'SVC_linear' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing SVC_linear\n")
        print('='*80)

        def objective(trial, X, y, cv, random_state=42):

            C = trial.suggest_float('C', 1e-5, 10, log=True)
            kernel = trial.suggest_categorical('kernel', ['linear'])
            degree = trial.suggest_int('degree', 1, 10) if kernel == 'poly' else 3
            gamma = trial.suggest_float('gamma', 1e-5, 10, log=True)
            coef0 = trial.suggest_float('coef0', 0.0, 10.0) if kernel in ['poly', 'sigmoid'] else 0.0
            #shrinking = trial.suggest_categorical('shrinking', [True, False])
            probability = trial.suggest_categorical('probability', [True, False])
            class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])


            #-------------------------------Model--------------------------------------#
            model = SVC(
                    C=C,
                    kernel=kernel,
                    degree=degree,
                    gamma=gamma,
                    coef0=coef0,
                    #shrinking=shrinking,
                    class_weight=class_weight,
                    probability=probability,
                    random_state=random_state
                )
            #--------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)
            return np.mean(scores)
        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=n_trials, seed=seed)
        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            #-------------------------------------------------------------------------#
            model = SVC(**params, random_state=seed)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)    

        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        kernel = model.kernel
        all_results.to_csv(f'{EXP}_{model_name}_{kernel}.csv')

    if all_models or 'SVC_poly' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing SVC_poly\n")
        print('='*80)


        def objective(trial, X, y, cv, random_state=42):

            C = trial.suggest_float('C', 1e-5, 10, log=True)
            kernel = trial.suggest_categorical('kernel', ['poly'])
            degree = trial.suggest_int('degree', 1, 10) if kernel == 'poly' else 3
            #gamma = trial.suggest_categorical('gamma', ['scale','auto'])
            gamma = trial.suggest_float('gamma', 1e-2, 10.0,log=True)
            coef0 = trial.suggest_float('coef0', 0.0, 10.0) if kernel in ['poly', 'sigmoid'] else 0.0
            #shrinking = trial.suggest_categorical('shrinking', [True, False])
            probability = trial.suggest_categorical('probability', [True, False])
            class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])

            #-------------------------------Model--------------------------------------#
            model = SVC(
                    C=C,
                    kernel=kernel,
                    degree=degree,
                    gamma=gamma,
                    coef0=coef0,
                    #shrinking=shrinking,
                    #probability=probability,
                    max_iter=20000,
                    #class_weight=class_weight,
                    random_state=random_state
                )

            #--------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)
            return np.mean(scores)
        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=n_trials, seed=seed)
        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            #-------------------------------------------------------------------------#
            model = SVC(**params, random_state=seed)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)
        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        kernel = model.kernel
        all_results.to_csv(f'{EXP}_{model_name}_{kernel}.csv')


    if all_models or 'SVC_sigmoid' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing SVC_sigmoid\n")
        print('='*80)


        def objective(trial, X, y, cv, random_state=42):

            C = trial.suggest_float('C', 1e-5, 10, log=True)
            kernel = trial.suggest_categorical('kernel', ['sigmoid'])
            degree = trial.suggest_int('degree', 1, 10) if kernel == 'poly' else 3
            #gamma = trial.suggest_categorical('gamma', ['scale','auto'])
            gamma = trial.suggest_float('gamma', 1e-2, 10.0,log=True)
            coef0 = trial.suggest_float('coef0', 0.0, 10.0) if kernel in ['poly', 'sigmoid'] else 0.0
            shrinking = trial.suggest_categorical('shrinking', [True, False])
            probability = trial.suggest_categorical('probability', [True, False])
            class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])

            #-------------------------------Model--------------------------------------#
            model = SVC(
                    C=C,
                    kernel=kernel,
                    degree=degree,
                    gamma=gamma,
                    coef0=coef0,
                    #shrinking=shrinking,
                    probability=probability,
                    class_weight=class_weight,
                    random_state=random_state
                )

            #--------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)

            return np.mean(scores)
        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=n_trials, seed=seed)    
        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            #-------------------------------------------------------------------------#
            model = SVC(**params, random_state=seed)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)
        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        kernel = model.kernel
        all_results.to_csv(f'{EXP}_{model_name}_{kernel}.csv')


    if all_models or 'SVC_rbf' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing SVC_rbf\n")
        print('='*80)


        def objective(trial, X, y, cv, random_state=42):

            C = trial.suggest_float('C', 1e-5, 10,log=True)
            kernel = trial.suggest_categorical('kernel', ['rbf'])
            degree = trial.suggest_int('degree', 1, 5) if kernel == 'poly' else 3 # Only used for 'poly' kernel
            # gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
            gamma = trial.suggest_float('gamma', 1e-5, 10, log=True)
            coef0 = trial.suggest_float('coef0', 0.0, 10.0) if kernel in ['poly', 'sigmoid'] else 0.0
            shrinking = trial.suggest_categorical('shrinking', [True, False])
            class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])
            probability = trial.suggest_categorical('probability', [True, False])

            #-------------------------------Model--------------------------------------#
            model = SVC(
                    C=C,
                    kernel=kernel,
                    degree=degree,
                    gamma=gamma,
                    coef0=coef0,
                    #shrinking=shrinking,
                    probability=probability,
                    #class_weight=class_weight,
                    random_state=random_state
                )

            #--------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)

            return np.mean(scores)
        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=n_trials, seed=seed)
        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            #-------------------------------------------------------------------------#
            model = SVC(**params, random_state=seed)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)
        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        kernel = model.kernel
        all_results.to_csv(f'{EXP}_{model_name}_{kernel}.csv')



    if all_models or 'DT' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing DT\n")
        print('='*80)

        def objective(trial, X, y, cv, random_state=42):
            criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
            splitter = trial.suggest_categorical('splitter', ['best', 'random'])
            max_depth = trial.suggest_int('max_depth', 3, 50)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 30)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 30)
            max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
            max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 10, 1000, log=True)
            min_impurity_decrease = trial.suggest_float('min_impurity_decrease', 0.0, 1.0)
            class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])


            #-------------------------------Model--------------------------------------#
            model = DecisionTreeClassifier(
                criterion=criterion,
                splitter=splitter,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                min_impurity_decrease=min_impurity_decrease,
                #class_weight=class_weight,
                random_state=random_state
            )
            #--------------------------------------------------------------------------#
            pipeline = create_pipeline(model, numerical_features, categorical_features)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)
            return np.mean(scores)
        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=500, seed=seed)
        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            #-------------------------------------------------------------------------#
            model = DecisionTreeClassifier(**params, random_state=seed)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)
        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        all_results.to_csv(f'{EXP}_{model_name}.csv')



    if all_models or 'Adaboost' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing Adaboost\n")
        print('='*80)

        def objective(trial, X, y, cv, random_state=42):

            n_estimators = trial.suggest_int('n_estimators', 10, 1000, log=True)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 3.0, log=True)
            estimator = DecisionTreeClassifier(
                max_depth=trial.suggest_int('base_max_depth', 3, 50),
                criterion=trial.suggest_categorical('criterion', ['gini', 'entropy']),
                splitter=trial.suggest_categorical('splitter', ['best', 'random']),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 30),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 30),
                max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                max_leaf_nodes=trial.suggest_int('max_leaf_nodes', 10, 1000, log=True),
                min_impurity_decrease=trial.suggest_float('min_impurity_decrease', 0.0, 1.0),
                class_weight=trial.suggest_categorical('class_weight', [None, 'balanced'])
            )

            #-------------------------------Model--------------------------------------#
            model = AdaBoostClassifier(
                estimator=estimator,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state
            )
            #--------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)

            return np.mean(scores)
        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=n_trials, seed=seed)
        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            ada_params = {
                'n_estimators': params['n_estimators'],
                'learning_rate': params['learning_rate']
            }
            dt_params = {
                'max_depth': params['base_max_depth'],
                'criterion': params['criterion'],
                'splitter': params['splitter'],
                'min_samples_split': params['min_samples_split'],
                'min_samples_leaf': params['min_samples_leaf'],
                'max_features': params['max_features'],
                'max_leaf_nodes': params['max_leaf_nodes'],
                'min_impurity_decrease': params['min_impurity_decrease'],
                #'class_weight': params['class_weight']
            }

            #-------------------------------------------------------------------------#
            estimator = DecisionTreeClassifier(**dt_params)
            model = AdaBoostClassifier(estimator=estimator, **ada_params, random_state=42)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)

        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        all_results.to_csv(f'{EXP}_{model_name}.csv')


    if all_models or 'RF' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing RF\n")
        print('='*80)


        def objective(trial, X, y, cv, random_state=42):

            n_estimators = trial.suggest_int('n_estimators', 10, 1000, log=True)
            max_depth = trial.suggest_int('max_depth', 3, 15)
            min_samples_split = trial.suggest_float('min_samples_split', 1e-5, 1.0, log=True)
            min_samples_leaf = trial.suggest_float('min_samples_leaf', 1e-5, 1.0, log=True)
            max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
            criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
            max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 10, 1000,log=True)

            #-------------------------------Model--------------------------------------#
            model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    criterion=criterion,
                    max_leaf_nodes=max_leaf_nodes,
                    random_state=random_state
                )
            #--------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)

            return np.mean(scores)
        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=n_trials, seed=seed)
        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            #-------------------------------------------------------------------------#
            model = RandomForestClassifier(**params, random_state=seed)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)

        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        all_results.to_csv(f'{EXP}_{model_name}.csv')

    if all_models or 'MLP' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing MLP\n")
        print('='*80)


        def objective(trial, X, y, cv, random_state=42):

            activation = trial.suggest_categorical('activation', ['relu'])
            solver = trial.suggest_categorical('solver', ['adam'])
            alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
            batch_size = trial.suggest_categorical('batch_size', [256])
            learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive'])
            max_iter = trial.suggest_int('max_iter', 200, 1000)
            learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 2e-2, log=True)

            #-------------------------------Model--------------------------------------#
            model = MLPClassifier(
                    activation=activation,
                    solver=solver,
                    alpha=alpha,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    max_iter=max_iter,
                    learning_rate_init=learning_rate_init,
                    random_state=random_state
                )
            #--------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)

            return np.mean(scores)
        best_params_dict, dfs = optimize_model_parameters(combinations, X, y, remap_labels, objective, cv=cv, n_trials=n_trials, seed=seed)
        all_results = []
        for idx, combination in enumerate(combinations):
            y_remapped = remap_labels(y, combination)
            study_name = f"Combo_{idx}"
            params = best_params_dict[study_name]

            #-------------------------------------------------------------------------#
            model = MLPClassifier(**params, random_state=seed)
            #-------------------------------------------------------------------------#

            pipeline = create_pipeline(model, numerical_features, categorical_features)
            process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)
        all_results = pd.DataFrame(all_results)
        model_name = type(model).__name__
        all_results.to_csv(f'{EXP}_{model_name}.csv')
    

    if all_models or 'XGB' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing XGB\n")
        print('='*80)


        def objective(trial, X, y, cv, random_state=42):

		    param = {
		        'objective': 'multi:softmax',  # For multiclass classification
		        'eval_metric': 'mlogloss',  # Multiclass logloss
		        'use_label_encoder': False,  # To avoid warning for deprecation
		        'booster': trial.suggest_categorical('booster', ['gbtree']),
		        'lambda': trial.suggest_float('lambda', 1e-5, 1.0, log=True),   
		        'alpha': trial.suggest_float('alpha', 1e-5, 1.0, log=True),   
		        'max_depth': trial.suggest_int('max_depth', 3, 15),
		        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1, log=True),
		        'n_estimators': trial.suggest_int('n_estimators', 100, 1500,log=True),
		        'subsample': trial.suggest_float('subsample', 0.1, 1.0),
		        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
		        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
		        'gamma': trial.suggest_float('gamma', 0, 5),   
		     
		    }
		    
		    #-------------------------------Model--------------------------------------#
		    model = xgb.XGBClassifier(**param, random_state=random_state,tree_method = "hist", verbosity=1)
		    #--------------------------------------------------------------------------#

		    pipeline = create_pipeline(model, numerical_features, categorical_features)
		    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)  

		    return np.mean(scores)

		best_params_dict, dfs = optimize_model_parameters(combinations_, X, y, remap_labels, objective, cv=cv, n_trials=n_trials, seed=seed)
		all_results = []
		for idx, combination in enumerate(combinations_):
		    y_remapped = remap_labels(y, combination)
		    study_name = f"Combo_{idx}"
		    params = best_params_dict[study_name]
		    
		    #-------------------------------------------------------------------------#
		    model = xgb.XGBClassifier(**params,
		                              objective="multi:softmax",
		                              eval_metric="mlogloss",
		                              use_label_encoder=False , 
		                              random_state=seed,
		                              tree_method = "hist", 
		                              verbosity=1)
		    #-------------------------------------------------------------------------#

		    pipeline = create_pipeline(model, numerical_features, categorical_features)
		    process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)
		all_results = pd.DataFrame(all_results)
		model_name = type(model).__name__
		all_results.to_csv(f'{EXP}_{model_name}.csv') 


    if all_models or 'LGBM' in selected_models:
        print('='*80)
        print(f"\n{EXP} for Optimizing LGBM\n")
        print('='*80)

		def objective(trial, X, y, cv, random_state=42):
		    
		    # LightGBM specific hyperparameters
		    param = {
		        'objective': 'multiclass',   
		        'metric': 'multi_logloss',   
		        'verbosity': -1,
		        #'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
		        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt']),
		        'num_leaves': trial.suggest_int('num_leaves', 100, 3000, log=True),
		        'max_depth': trial.suggest_int('max_depth', 3, 15),
		        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1, log=True),
		        'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
		        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
		        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 1.0,log=True),
		        'subsample': trial.suggest_float('subsample', 1e-3, 1.0, log=True),
		        'colsample_bytree': trial.suggest_float('colsample_bytree', 1e-3, 1.0, log=True),
		        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
		        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
		    }
		    
		    #-------------------------------Model--------------------------------------#
		    model = lgb.LGBMClassifier(**param, random_state=random_state)
		    #--------------------------------------------------------------------------#

		    pipeline = create_pipeline(model, numerical_features, categorical_features)
		    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)  

		    return np.mean(scores)
		best_params_dict, dfs = optimize_model_parameters(combinations_, X, y, remap_labels, objective, cv=cv, n_trials=n_trials, seed=seed)
		all_results = []
		for idx, combination in enumerate(combinations_):
		    y_remapped = remap_labels(y, combination)
		    study_name = f"Combo_{idx}"
		    params = best_params_dict[study_name]
		    
		    #-------------------------------------------------------------------------#
		    model = lgb.LGBMClassifier(**params, objective='multiclass', metric='multi_logloss', verbosity=-1, random_state=seed)
		    #-------------------------------------------------------------------------#

		    pipeline = create_pipeline(model, numerical_features, categorical_features)
		    process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results)

		all_results = pd.DataFrame(all_results)
		model_name = type(model).__name__
		all_results.to_csv(f'{EXP}_{model_name}.csv')





#-------------------------------------------------------------------------------#
#--------------------------------Experiment-------------------------------------#
#-------------------------------------------------------------------------------#
    print('='*80)
    print('='*80)
    print(f'\nExperiment {EXP} with {n_trials} trails is done\n')
    print('='*80)
    print('='*80)





