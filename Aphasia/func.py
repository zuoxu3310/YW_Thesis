import numpy as np
import pandas as pd
import yaml
import random
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
import optuna
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, make_scorer
from sklearn.utils.class_weight import compute_class_weight

def weighted_balanced_accuracy(y_true, y_pred):
    # Unique classes in the true labels
    classes = np.unique(y_true)
    # Compute class weights based on the frequency of each class
    class_weights = compute_class_weight('balanced', classes=classes, y=y_true)
    # Mapping class weights to target classes
    class_weight_dict = dict(zip(classes, class_weights))
    # Compute recall for each class
    recalls = recall_score(y_true, y_pred, labels=classes, average=None)
    # Weighted sum of recalls
    weighted_recalls = recalls * np.array([class_weight_dict[cls] for cls in classes])
    # Sum of weights
    sum_of_weights = np.sum(class_weights)
    # Weighted Balanced Accuracy
    weighted_balanced_accuracy = np.sum(weighted_recalls) / sum_of_weights
    return weighted_balanced_accuracy

def fbeta_macro(y_true, y_pred, beta=1):
    return fbeta_score(y_true, y_pred, beta=beta, average='macro')

def remap_labels(y, combos):
    remapped_y = y.copy()
    for i, combo in enumerate(combos):
        for label in combo:
            remapped_y[y == label] = i
    return remapped_y

def optimize_model_parameters(combinations, X, y, remap_labels, objective, cv, n_trials, seed=42):
    dfs = {}
    drop_id = ['datetime_start', 'datetime_complete', 'duration', 'state']
    best_params_dict = {}

    for idx, combination in enumerate(combinations):
        y_remapped = remap_labels(y, combination)
        study_name = f"Combo_{idx}"
        print('='*80)
        print(f'\n{idx}-{combination}-{study_name} started\n')
        print('='*80)
        #optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize', study_name=study_name)
        study.optimize(lambda trial: objective(trial, X, y_remapped, cv=cv, random_state=seed), n_trials=n_trials)
        dfs[study_name] = study.trials_dataframe().drop(drop_id, axis=1)

        best_params_dict[study_name] = study.best_params
        print(f'\nBest params of {study_name}\n {best_params_dict[study_name]}\n')
    return best_params_dict, dfs

def compute_scores(pipeline, X, y_remapped, cv, scoring_metrics):
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   
        scores = cross_validate(pipeline, X, y_remapped, cv=cv, scoring=scoring_metrics)
        
    mean_scores = {metric: np.mean(scores[f'test_{metric}']) for metric in scoring_metrics.keys()}
    std_scores = {metric: np.std(scores[f'test_{metric}']) for metric in scoring_metrics.keys()}
    formatted_scores = {metric: f"{mean_scores[metric]:.3f} ± {std_scores[metric]:.3f}" for metric in scoring_metrics.keys()}
    return formatted_scores

def process_results(pipeline, X, y_remapped, cv, scoring_metrics, idx, combination, params, all_results, plot=False):
    formatted_scores = compute_scores(pipeline, X, y_remapped, cv, scoring_metrics)

    if plot == True:
        sum_conf_matrix = np.zeros((len(combination), len(combination)))

        for train_index, test_index in cv.split(X, y_remapped):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y_remapped[train_index], y_remapped[test_index]
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            conf_matrix = confusion_matrix(y_test, y_pred)
            sum_conf_matrix += conf_matrix

        avg_conf_matrix = sum_conf_matrix / 5
        plt.figure(figsize=(8, 6),dpi=300)
        sns.heatmap(avg_conf_matrix, annot=True, fmt=".2f", cmap="Blues", cbar=False)
        plt.title(f"{idx}-{combination}-Average Confusion Matrix")
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.show()

    all_results.append({
        'idx': idx,
        'combination': combination,
        **formatted_scores,
        'best_params': params
    })

def plot_cv_results(all_results, scoring):
    modified_results = pd.DataFrame()
    if callable(scoring):
        scoring_name = 'weighted_balanced_accuracy'
    else:
        scoring_name = scoring
    modified_results['idx'] = all_results['idx']
    modified_results[f'mean_{scoring_name}'] = all_results[scoring_name].apply(lambda x: float(x.split(' ± ')[0]))
    modified_results[f'std_{scoring_name}'] = all_results[scoring_name].apply(lambda x: float(x.split(' ± ')[1]))

    sns.barplot(data=modified_results, x='idx', y=f'mean_{scoring_name}',  errorbar='sd')
    plt.errorbar(modified_results['idx'], modified_results[f'mean_{scoring_name}'], yerr=modified_results[f'std_{scoring_name}'], fmt='none', color='black')
    plt.title(f'CV Score for {scoring_name}')
    plt.xlabel('Index')
    plt.ylabel('Mean Score')
    plt.show()