import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from clean_data import remove_nan, fill_values, fix_values, to_numeric, one_hot_vectors
from visualize_data import table_visualize_test_train, feature_frequency
from optimize_models import LogReg_CrossVal, plot_radar_logReg, C_Support_Vector_Classification,\
                            Random_forest_classifier, compare_classifiers_AUC, features_select_rfc
from dimension_reduce import plt_2d_pca, PCA_trans


if __name__ =='__main__':
    # load data
    file = Path.cwd().joinpath('HW2_data.csv')
    T1D_data = pd.read_csv(file, thousands=',')
    # exchange 'Yes,'Positive, 'No','Negative to numeric data (1,0)
    T1D_data_numeric = to_numeric(T1D_data)
    # extract features names
    T1D_features_names = pd.DataFrame(T1D_data_numeric.columns)[0]
    '''
    T1D_features = T1D_data_numeric[['Age', 'Gender', 'Increased Urination', 'Increased Thirst', 'Sudden Weight Loss', 'Weakness',
                             'Increased Hunger', 'Genital Thrush', 'Visual Blurring', 'Itching', 'Irritability',
                             'Delayed Healing', 'Partial Paresis', 'Muscle Stiffness', 'Hair Loss',
                             'Obesity', 'Family History']]
    # Diagnosis = T1D_data_numeric[['Diagnosis']]

    # remove nan values
    T1D_data_clean = remove_nan(T1D_features)
    # TODO: diagnosis is not get cleaned - feature length not fit

    # or!
    # change nan values to random value by probability disturbution
    T1D_data_filled = fill_values(T1D_features)
    '''

    T1D_data_clean, Diagnosis = fix_values(T1D_data_numeric, flag='fv')
    # split data to train and test
    # orig_feat = T1D_features.columns.values
    X_train, X_test, y_train, y_test = train_test_split(T1D_data_clean, np.ravel(Diagnosis), test_size=0.2,
                                                        random_state=0, stratify=np.ravel(Diagnosis))
    # a. visualize train and test into table
    class_feature = 'Diagnosis'
    table_visualize_test_train(T1D_features_names, X_train, X_test, class_feature)

    # b. plot frequency of features according to Diagnosis
    T1D_data_clean_full = pd.concat([T1D_data_clean, Diagnosis], axis=1)

    fig, axes = plt.subplots(4, 4)
    i = 0
    j = 0
    for idx, feat in enumerate(T1D_data_clean):
        if feat == 'Age':
            continue
        i, j, axes = feature_frequency(T1D_data_clean_full, feat, class_feature, i, j, axes)
    plt.show()

    # c. additional plots
    # age, Family History, diagnosis
    sns.scatterplot(data=T1D_data_clean_full, x='Age', y='Diagnosis', hue='Family History', style='Diagnosis')
    plt.show()
    # age, gender, diagnosis
    sns.displot(data=T1D_data_clean_full, x='Age', y='Diagnosis', hue='Gender')
    plt.show()

    # 4. encoding data as one hot vectors (0,1)
    T1D_data_oneHotVecs = one_hot_vectors(T1D_data_clean)

    # 5. choose, build and optimize ML models
    # a 5K cross fold validation for tune so -> AUC highest test
    X_train, X_test, y_train, y_test = train_test_split(T1D_data_oneHotVecs, np.ravel(Diagnosis), test_size=0.2,
                                                        random_state=0, stratify=np.ravel(Diagnosis))
    n_splits = 5

    # logistic regression model
    pen = 'l2'  # 'none'
    lmbda = np.array([0.001, 0.01, 1, 10, 100, 1000])
    chosen_clf, clf = LogReg_CrossVal(n_splits, pen, lmbda, X_train, X_test, y_train, y_test)

    clf_type = 'log_reg'
    plot_radar_logReg(clf, lmbda, clf_type)

    # C_Support_Vector_Classification:
    best_svm_lin = C_Support_Vector_Classification(X_train, X_test, y_train, y_test, n_splits=5, Classifier='linear')
    best_svm_non_lin = C_Support_Vector_Classification(X_train, X_test, y_train, y_test, n_splits=5, Classifier='rbf')

    # random forest
    rfc = Random_forest_classifier(X_train, X_test, y_train, y_test)


    # features selection
    names = pd.DataFrame(T1D_data_oneHotVecs.columns)[0]
    features_select_rfc(rfc, names)

    # comparison of classifiers
    classifiers = [best_svm_lin, best_svm_non_lin, rfc]
    classifiers_str = ['svm_lin', 'svm_non_lin', 'rfc']
    classifiers_str = [s + ', AUROC = ' for s in classifiers_str]
    compare_classifiers_AUC(classifiers, classifiers_str, X_test, y_test)

    # 7. dimensions reduction
    # a. dimensionality reduction on the dataset so that you can plot your data in a 2d plot
    X_train_pca, X_test_pca = PCA_trans(T1D_data_oneHotVecs, X_train, X_test, y_test, scale_flag=True)
    plt_2d_pca(X_test_pca[:, 0:2], y_test)

    # c. training models on dimensionality reduces training set:
    chosen_clf_pca, clf_pca = LogReg_CrossVal(n_splits, pen, lmbda, X_train_pca, X_test_pca, y_train, y_test)
    best_svm_lin_pca = C_Support_Vector_Classification(X_train_pca, X_test_pca, y_train, y_test, n_splits=5, Classifier='linear')
    best_svm_non_lin_pca = C_Support_Vector_Classification(X_train_pca, X_test_pca, y_train, y_test, n_splits=5, Classifier='rbf')
    rfc_pca = Random_forest_classifier(X_train_pca, X_test_pca, y_train, y_test)

    classifiers_pca = [best_svm_lin_pca, best_svm_non_lin_pca, rfc_pca]
    classifiers_pca_str = ['svm_lin_pca', 'svm_non_lin_pca', 'rfc_pca']
    classifiers_pca_str = [s + ', AUROC = ' for s in classifiers_str]
    compare_classifiers_AUC(classifiers_pca, classifiers_pca_str, X_test_pca, y_test)

    # d. training models on the best 2 features from, section 6
    # Increased Urination  and  Increased Thirst
    X_train_2_feat = pd.concat([X_train['Increased Urination'], X_train['Increased Thirst']], axis=1)
    X_test_2_feat = pd.concat([X_test['Increased Urination'], X_test['Increased Thirst']], axis=1)

    chosen_clf_2feat, clf_2feat = LogReg_CrossVal(n_splits, pen, lmbda, X_train_2_feat, X_test_2_feat, y_train, y_test)
    best_svm_lin_2feat = C_Support_Vector_Classification(X_train_2_feat, X_test_2_feat, y_train, y_test, n_splits=5, Classifier='linear')
    best_svm_non_lin_2feat = C_Support_Vector_Classification(X_train_2_feat, X_test_2_feat, y_train, y_test, n_splits=5, Classifier='rbf')
    rfc_2feat = Random_forest_classifier(X_train_2_feat, X_test_2_feat, y_train, y_test)

    classifiers_2feat = [best_svm_lin_2feat, best_svm_non_lin_2feat, rfc_2feat]
    classifiers_2feat_str = ['svm_lin_2feat', 'svm_non_lin_2feat', 'rfc_2feat']
    classifiers_2feat_str = [s + ', AUROC = ' for s in classifiers_str]
    compare_classifiers_AUC(classifiers_2feat, classifiers_2feat_str, X_test_2_feat, y_test)

    print('done')