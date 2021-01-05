import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
mpl.style.use(['ggplot'])
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, plot_confusion_matrix, roc_auc_score, plot_roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier



def check_penalty(penalty='none'):
    """
    fitting solver to penalty
    :param penalty: penalty type chosen. ['l1', 'l2', 'none']
    :return: solver
    """
    # from tutorial 5
    if penalty == 'l1':
        solver='liblinear'
    if penalty == 'l2' or penalty == 'none':
        solver='lbfgs'
    return solver


def LogReg_CrossVal(n_splits, pen, lmbda, X_train, X_test, Y_train, y_test):
    """
    k-cross-validation using logistic regression and classifier building
    :param X_train, X_test, Y_train, y_test: train-test splitted data
    :param n_splits : number of splitted segmnets (validation and test) from train set
    :return: best_clf, clf (classifier)
    """
    verbose = 0
    skf = StratifiedKFold(n_splits=n_splits, random_state=10, shuffle=True)
    max_iter = 2000
    solver = check_penalty(penalty=pen)
    log_reg = LogisticRegression(random_state=5, max_iter=max_iter, solver=solver)
    pipe = Pipeline(steps=[('logistic', log_reg)])  # ('scale', StandardScaler()),
    # when working with pipe line we need to be specify with params names _
    clf = GridSearchCV(estimator=pipe, param_grid={'logistic__C': 1 / lmbda, 'logistic__penalty': [pen]},
                       scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'], cv=skf,
                       refit='roc_auc', verbose=verbose, return_train_score=True)  # best estimator by refit param
    # TODO: loss calc (‘neg_root_mean_squared_error’ ?)
    clf.fit(X_train, Y_train)
    best_clf = clf.best_estimator_

    y_pred_test = best_clf.predict(X_test)
    y_pred_proba_test = best_clf.predict_proba(X_test)

    Classifier = 'log_reg'
    model_performance(best_clf, X_test, y_test, y_pred_test, y_pred_proba_test, Classifier)
    J_loss = log_loss(y_test, y_pred_proba_test)
    print('log loss is: ', f'{J_loss}')
    print('with params: ', f'{clf.best_params_}')
    return best_clf, clf


def C_Support_Vector_Classification(X_train, X_test, y_train, y_test, n_splits=5, Classifier='rbf'):
    """
    k-cross-validation using SVM and classifier building
    :param X_train, X_test, Y_train, y_test: train-test splitted data
    :param n_splits : number of splitted segmnets (validation and test) from train set
    :param Classifier : kernel for SVM
    :return: best_svm (classifier)
             + plots model performance (confusion matrix) + radar plot
    """
    verbose = 0
    skf = StratifiedKFold(n_splits=n_splits, random_state=15, shuffle=True)
    svc = SVC(probability=True)
    C = np.array([0.01, 1, 10, 100])
    pipe = Pipeline(steps=[('svm', svc)])
    if Classifier == 'linear':
        svm = GridSearchCV(estimator=pipe,
                           param_grid={'svm__kernel': [Classifier], 'svm__C': C},
                           scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'],
                           cv=skf, refit='roc_auc', verbose=verbose, return_train_score=True)
        clf_type = ['linear']
    if Classifier == 'rbf' or Classifier == 'poly':
        svm = GridSearchCV(estimator=pipe,
                           param_grid={'svm__kernel': [Classifier], 'svm__C': C, 'svm__degree': [3],
                                       'svm__gamma': ['auto', 'scale']},
                           scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'],
                           cv=skf, refit='roc_auc', verbose=verbose, return_train_score=True)
        clf_type = [Classifier, 'scale']

    svm.fit(X_train, y_train)

    best_svm = svm.best_estimator_
    print(best_svm)

    y_pred_test = best_svm.predict(X_test)
    y_pred_proba_test = best_svm.predict_proba(X_test)
    model_performance(best_svm, X_test, y_test, y_pred_test, y_pred_proba_test, Classifier)

    # '''
    # Radar function:
    plot_radar_svm(svm, clf_type)
    plt.grid(False)
    # '''

    print('C Support Vector Classification -> Done')
    return best_svm


def Random_forest_classifier(X_train, X_test, y_train, y_test):
    """
    random forest classifier building
    :param X_train, X_test, Y_train, y_test: train-test splitted data
    :return: rfc (classifier)
             + plots model performance (confusion matrix) + radar plot
    """
    rfc = RandomForestClassifier(max_depth=4, random_state=0, criterion='gini')
    rfc.fit(X_train, y_train)
    y_pred_test = rfc.predict(X_test)
    y_pred_proba_test = rfc.predict_proba(X_test)
    Classifier = 'rfc'
    model_performance(rfc, X_test, y_test, y_pred_test, y_pred_proba_test, Classifier)

    return rfc


def features_select_rfc(rfc, names):
    """
    plots features importance in bar-plot
    :param rfc : trained classifier (random forest)
    :param names : features names list
    :return: plot features importance + display 2 most important features
    """
    importance_feat_zip = sorted(zip(map(lambda x: round(x, 4), rfc.feature_importances_), names))
    importance_feat_zip.reverse()
    print("Features sorted by their score:")
    print(importance_feat_zip)
    sorted_names = [x[1] for x in importance_feat_zip]
    importance_vals = [x[0] for x in importance_feat_zip]
    sns.barplot(x=sorted_names, y=importance_vals)
    plt.rcParams['xtick.labelsize'] = 9
    plt.title('Features Importance - RFC')
    plt.show()
    print('best 2 features by rfc are: ', sorted_names[0], ' and ', sorted_names[1])


def compare_classifiers_AUC(classifiers, classifiers_str, x_test, y_test):
    """
    compares different classifiers by AUC test
    :param classifiers : list of classifiers
    :param classifiers_str : list of classifiers names
    :param x_test, y_test: test set
    :return: TP + FN graph (ROC) of all classifiers
    """
    roc_score = np.zeros([len(classifiers_str), 1])
    legend_txt = []
    plt.figure()
    ax = plt.gca()
    for idx, clf in enumerate(classifiers):
        plot_roc_curve(clf, x_test, y_test, ax=ax)
        roc_score[idx] = np.round_(roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1]), decimals=3)
        txt = [classifiers_str[idx] + str(roc_score[idx])]
        legend_txt += txt

    legend_txt += ['flipping a coin']
    ax.plot(np.linspace(0, 1, x_test.shape[0]), np.linspace(0, 1, x_test.shape[0]))
    plt.legend(legend_txt)
    #plt.legend((classifiers_str[0] + str(roc_score[0]), classifiers_str[1] + str(roc_score[1]),
    #           classifiers_str[2] + str(roc_score[2]), 'flipping a coin'))
    plt.show()


def model_performance(model, X_test, y_test, y_pred_test, y_pred_proba_test, Classifier):
    """
    confusion matrix and statistics of the classifier
    :param Classifier : trained classifier
    :param y_pred_test, y_pred_proba_test : prediction set
    :param  X_test, y_test: test set
    :return: confusion matrix plot and statistics printed
    """
    calc_TN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 0]
    calc_FP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 1]
    calc_FN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 0]
    calc_TP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 1]

    plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues)
    plt.show()

    TN = calc_TN(y_test, y_pred_test)
    FP = calc_FP(y_test, y_pred_test)
    FN = calc_FN(y_test, y_pred_test)
    TP = calc_TP(y_test, y_pred_test)
    Se = TP / (TP + FN)
    Sp = TN / (TN + FP)
    PPV = TP / (TP + FP)
    NPV = TN / (TN + FN)
    Acc = (TP + TN) / (TP + TN + FP + FN)
    F1 = (2 * Se * PPV) / (Se + PPV)
    print(Classifier, ':')
    print('Sensitivity is {:.2f} \nSpecificity is {:.2f} \nPPV is {:.2f} \nNPV is {:.2f} \nAccuracy is {:.2f} \n'
          'F1 is {:.2f} '.format(
        Se, Sp, PPV, NPV, Acc, F1))
    print('AUROC is {:.3f}'.format(roc_auc_score(y_test, y_pred_proba_test[:, 1])))


def plot_radar_logReg(clf, lmbda, clf_type):
    """
    radar plot of log-reg classifier
    :param clf : trained classifier (LogReg)
    :param lmbda : list of lambdas (1/penalty)
    :param  clf_type: classifier type (log reg)
    :return: radar plot
    """
    labels = np.array(['Accuracy', 'F1', 'PPV', 'Sensitivity', 'AUROC'])
    score_mat_train = np.stack((clf.cv_results_['mean_train_accuracy'], clf.cv_results_['mean_train_f1'],
                               clf.cv_results_['mean_train_precision'], clf.cv_results_['mean_train_recall'],
                               clf.cv_results_['mean_train_roc_auc']), axis=0)
    score_mat_val = np.stack((clf.cv_results_['mean_test_accuracy'], clf.cv_results_['mean_test_f1'],
                               clf.cv_results_['mean_test_precision'], clf.cv_results_['mean_test_recall'],
                               clf.cv_results_['mean_test_roc_auc']), axis=0)


    angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    # close the plot

    angles=np.concatenate((angles, [angles[0]]))
    cv_dict = clf.cv_results_['params']
    fig=plt.figure(figsize=(18,14))
    for idx, loc in enumerate(cv_dict):
        ax = fig.add_subplot(1, len(lmbda), 1+idx, polar=True)
        stats_train = score_mat_train[:, idx]
        stats_train=np.concatenate((stats_train,[stats_train[0]]))
        ax.plot(angles, stats_train, 'o-', linewidth=2)
        ax.fill(angles, stats_train, alpha=0.25)
        stats_val = score_mat_val[:, idx]
        stats_val=np.concatenate((stats_val,[stats_val[0]]))
        ax.plot(angles, stats_val, 'o-', linewidth=2)
        ax.fill(angles, stats_val, alpha=0.25)
        ax.set_thetagrids(angles[0:-1] * 180/np.pi, labels)
        if idx == 0:
            ax.set_ylabel('$L_2$', fontsize=18)
        if cv_dict[idx]['logistic__C'] <= 1:
            ax.set_title('$\lambda$ = %d'  % (1 / cv_dict[idx]['logistic__C']))
        else:
            ax.set_title('$\lambda$ = %.3f' % (1 / cv_dict[idx]['logistic__C']))
        ax.set_ylim([0,1])
        ax.legend(['Train','Validation'])
        ax.grid(True)
    plt.show()


def plot_radar_svm(clf, clf_type):
    """
    radar plot of SVM classifier
    :param clf : trained classifier (SVM)
    :param  clf_type: classifier type (SVM)
    :return: radar plot
    """
    labels = np.array(['Accuracy', 'F1', 'PPV', 'Sensitivity', 'AUROC'])
    score_mat_train = np.stack((clf.cv_results_['mean_train_accuracy'], clf.cv_results_['mean_train_f1'],
                                clf.cv_results_['mean_train_precision'], clf.cv_results_['mean_train_recall'],
                                clf.cv_results_['mean_train_roc_auc']), axis=0)
    score_mat_val = np.stack((clf.cv_results_['mean_test_accuracy'], clf.cv_results_['mean_test_f1'],
                              clf.cv_results_['mean_test_precision'], clf.cv_results_['mean_test_recall'],
                              clf.cv_results_['mean_test_roc_auc']), axis=0)

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)

    angles = np.concatenate((angles, [angles[0]]))
    cv_dict = clf.cv_results_['params']
    fig = plt.figure(figsize=(18, 14))
    if 'svm__gamma' in cv_dict[0]:
        new_list = [(i, item) for i, item in enumerate(cv_dict) if
                    item["svm__kernel"] == clf_type[0] and item["svm__gamma"] == clf_type[1]]
    else:
        new_list = [(i, item) for i, item in enumerate(cv_dict) if
                    item["svm__kernel"] == clf_type[0]]
    for idx, val in enumerate(new_list):
        ax = fig.add_subplot(1, len(new_list), 1 + idx, polar=True)
        rel_idx, rel_dict = val
        stats_train = score_mat_train[:, rel_idx]
        stats_train = np.concatenate((stats_train, [stats_train[0]]))
        ax.plot(angles, stats_train, 'o-', linewidth=2)
        ax.fill(angles, stats_train, alpha=0.25)
        stats_val = score_mat_val[:, rel_idx]
        stats_val = np.concatenate((stats_val, [stats_val[0]]))
        ax.plot(angles, stats_val, 'o-', linewidth=2)
        ax.fill(angles, stats_val, alpha=0.25)
        ax.set_thetagrids(angles[0:-1] * 180 / np.pi, labels)
        if idx == 0:
            ax.set_ylabel(clf_type[0], fontsize=18)
        ax.set_title('C = %.3f' % (rel_dict['svm__C']))
        if 'svm__gamma' in cv_dict[0]:
            ax.set_xlabel('$\gamma = %s $' % (rel_dict['svm__gamma']))
        ax.set_ylim([0, 1])
        ax.legend(['Train', 'Validation'])
        ax.grid(True)



if __name__ =='__main__':

    from pathlib import Path
    from clean_data import one_hot_vectors, fix_values, to_numeric

    file = Path.cwd().joinpath('HW2_data.csv')
    T1D_data = pd.read_csv(file, thousands=',')
    T1D_data_numeric = to_numeric(T1D_data)
    T1D_data_clean, Diagnosis = fix_values(T1D_data_numeric, flag='fv')

    T1D_data_oneHotVecs = one_hot_vectors(T1D_data_clean)

    X_train, X_test, y_train, y_test = train_test_split(T1D_data_oneHotVecs, np.ravel(Diagnosis), test_size=0.2,
                                                        random_state=0, stratify=np.ravel(Diagnosis))
    #T1D_data_oneHotVecs = one_hot_vectors(T1D_data_clean)

    # logistic regression model
    pen = 'l2'  # 'none'
    n_splits = 5
    lmbda = np.array([0.001, 0.01, 1, 10, 100, 1000])
    chosen_clf, clf = LogReg_CrossVal(n_splits, pen, lmbda, X_train, X_test, y_train, y_test)

    clf_type = 'log_reg'
    plot_radar_logReg(clf, lmbda, clf_type)

    #C_Support_Vector_Classification:
    best_svm_lin = C_Support_Vector_Classification(X_train, X_test, y_train, y_test, n_splits=5, Classifier='linear')
    best_svm_non_lin = C_Support_Vector_Classification(X_train, X_test, y_train, y_test, n_splits=5, Classifier='rbf')

    # 6. random forest
    rfc = Random_forest_classifier(X_train, X_test, y_train, y_test)

    names = pd.DataFrame(T1D_data_oneHotVecs.columns)[0]
    features_select_rfc(rfc, names)


    classifiers = [best_svm_lin, best_svm_non_lin, rfc]
    classifiers_str = ['svm_lin', 'svm_non_lin', 'rfc']
    classifiers_str = [s + ', AUROC = ' for s in classifiers_str]
    compare_classifiers_AUC(classifiers, classifiers_str, X_test, y_test)
