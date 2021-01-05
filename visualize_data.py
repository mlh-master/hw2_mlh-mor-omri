import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def table_visualize_test_train(T1D_features_names, X_train, X_test, class_feature):
    """
    :param T1D_features_names: all features names
    :param X_train, X_test: the train and test segments
    :param class_feature: test feature name not included in learning
    :return: None (table plot)
    """
    columns = ('Train %', 'Test %', 'Delta %')
    feat_names = T1D_features_names[1:]
    class_feature_index = pd.Index(feat_names).get_loc(class_feature)
    rowLabels = list(feat_names.drop(class_feature_index + 1))  # 16 = index of Diagnosis
    cell_text = []
    for (feat_train, feat_test) in zip(X_train.loc[:, rowLabels], X_test.loc[:, rowLabels]):
        num_yes_train = X_train.loc[:, feat_train].sum()
        num_yes_test = X_test.loc[:, feat_test].sum()
        feat_len_train = len(X_train.loc[:, feat_train])
        feat_len_test = len(X_test.loc[:, feat_test])
        train_pos_precent = round((num_yes_train / feat_len_train) * 100)
        test_pos_precent = round((num_yes_test / feat_len_test) * 100)
        delta = train_pos_precent - test_pos_precent
        cell_text.append(
            ['{0:.0f}'.format(train_pos_precent), '{0:.0f}'.format(test_pos_precent), '{0:.0f}'.format(delta)])

    test_train_positive_table = plt.table(cellText=cell_text,
                                          rowLabels=rowLabels,
                                          colLabels=columns,
                                          colLoc='center',
                                          )
    test_train_positive_table.auto_set_font_size(False)
    test_train_positive_table.set_fontsize(8)
    plt.axis("off")
    plt.grid(False)
    plt.show()


def feature_frequency(features_, feature, class_feature, i, j, axes):
    """
    plot frequency of features according to class_feature
    :param feature: specific feature name
    :param features_:
    :param class_feature: test feature not included in learning name
    :return: None (bar plot)
    """
    sns.countplot(ax=axes[i, j], x=feature,
                  hue=class_feature,
                  data=features_
                  )
    plt.ylabel("count", size=14)
    plt.xlabel(f'{feature}', size=14)
    plt.title(f'{class_feature} by {feature}', size=18)

    if j < 3:
        j = j + 1
    else:
        i = i + 1
        j = 0
    return i, j, axes
