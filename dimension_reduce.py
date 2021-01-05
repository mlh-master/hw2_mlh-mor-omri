
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use(['ggplot'])
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def plt_2d_pca(X_pca, y):
    """
    plots 2D data after PCA
    :param X_pca : X_test after PCA fit
    :param  y: y test
    :return: scatter plot, colored red and blue
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    ax.scatter(X_pca[y==0, 0], X_pca[y==0, 1], color='b')
    ax.scatter(X_pca[y==1, 0], X_pca[y==1, 1], color='r')
    ax.legend(('Healthy', 'T1D Diagnosed'))
    ax.plot([0], [0], "ko")
    ax.arrow(0, 0, 0, 1, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
    ax.arrow(0, 0, 1, 0, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
    ax.set_xlabel('$U_1$')
    ax.set_ylabel('$U_2$')
    ax.set_title('2D PCA')
    plt.show()


def PCA_trans (X, X_train, X_test, y_test, scale_flag=True):
    """
    transforms data to PCA space
    :param X_train, X_test, y_test : train test set
    :param  X: the whole data
    :param scale_flag : true for scaling, false otherwise
    :return: X_train_pca, X_test_pca :  train test set after PCA
    """
    n_components = X.shape[1]  # 2D required
    pca = PCA(n_components=n_components, whiten=True)
    # apply PCA transformation
    if scale_flag:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_test_pca