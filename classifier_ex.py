import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_blobs, make_classification
from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


# 설정
step_size = 0.2
color = plt.get_cmap('RdBu')
clf_name = "KNN"

# TODO: 분류기 선택
clf = SVC(kernel='rbf', C=10, gamma=0.1)

# 데이터 셋 생성
data_sets = [make_classification(n_features=2, n_classes=3, n_informative=2, n_redundant=0, random_state=1,
                                 n_clusters_per_class=1),
             make_moons(noise=0.3, random_state=1),
             make_circles(noise=0.2, factor=0.5, random_state=1),
             make_blobs(n_features=2, centers=3, cluster_std=1.0, random_state=1)]

# 결과를 보기 위한 그림창 생성
figure = plt.figure(figsize=(8, 8))

plt_idx = 0
# 각 데이터셋에 대하여...
for ds_idx, ds in enumerate(data_sets):
    X, y = ds

    # 훈련과 실험을 위한 데이터 셋 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

    # 첫 번째 서브 창 : input data - training set
    plt_idx += 1
    ax = plt.subplot(len(data_sets), 4, plt_idx)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=color, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    if ds_idx == 0:
        ax.set_title("Input data (training)")

    # 두 번째 서브 창 : input data - testing set
    plt_idx += 1
    ax = plt.subplot(len(data_sets), 4, plt_idx)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=color, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    if ds_idx == 0:
        ax.set_title("Input data (testing)")

    # TODO: 분류기 훈련
    clf.fit(X_train,y_train)

    # 만들어진 분류기 모델에 실험용 셋을 적용한 결과
    score = clf.score(X_test, y_test)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 세 번째 서브 창 : 훈련용 셋을 가지고 만든 분류기 모델 결과
    plt_idx += 1
    ax = plt.subplot(len(data_sets), 4, plt_idx)
    ax.contourf(xx, yy, Z, cmap=color, alpha=0.6)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=color, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    if ds_idx == 0:
        ax.set_title(clf_name + " (training)")

    # 네 번째 서브 창 : 만들어진 분류기 모델에 실험용 셋을 적용한 결과
    plt_idx += 1
    ax = plt.subplot(len(data_sets), 4, plt_idx)
    ax.contourf(xx, yy, Z, cmap=color, alpha=0.6)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=color, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    if ds_idx == 0:
        ax.set_title(clf_name + " (testing)")
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=15, horizontalalignment='right')

plt.tight_layout()
plt.show()
