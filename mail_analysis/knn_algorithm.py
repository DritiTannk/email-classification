import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, \
    precision_score, recall_score, f1_score, precision_recall_fscore_support


class KnnAlgorithmAnalysis:

    def knn_analysis(self, x, y_ds):
        knn_dict, stats_dict = {},{}
        for i in range(1, 4):
            train_size = 0.50
            if i == 1:
                train_size = 0.60
            if i == 2:
                train_size = 0.70
            if i == 3:
                train_size = 0.80

            x_train, x_test, y_train, y_test = train_test_split(x, y_ds, train_size=train_size, random_state=42)

            print(f'\n\n {"-" * 5} KNN Analysis {"-" * 5}')
            print('\n\n Train Size is ==> ', train_size)

            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(x_train, y_train)
            knn_y_pred = knn.predict(x_test)

            print('\n\n Prediction ==> ', knn_y_pred)

            knn_conf = confusion_matrix(y_test, knn_y_pred, labels=[0, 1])
            print('\n\n Confusion Matrix --> ', knn_conf)
            ConfusionMatrixDisplay(knn_conf,
                                   display_labels=knn.classes_).plot(
                                                                    xticks_rotation='horizontal',
                                                                    colorbar=False
                                                                    )
            file_name = f'conf_mat_{train_size}.png'

            knn_precision_rate = precision_score(y_test, knn_y_pred)
            print('\n\n PRECISION RATE ==> ', knn_precision_rate)

            knn_recall_rate = recall_score(y_test, knn_y_pred)
            print('\n\n RECALL RATE ==> ', knn_recall_rate)

            knn_f1_rate = f1_score(y_test, knn_y_pred)
            print('\n\n F1 SCORE  ==>  ', knn_f1_rate)

            knn_prf_rate = precision_recall_fscore_support(y_test, knn_y_pred)
            print('\n\n PRF RATE ==> ', knn_prf_rate)

            plt.savefig(f'output/knn/{file_name}')

            stats_dict['precision'] = knn_precision_rate
            stats_dict['recall'] = knn_recall_rate
            stats_dict['f1_score'] = knn_f1_rate

            knn_dict[train_size] = stats_dict

        return knn_dict