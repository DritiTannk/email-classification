import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, \
    precision_score, recall_score, f1_score, precision_recall_fscore_support


class SVMAlgorithmAnalysis:
    def svm_analysis(self, x, y_ds):
        svm_dict, stats_dict = {}, {}
        for i in range(1, 4):
            train_size = 0.50

            if i == 1:
                train_size = 0.60
            if i == 2:
                train_size = 0.70
            if i == 3:
                train_size = 0.80

            x_train, x_test, y_train, y_test = train_test_split(x, y_ds,
                                                                train_size=train_size,
                                                                random_state=42)

            print(f'\n\n {"-" * 5} SVM Analysis {"-" * 5}')
            print('\n\n Train Size is ==> ', train_size)

            svm = SVC(kernel='linear')
            svm.fit(x_train, y_train)
            svm_y_pred = svm.predict(x_test)

            print('\n\n Prediction \n\n', svm_y_pred)

            svm_conf = confusion_matrix(y_test, svm_y_pred, labels=[0, 1])
            ConfusionMatrixDisplay(svm_conf,
                                   display_labels=svm.classes_).plot(
                                                                    cmap='binary',
                                                                    xticks_rotation='horizontal',
                                                                    colorbar=False
                                                                    )
            file_name = f'conf_mat_{train_size}.png'
            print('\n\n Confusion Matrix --> ', svm_conf)

            svm_precision_rate = precision_score(y_test, svm_y_pred)
            print('\n\n PRECISION RATE ==> ', svm_precision_rate)

            svm_recall_rate = recall_score(y_test, svm_y_pred)
            print('\n\n RECALL RATE ==> ', svm_recall_rate)

            svm_f1_rate = f1_score(y_test, svm_y_pred)
            print('\n\n F1 SCORE  ==>  ', svm_f1_rate)

            svm_prf_rate = precision_recall_fscore_support(y_test, svm_y_pred)
            print('\n\n PRF RATE ==> ', svm_prf_rate)

            plt.savefig(f'output/svm/{file_name}')

            stats_dict['precision'] = svm_precision_rate
            stats_dict['recall'] = svm_recall_rate
            stats_dict['f1_score'] = svm_f1_rate

            svm_dict[train_size] = stats_dict

        return svm_dict