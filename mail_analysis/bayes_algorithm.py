import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, \
                            precision_score, recall_score, \
                            f1_score, precision_recall_fscore_support


class BayesAlgorithmAnalysis:
    def bayes_analysis(self, x, y_ds):
        bayes_dict = {}
        stats_dict = {}

        for i in range(1, 4):
            train_size = 0.50
            if i == 1:
                train_size = 0.60
            if i == 2:
                train_size = 0.70
            if i == 3:
                train_size = 0.80

            x_train, x_test, y_train, y_test = train_test_split(x.toarray(), y_ds, train_size=train_size, random_state=42)

            print(f'\n\n {"-" * 5} Bayes Analysis {"-" * 5}')
            print('\n\n Train Size is ==> ', train_size)

            bayes = GaussianNB()
            bayes.fit(x_train, y_train)
            bayes_y_pred = bayes.predict(x_test)

            print('\n\n Prediction ==> ', bayes_y_pred)

            bayes_conf = confusion_matrix(y_test, bayes_y_pred, labels=[0, 1])
            print('\n\n Confusion Matrix Result  --> \n\n', bayes_conf)

            ConfusionMatrixDisplay(bayes_conf, display_labels=bayes.classes_).plot(cmap='Blues', xticks_rotation='horizontal', colorbar=False)
            file_name = f'conf_mat_{train_size}.png'
            plt.savefig(f'output/bayes/{file_name}')

            bayes_precision_rate = precision_score(y_test, bayes_y_pred)
            print('\n\n PRECISION RATE ==> ', bayes_precision_rate)

            bayes_recall_rate = recall_score(y_test, bayes_y_pred)
            print('\n\n RECALL RATE ==> ', bayes_recall_rate)

            bayes_f1_rate = f1_score(y_test, bayes_y_pred)
            print('\n\n F1 SCORE  ==>  ', bayes_f1_rate)

            bayes_prf_rate = precision_recall_fscore_support(y_test, bayes_y_pred)
            print('\n\n PRF RATE ==> ', bayes_prf_rate)

            stats_dict['precision'] = bayes_precision_rate
            stats_dict['recall'] = bayes_recall_rate
            stats_dict['f1_score'] = bayes_f1_rate

            bayes_dict[train_size] = stats_dict

        return bayes_dict








