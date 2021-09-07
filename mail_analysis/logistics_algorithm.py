import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, \
    precision_score, recall_score, f1_score, precision_recall_fscore_support


class LogisticAlgorithmAnalysis:
    def log_analysis(self, x, y_ds):
        log_dict, stats_dict = {}, {}

        for i in range(1, 4):
            train_size = 0.50
            if i == 1:
                train_size = 0.60
            if i == 2:
                train_size = 0.70
            if i == 3:
                train_size = 0.80

            x_train, x_test, y_train, y_test = train_test_split(x, y_ds, train_size=train_size, random_state=42)

            print(f'\n\n {"-" * 5} Logistics Analysis {"-" * 5}')
            print('\n\n Train Size is ==> ', train_size)

            log_algo = LogisticRegression()
            log_algo.fit(x_train, y_train)
            log_y_pred = log_algo.predict(x_test)

            print('\n\n Prediction ==> ', log_y_pred)

            result = confusion_matrix(y_test,log_y_pred, labels=[0, 1])
            print('\n\n result --> ', result)
            ConfusionMatrixDisplay(result, display_labels=log_algo.classes_).plot(cmap='Blues', xticks_rotation='horizontal', colorbar=False)
            file_name = f'conf_mat_{train_size}.png'

            precision_rate = precision_score(y_test, log_y_pred)
            print('\n\n PRECISION RATE ==> ', precision_rate)

            recall_rate = recall_score(y_test, log_y_pred)
            print('\n\n RECALL RATE ==> ', recall_rate)

            f1_rate = f1_score(y_test, log_y_pred)
            print('\n\n F1 SCORE  ==>  ', f1_rate)

            prf_rate = precision_recall_fscore_support(y_test, log_y_pred)
            print('\n\n PRF RATE ==> ', prf_rate)

            plt.savefig(f'output/logistics/{file_name}')

            stats_dict['precision'] = precision_rate
            stats_dict['recall'] = recall_rate
            stats_dict['f1_score'] = f1_rate

            log_dict[train_size] = stats_dict

        return log_dict
