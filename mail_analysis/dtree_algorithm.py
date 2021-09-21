import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, \
                     precision_score, recall_score, f1_score, \
                     precision_recall_fscore_support, accuracy_score


class DecisionTreeAnalysis:
    def dtree_Analysis(self, x, y_ds):
        stats_dict, dtree_dict = {}, {}

        for i in range(1, 4):
            train_size = 0.50
            if i == 1:
                train_size = 0.60
            if i == 2:
                train_size = 0.70
            if i == 3:
                train_size = 0.80

            x_train, x_test, y_train, y_test = train_test_split(x, y_ds, train_size=train_size, random_state=42)

            print(f'\n\n {"-" * 5} Decision Tree Analysis {"-" * 5}')
            print('\n\n Train Size is ==> ', train_size)

            dtree = DecisionTreeClassifier()
            dtree.fit(x_train, y_train)
            tree_y_pred = dtree.predict(x_test)

            print('\n\n Prediction \n\n ', tree_y_pred)

            tree_conf = confusion_matrix(y_test, tree_y_pred, labels=[0, 1])

            ConfusionMatrixDisplay(tree_conf,
                                   display_labels=dtree.classes_).plot(
                                                                    cmap='RdPu',
                                                                    xticks_rotation='horizontal',
                                                                    colorbar=False
                                                                    )
            file_name = f'conf_mat_{train_size}.png'
            print('\n\n Confusion Matrix --> ', tree_conf)

            accuracy_rate = accuracy_score(y_test, tree_y_pred)
            print('\n\n ACCURACY RATE ==> ', accuracy_rate)

            tree_precision_rate = precision_score(y_test, tree_y_pred)
            print('\n\n PRECISION RATE ==> ', tree_precision_rate)

            tree_recall_rate = recall_score(y_test, tree_y_pred)
            print('\n\n RECALL RATE ==> ', tree_recall_rate)

            tree_f1_rate = f1_score(y_test, tree_y_pred)
            print('\n\n F1 SCORE  ==>  ', tree_f1_rate)

            tree_prf_rate = precision_recall_fscore_support(y_test, tree_y_pred)
            print('\n\n PRF RATE ==> ', tree_prf_rate)

            plt.savefig(f'output/dtree/{file_name}')

            stats_dict['precision'] = tree_precision_rate
            stats_dict['recall'] = tree_recall_rate
            stats_dict['f1_score'] = tree_f1_rate

            dtree_dict[train_size] = stats_dict

        return dtree_dict

