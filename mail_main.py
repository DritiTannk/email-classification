import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from mail_analysis.processing import DataProcess
from mail_analysis.logistics_algorithm import LogisticAlgorithmAnalysis
from mail_analysis.bayes_algorithm import BayesAlgorithmAnalysis
from mail_analysis.knn_algorithm import KnnAlgorithmAnalysis
from mail_analysis.dtree_algorithm import DecisionTreeAnalysis
from mail_analysis.svm_algorithm import SVMAlgorithmAnalysis


if __name__ == '__main__':
    print('\n\n 1. Email Dataset Generation')

    dp = DataProcess()
    dp.data_creation()

    print('\n\n 2. Dataset Processing')

    e_ds = pd.read_csv('input/EMAIL_DATASET.csv')
    e_columns = e_ds.columns

    x_ds = e_ds.FILE_NAME
    y_ds = e_ds.LABEL

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(x_ds)

    print('\n\n 3. Applying Logistics Algorithm')

    logalgo = LogisticAlgorithmAnalysis()
    log_dict = logalgo.log_analysis(X, y_ds)

    print('\n\n 4. Applying KNN Algorithm')

    knn = KnnAlgorithmAnalysis()
    knn_dict = knn.knn_analysis(X, y_ds)

    print('\n\n 5. Applying Decision Tree Algorithm')

    dtree = DecisionTreeAnalysis()
    tree_dict = dtree.dtree_Analysis(X, y_ds)

    print('\n\n 6. Applying SVM Algorithm')

    svm = SVMAlgorithmAnalysis()
    svm_dict = svm.svm_analysis(X, y_ds)

    print('\n\n 7. Applying Bayes Algorithm')

    bayesalgo = BayesAlgorithmAnalysis()
    bayes_dict = bayesalgo.bayes_analysis(X, y_ds)
    print('\n\n bayes dict ==> ', bayes_dict)



