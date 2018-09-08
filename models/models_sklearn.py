from sklearn import tree
from sklearn.model_selection import train_test_split
from utils import load_relevant_data
from preprocess.preprocessing import get_x_y, cat2int
from sklearn.metrics import accuracy_score
from itertools import product
import pandas as pd
from os.path import join
from datetime import datetime


def run_dtree(data_df, method='gini', max_depth=None, min_samples_leaf=1):
    """
    Provides predictions made by decision tree and prints the accuracy score
    on test dataset
    :param method: criterion for split. Options:
                   gini
                   entropy
    :return: numpay array of predictions
    """

    X, y = get_x_y(data_df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=7)

    clf = tree.DecisionTreeClassifier(criterion=method, max_depth=max_depth,
                                      min_samples_leaf=min_samples_leaf)

    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print 'Your accuracy is {}'.format(accuracy_score(y_test, y_pred))

    return round((accuracy_score(y_test, y_pred)), 2)


if __name__ == '__main__':

    app_df = load_relevant_data()

    app_df = cat2int(app_df)

    base_dir = base_dir = '/home/sagar/Desktop/DecisionTree'
    res_dir = join(base_dir, 'results')

    method_ls = ['gini', 'entropy']
    max_depth_ls = [None, 3, 4]
    min_sample_ls = [1, 10, 20]

    params_ls = []
    for i in product(method_ls, max_depth_ls, min_sample_ls):
        params_ls.append(list(i))

    res_ls =[]

    run = 1
    for params in params_ls:
        run = run
        method = params[0]
        max_depth = params[1]
        min_sample = params[2]

        print 'Running experiment # {} using {} as splitting criterion and ' \
              'with max depth of {} & min sample leaf of {}'.format(run,
                                                                    method,
                                                                max_depth,
                                                                min_sample)
        accuracy = run_dtree(app_df, method=method, max_depth=max_depth,
                             min_samples_leaf=min_sample)

        res_ls.append((run, method, max_depth, min_sample, accuracy))
        run +=1

    fin_df = pd.DataFrame(res_ls, columns=['Run No.', 'Mehtod', 'Max Depth',
                                           'Min Samples Leaf', 'Accuracy'])

    print fin_df.head()

    today = datetime.today().date()

    file = str(today)+'.csv'
    file = join(res_dir, file)

    fin_df.to_csv(file, index=False)
