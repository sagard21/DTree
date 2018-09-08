import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.tree import export_graphviz
import graphviz
from models.models_sklearn import *
from utils import load_relevant_data
from preprocess.preprocessing import cat2int, get_x_y
import pydotplus


def show_all_counts(data_df, save_loc=None):
    """
    Provides a visual representation of counts of all the parameters except
    Application_ID
    :param
        data_df: your complete / test / train dataset
        save_loc: location of direcory and name of file for saving. Use png
                  to save images
    :return: grid of sns.countplot
    """

    # Creating subplots / grids
    fig, ax = plt.subplots(nrows=3, ncols=3)

    # Setting size of the fig to accommodate all viz
    fig.set_size_inches(15, 15)

    # Initiate a list to save ax position values
    pos_ls = []
    for i in itertools.product(range(3), range(3)):
        pos_ls.append(list(i))

    # Iterate over columns and add viz in the ax positions
    for col, pos in zip(data_df.columns.values[1:], pos_ls):
        a = sns.countplot(data_df[col], ax=ax[pos[0], pos[1]])
        # Annotate the count
        for p in a.patches:
            a.annotate('{}'.format(p.get_height()),
                       (p.get_x() + 0.3, p.get_height() + 0.1))

    fig.suptitle('Countplot of all the parameters', color='navy', fontsize=21)
    if save_loc:
        plt.savefig(save_loc)
    plt.show()


def create_tree(data_df, method='gini', max_depth=None, min_samples_leaf=1,
                image_name='1'):
    """
    Creates a tree graph
    :param data_df: raw data frame
    :param method: gini or entropy
    :param max_depth: int or None
    :param min_samples_leaf: 1 or int
    :return: tree visualization
    """

    X, y = get_x_y(data_df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=7)

    clf = tree.DecisionTreeClassifier(criterion=method, max_depth=max_depth,
                                      min_samples_leaf=min_samples_leaf)

    clf = clf.fit(X_train, y_train)

    dot_data = export_graphviz(clf, out_file=None,
                               feature_names=X_train.columns.values,
                               class_names=y_train,
                               rounded=True, proportion=False,
                               precision=2, filled=True)

    graph = graphviz.Source(dot_data)

    img = pydotplus.graph_from_dot_data(dot_data)

    img.write_jpeg('Images/'+image_name)

if __name__ == '__main__':

    app_df = load_relevant_data()

    app_df = cat2int(app_df)

    create_tree(app_df)