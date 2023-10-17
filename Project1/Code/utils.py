import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_results(xx,yy, X, y, y_hat, title):
    '''
    utilitary function to plot results.
    It displays the training data with different colours and uses the same colours to differentiate 
    the different regions defined by the decision boundaries.
    '''
    '''
    INPUTS :
    - xx : x-axis coordinates of input testing points
    - yy : y-axis coordinates of input testing points
    - X : set of coordinates of the input training points
    - y : set of labels of the training points
    - y_hat : set of predicted labels 
    - title : character string specifying a description of the plot (e.g. how y_hat was obtained)
	'''
    '''
	OUTPUTS: /
    '''
    
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    plt.rcParams['figure.figsize'] = [5, 5]
    plt.pcolormesh(xx, yy, y_hat, cmap=cmap_light)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.show()