from sklearn.metrics import plot_roc_curve, f1_score, confusion_matrix
import numpy as np

def thresh_selection(xtest, ytest, classifier, start, stop, step=None):
    '''
    Selects best threshold for tuning model for Precision or Recall
    :param xtest: test set data
    :param ytest: test set labels
    :param classifier: sklearn model
    :param start: starting point for thresh value
    :param stop: stopping point for thresh value
    :param step: iterative value between start and stop (optional)
    :return: prints f1 scores and fp/fn results, also plots roc curve
    '''

    for thresh in np.arange(start, stop, step):
        threshold = thresh
        predicted_proba = classifier.predict_proba(xtest)
        predicted = (predicted_proba[:, 1] >= threshold).astype('int')
        print(f'F1 Score at {round(threshold, 2)} prediction threshold: {round(f1_score(ytest, predicted), 2)}')
        tp, fp, fn, tn = confusion_matrix(ytest, predicted).ravel()
        print(f'False Positives = {fp}, False Negatives = {fn}\n\n')
    plot_roc_curve(classifier, xtest, ytest, lw=3)
    
def odds_to_prob(log_odds):
    '''
    Given log odds returns probability
    :param log_odds: float or int
    :returns : float, probability
    '''
    return np.exp(log_odds)/(1+np.exp(log_odds))

    