def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    fp = 0
    tp = 0
    fn = 0
    tn = 0
    count_for_accuracy = 0
    l = len(prediction)

    for i in range(l):
        if (prediction[i] == ground_truth[i]):
            count_for_accuracy += 1
        if ((prediction[i] == False) & (ground_truth[i] == False)):
            tp += 1
        if ((prediction[i] == True) & (ground_truth[i] == False)):
            fn += 1
        if ((prediction[i] == True) & (ground_truth[i] == True)):
            tn += 1
        if ((prediction[i] == False) & (ground_truth[i] == True)):
            fp += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = count_for_accuracy / l
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    accuracy = 0
    count = 0
    l = len(prediction)
    for i in range(l):
        if(prediction[i] == ground_truth[i]):
            count += 1
    accuracy = count / l
    return accuracy
