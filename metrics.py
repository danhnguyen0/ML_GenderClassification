def trunc(n, k):
    """
    Return the number truncated to k decimal places, e.g, k = 0 returns only the integer part
    """

    return int(n * 10**k) / 10**k

def measure(y_true, y_pred, k):
    """
    y_pred = binary vector
    y_true = binary vector

    Desc: Computes the accuracy, TP, TN, FP, FN, BER truncated to k decimal places
    """
    
    if len(y_pred) != len(y_true):
        print("Lengths not equal: y_pred={}, y_true={}".format(len(y_pred), len(y_true)))
        return

    TP, TN, FP, FN = 0, 0, 0, 0,
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i] and y_true[i] == 1:
            TP += 1
        elif y_pred[i] == y_true[i] and y_true[i] == 0:
            TN += 1
        elif y_pred[i] != y_true[i] and y_true[i] == 1:
            FN += 1
        elif y_pred[i] != y_true[i] and y_true[i] == 0:
            FP += 1 
    
    total = TP + TN + FP + FN
    FPR = TP / (TP + FP) if TP + FP > 0 else 0
    FPN = TN / (TN + FN) if TN + FN > 0 else 0
    acc = (TP + TN) / total
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    BER = (FPR + FPN) / 2
    return (TP, TN, FP, FN, trunc(precision, k), trunc(recall, k), trunc(acc, k), trunc(f_score, k))

def prod(S):
    product = 1
    for x in S:
        product *= x
    return product

def writeMetrics(y_true, y_pred, k):
    acc, TP, TN, FP, FN, BER = measure(y_true, y_pred, k)
    f = open("README.me","a")
    f.write('\n')
    f.write('--- Model _: ___________ ---\n')
    f.write('Accuracy, BER ~ {}, {}\n'.format(acc, BER))
    f.write('TP, TN, FP, FN = {}, {}, {}, {}\n'.format(TP,TN,FP,FN))
    f.close()

def measureMetrics(y_true, y_pred, k):
    acc, TP, TN, FP, FN, BER = measure(y_true, y_pred, k)
    print('\n')
    print('--- Model _: ___________ ---\n')
    print('Accuracy, BER ~ {}, {}\n'.format(acc, BER))
    print('TP, TN, FP, FN = {}, {}, {}, {}\n'.format(TP,TN,FP,FN))
