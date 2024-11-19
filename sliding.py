import numpy as np


def get_win_emg(s, yindex, ws=200, ss=100, file='tmp.txt'):
    """
    process dataset EMG with sliding window
    """
    x = np.array([[float(j) for j in item.split('\t')[1:-1]] for item in s])
    y = np.array([int(item.split('\t')[-1]) for item in s])
    l = len(y)
    i = 0
    tx, ty = [], []
    
    while i <= (l - ws):
        while (i <= (l - ws)) and (y[i] in [0, 7]):
            i += 1
        j = i + ws
        if j < l:
            if len(np.unique(y[i:j])) == 1:
                tx.append(x[i:j].T)
                ty.append(yindex[y[i]])
                i += ss
            else:
                while y[i] == y[i + 1]:
                    i += 1
                i += 1
                
    return np.array(tx), np.array(ty)
