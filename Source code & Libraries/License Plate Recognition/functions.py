import numpy as np
import itertools

from utils import letters, Region, Hangul

def labels_to_text(labels):
    '''
    Convert labels to text:
    @labels [in]: label list
    '''
    return ''.join(list(map(lambda x: letters[int(x)], labels)))

def text_to_labels(text):
    '''
    Convert text to list of labels:
    @labels [in]: input text
    '''
    return list(map(lambda x: letters.index(x), text))

def decode_label(out):
    # out : (1, 32, 42)
    out_best = list(np.argmax(out[0, 2:], axis=1))  # get max index -> len = 32
    out_best = [k for k, g in itertools.groupby(out_best)]  # remove overlap value
    outstr = ''
    for i in out_best:
        if i < len(letters):
            outstr += letters[i]
    return outstr

def label_to_hangul(label):  # eng -> hangul
    '''
    Convert label list to Hangul license plate:
    @label [in]: label list
    '''
    region = label[0]
    two_num = label[1:3]
    hangul = label[3:5]
    four_num = label[5:9]

    try:
        region = Region[region] if region != 'Z' else ''
    except:
        pass
    try:
        hangul = Hangul[hangul]
    except:
        pass
    return region + two_num + hangul + four_num