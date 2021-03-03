import numpy as np
import itertools

#CHAR_VECTOR = "adefghjknqrstwABCDEFGHIJKLMNOPZ0123456789"

letters = [letter for letter in "dkqtwABCDEFGHIJKLMNOP0123456789"]

num_classes = len(letters) + 1

img_w, img_h = 128, 64

# Network parameters
batch_size = 128
val_batch_size = 16

downsample_factor = 4
max_text_len = 9

epochs = 30

learning_phase = 0

Region = {"A": "서울 ", "B": "경기 ", "C": "인천 ", "D": "강원 ", "E": "충남 ", "F": "대전 ",
          "G": "충북 ", "H": "부산 ", "I": "울산 ", "J": "대구 ", "K": "경북 ", "L": "경남 ",
          "M": "전남 ", "N": "광주 ", "O": "전북 ", "P": "제주 "}
Hangul = {"dk": "아", "wk": "자", "tk": "사", "qk": "바"}
# Hangul = {"dk": "아", "dj": "어", "dh": "오", "dn": "우", "qk": "바", "qj": "버", "qh": "보", "qn": "부",
#           "ek": "다", "ej": "더", "eh": "도", "en": "두", "rk": "가", "rj": "거", "rh": "고", "rn": "구",
#           "wk": "자", "wj": "저", "wh": "조", "wn": "주", "ak": "마", "aj": "머", "ah": "모", "an": "무",
#           "sk": "나", "sj": "너", "sh": "노", "sn": "누", "fk": "라", "fj": "러", "fh": "로", "fn": "루",
#           "tk": "사", "tj": "서", "th": "소", "tn": "수", "gj": "허"}



