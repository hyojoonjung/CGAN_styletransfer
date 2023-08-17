# preprocessing.py
import pandas as pd
import torch
from transformers import T5Tokenizer
from sklearn.model_selection import train_test_split

def preprocess_data():
    # 데이터 로드
    pandora = pd.read_csv('data/pandora.csv')
    essays = pd.read_csv('data/essays.csv')

    # TEXT 컬럼에서 NaN 값 제거
    pandora = pandora[pandora['TEXT'].notna()]
    essays = essays[essays['TEXT'].notna()]

    # TEXT 컬럼 내용을 리스트로 추출
    texts_pandora = pandora['TEXT'].tolist()
    texts_essays = essays['TEXT'].tolist()

    # 타겟 레이블을 리스트로 추출
    labels_pandora = pandora[['OPN', 'CON', 'EXT', 'AGR', 'NEU']].values.tolist()
    labels_essays = essays[['cOPN', 'cCON', 'cEXT', 'cAGR', 'cNEU']].values.tolist()

    return texts_pandora, texts_essays, labels_pandora, labels_essays

