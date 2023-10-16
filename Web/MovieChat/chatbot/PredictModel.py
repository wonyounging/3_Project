import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import preprocessing
import numpy as np

# 분류 모델 모듈
class PredictModel:
    def __init__(self, category, model_name, proprocess):

        self.MAX_SEQ_LEN = 15
        self.category = category

        if self.category == 'intent':
            self.labels = {0: '기타', 1: '추천', 2: '후기', 3: '정보', 4: '예매', 5: '욕설'}

        elif self.category == 'emotion':
            self.labels = {0: '무서움', 1: '슬픔', 2: '신남', 3: '없음', 4: '웃김', 5: '재미'}

        elif self.category == 'binary':
            self.labels = {0: '긍정', 1: '부정', 2: '없음'}

        elif self.category == 'trend':
            self.labels = {0: '없음', 1: '인기', 2: '최신'}
        elif self.category == 'ner':
            self.MAX_SEQ_LEN = 30
            self.labels = {1: 'O', 2: 'B_MOVIE', 3: 'B_ACT', 4: 'B_GEN', 5: 'B_NAT',
                           6: 'B_DIR', 7: 'B_DT', 8: 'B_RAT', 0: 'PAD'}
        else:
            self.labels = {}

        self.labels[len(self.labels)] = "-"
        # 분류 모델 불러오기
        self.model = load_model(model_name)
        # 챗봇 Preprocess 객체
        self.p = proprocess

    # 클래스 예측
    def predict_class(self, query):
        # 형태소 분석
        pos = self.p.pos(query)

        # 문장내 키워드 추출(불용어 제거)
        keywords = self.p.get_keywords(pos, without_tag=True)
        sequences = [self.p.get_wordidx_sequence(keywords)]

        # 패딩처리
        padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=self.MAX_SEQ_LEN, padding='post')
        predict = self.model.predict(padded_seqs)
        predict_class = tf.math.argmax(predict, axis=1)

        return predict_class.numpy()[0]

        # ner_전용

    def predict_ner(self, query):
        # 형태소 분석
        pos = self.p.pos(query)

        # 문장내 키워드 추출(불용어 제거)
        keywords = self.p.get_keywords(pos, without_tag=True)
        sequences = [self.p.get_wordidx_sequence(keywords)]

        # 패딩처리
        padded_seqs = preprocessing.sequence.pad_sequences(sequences, padding="post", value=0,
                                                           maxlen=self.MAX_SEQ_LEN)
        predict = self.model.predict(np.array([padded_seqs[0]]))
        predict_class = tf.math.argmax(predict, axis=-1)
        tags = [self.labels[i] for i in predict_class.numpy()[0]]
        return list(zip(keywords, tags))
