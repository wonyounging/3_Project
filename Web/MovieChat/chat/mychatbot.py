from chatbot.db.DatabaseConfig import *
from chatbot.db.Database import Database
from chatbot.Preprocess import Preprocess
from chatbot.PredictModel import PredictModel
from chatbot.FIndAnswer import FindAnswer


# 전처리 객체 생성
p = Preprocess(word2index_dic='c:/3rd_project/MovieChat/chatbot/data/chatbot_dict.bin',
                userdic='c:/3rd_project/MovieChat/chatbot/data/user_dic.txt')

intent = PredictModel(category='intent', model_name='c:/3rd_project/MovieChat/chatbot/model/question_intent_model.h5', proprocess=p)
emotion = PredictModel(category='emotion', model_name='c:/3rd_project/MovieChat/chatbot/model/question_emotion_model.h5', proprocess=p)
binary = PredictModel(category='binary', model_name='c:/3rd_project/MovieChat/chatbot/model/question_emotion_binary_model.h5', proprocess=p)
trend = PredictModel(category='trend', model_name='c:/3rd_project/MovieChat/chatbot/model/question_trend_model.h5', proprocess=p)
ner = PredictModel(category='ner', model_name='c:/3rd_project/MovieChat/chatbot/model/question_ner_model.h5', proprocess=p)


def ner_tag_sep(lsts):
    ner_movie = []
    ner_act = []
    ner_gen = []
    ner_nat = []
    ner_dir = []
    ner_dt = []
    ner_rat = []

    for lst in lsts:
        if lst[1] == 'B_MOVIE':
            ner_movie.append(lst[0])
        elif lst[1] == 'B_ACT':
            ner_act.append(lst[0])
        elif lst[1] == 'B_GEN':
            ner_gen.append(lst[0])
        elif lst[1] == 'B_NAT':
            ner_nat.append(lst[0])
        elif lst[1] == 'B_DIR':
            ner_dir.append(lst[0])
        elif lst[1] == 'B_DT':
            ner_dt.append(lst[0])
        elif lst[1] == 'B_RAT':
            ner_rat.append(lst[0])

        ner_movie = list(set(ner_movie))
        ner_act = list(set(ner_act))
        ner_gen = list(set(ner_gen))
        ner_nat = list(set(ner_nat))
        ner_dir = list(set(ner_dir))
        ner_dt = list(set(ner_dt))
        ner_rat = list(set(ner_rat))

    return ner_movie, ner_act, ner_gen, ner_nat, ner_dir, ner_dt, ner_rat

def predict_keyword(text):
    intent_pred = intent.predict_class(text)
    emotion_pred = emotion.predict_class(text)
    binary_pred = binary.predict_class(text)
    trend_pred = trend.predict_class(text)
    ner_pred = ner.predict_ner(text)

    intent_label = intent.labels[intent_pred]
    emotion_label = emotion.labels[emotion_pred]
    binary_label = binary.labels[binary_pred]
    trend_label = trend.labels[trend_pred]
    ner_label = ner_tag_sep(ner_pred)

    return intent_label, emotion_label, binary_label, trend_label, ner_label


def getMessage(query):
    lsts = predict_keyword(query)
    db = Database(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, db_name=DB_NAME)
    obj = FindAnswer(db, lsts)
    ans = obj.find_answer()
    json = {
        "Query" : query,
        "Intent" : lsts[0],
        "Emotion" : lsts[1],
        "Binary" : lsts[2],
        "Trend" : lsts[3],
        "Ner" : lsts[4],
        "Answer" : ans
    }
    return json

if __name__ == '__main__':
    msg = getMessage('요즘 봉준호 감독 영화 추천부탁드립니다.')
    print(msg)
