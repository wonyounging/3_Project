import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import pickle
import cv2

import extcolors
import base64

import easyocr
import random
from PIL import ImageFont, ImageDraw, Image

from konlpy.tag import Kkma

from ultralytics import YOLO
from collections import Counter

def read_file(file_path):

    words = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 줄바꿈 문자를 제거하고 단어를 리스트에 추가
            word = line.strip()
            words.append(word)
    return words

class PredictModel:
    def __init__(self, embeddings, X, y, final_model, new_model, act_nouns, ani_nouns, com_nouns, dra_nouns, hor_nouns, etc_nouns):
        with open(embeddings, 'rb') as file:
            self.embeddings = pickle.load(file)

        self.X = np.load(X)
        self.y = np.load(y)

        label_encoder = LabelEncoder()
        self.y_label = label_encoder.fit_transform(self.y)

        self.embedding_model = ResNet50(weights='imagenet', include_top=False)
        self.final_model = load_model(final_model)
        self.new_model = YOLO(new_model)  # Mask 모델

        self.labels = {0: '공포(호러)', 1: '드라마', 2: '애니메이션', 3: '액션', 4: '코미디'}
        self.color_lst = np.random.randint(0, 255, size=(255, 3), dtype="uint8")
        self.font = ImageFont.truetype('C:/3rd_project/MovieChat/predictions/data/NanumGothicBold.ttf', 10)
        self.kkma = Kkma()

        self.act_nouns = act_nouns
        self.ani_nouns = ani_nouns
        self.com_nouns = com_nouns
        self.dra_nouns = dra_nouns
        self.hor_nouns = hor_nouns
        self.etc_nouns = etc_nouns

    def embedding_pred(self, content):
        image_data = cv2.imdecode(content, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)  # 컬러

        test = cv2.resize(img, (224, 224))

        image = test.reshape(-1, 224, 224, 3)
        img_array = preprocess_input(image)

        reference_embedding = self.embedding_model.predict(img_array).reshape((1, -1))

        # 모든 이미지 간의 코사인 유사도 계산
        similarities = []
        for embedding in self.embeddings:
            # 이미지 임베딩을 2차원 배열로 변환
            embedding = embedding.reshape((1, -1))
            similarity = cosine_similarity(reference_embedding, embedding)[0][0]
            similarities.append(similarity)

        # 유사도를 기준으로 내림차순으로 정렬하고 상위 3개 이미지 인덱스 선택
        num_similar_images = 3
        most_similar_image_indices = np.argsort(similarities)[-num_similar_images:][::-1]

        X3 = []

        for idx in list(most_similar_image_indices):
            X_img = cv2.cvtColor(self.X[idx], cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode('.png', X_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            X3.append(img_base64)

        embedding_images = X3

        y3 = []

        for idx in list(most_similar_image_indices):
            if self.y[idx] == '-':
                y3.append("기타")
            else:
                y3.append(self.y[idx])

        embedding_images_genres = np.array(y3)

        y_label3 = []

        for idx in list(most_similar_image_indices):
            y_label3.append(self.y_label[idx])

        embedding_input = np.array(y_label3).reshape(1, 3)

        return embedding_images, embedding_images_genres, embedding_input

    def color_pred(self, content):
        image_data = cv2.imdecode(content, cv2.IMREAD_COLOR)
        test = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)  # 컬러
        pil_image = Image.fromarray(test)
        colors, pixel_count = extcolors.extract_from_image(pil_image)

        main_color3 = []

        a = 0
        b = 0
        c = 0
        e = 0
        f = 0
        g = 0

        if len(colors) == 4:
            for color in colors[:2]:
                a += color[0][0]
                b += color[0][1]
                c += color[0][2]

            colors.insert(1, colors[0])

        elif len(colors) == 3:
            for color in colors[:2]:
                a += color[0][0]
                b += color[0][1]
                c += color[0][2]

            colors.insert(1, colors[0])

            for color in colors[2:4]:
                e += color[0][0]
                f += color[0][1]
                g += color[0][2]

            colors.insert(3, colors[2])

        elif len(colors) == 2:
            for color in colors[:2]:
                a += color[0][0]
                b += color[0][1]
                c += color[0][2]

            colors.append(((int(a / 2), int(b / 2), int(c / 2)), None))
            colors.insert(1, colors[0])
            colors.insert(3, colors[2])

        main_color3.append(np.array([colors[i][0] for i in range(5)]))

        main_colors = []
        for R, G, B in np.array(main_color3).reshape(5, 3):
            main_colors.append(f'RGB({R}, {G}, {B})')

        color_input = np.array(main_color3).reshape(-1, 1, 5, 3) / 255

        return main_colors, color_input

    def object_pred(self, content):
        '''
        이미지 적용
        '''
        image_data = cv2.imdecode(content, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)  # 컬러
        image = cv2.resize(img, (320, 448))
        results = list(self.new_model.predict(source=image,
                                              conf=0.3, show=False, stream=False))
        #                                 예측률0.5초과                      detect할 클래스만 classes=[0, 2, ...] 추가
        #                                                                   없는 경우 생략
        '''
        이미지 출력
        '''
        numpy_image = results[0].plot()
        X_img = cv2.cvtColor(np.array(numpy_image), cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.png', X_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        object_image = img_base64

        '''
        개체 수 출력
        '''
        # gpu => cpu
        cls_cpu = results[0].boxes.cls.to('cpu').to(int)
        # cpu => list
        cls_list = cls_cpu.tolist()

        cls_counts = Counter(cls_list)
        # print(cls_counts)

        # 클래스 이름과 개수를 연결하는 딕셔너리 생성
        ditect_class_names = {
            self.new_model.names[class_idx]: str(count)
            for class_idx, count in cls_counts.items()
        }

        # 클래스 단순화(81 > 13)
        vehicle = 0
        food = 0
        animal = 0
        bird = 0
        person = 0
        objects = 0
        electronic_products = 0
        computer = 0
        sport = 0
        weapon = 0
        eating_utensil = 0
        bag = 0
        etc = 0

        for key, value in ditect_class_names.items():
            value = int(value)

            if key in ['airplane', 'bicycle', 'boat', 'bus', 'car', 'motorcycle', 'train', 'truck']:
                vehicle += value
            elif key in ['apple', 'banana', 'broccoli', 'cake', 'carrot', 'donut', 'hot dog', 'orange', 'pizza',
                         'sandwich']:
                food += value
            elif key in ['bear', 'cat', 'cow', 'dog', 'elephant', 'giraffe', 'horse', 'sheep', 'teddy bear', 'zebra']:
                animal += value
            elif key in ['bird']:
                bird += value
            elif key in ['person']:
                person += value
            elif key in ['bed', 'bench', 'book', 'chair', 'clock', 'couch', 'dining table', 'potted plant', 'suitcase',
                         'remote', 'tie', 'toilet', 'toothbrush', 'umbrella', 'vase']:
                objects += value
            elif key in ['cell phone', 'hair drier', 'microwave', 'oven', 'refrigerator', 'toaster', 'tv']:
                electronic_products += value
            elif key in ['keyboard', 'laptop', 'mouse']:
                computer += value
            elif key in ['baseball bat', 'baseball glove', 'frisbee', 'skateboard', 'skis', 'snowboard', 'sports ball',
                         'surfboard', 'tennis racket']:
                sport += value
            elif key in ['knife', 'scissors', 'gun']:
                weapon += value
            elif key in ['bottle', 'bowl', 'cup', 'fork', 'spoon', 'wine glass', 'sink']:
                eating_utensil += value
            elif key in ['backpack', 'handbag']:
                bag += value
            else:
                etc += value

        object_col = ['vehicle', 'food', 'animal', 'bird', 'person', 'objects', 'electronic_products', 'computer', 'sport', 'weapon',
                      'eating_utensil', 'bag', 'etc']
        object_lst = [vehicle, food, animal, bird, person, objects, electronic_products, computer, sport, weapon,
                      eating_utensil, bag, etc]
        object_input = np.array(object_lst).reshape(1, 13)


        return object_image, object_col, object_lst, object_input

    def ocr_image(self, content):
        content_3d = cv2.imdecode(content, cv2.IMREAD_COLOR)
        content_3d = cv2.cvtColor(content_3d, cv2.COLOR_BGR2RGB)  # 컬러
        img = cv2.resize(content_3d, (224, 336))

        reader = easyocr.Reader(['ko', 'en'], gpu=True)
        result = reader.readtext(img)

        image = Image.fromarray(img)
        draw = ImageDraw.Draw(image)

        for i in result:
            x = i[0][0][0]
            y = i[0][0][1]
            w = i[0][1][0] - i[0][0][0]
            h = i[0][2][1] - i[0][1][1]

            color_idx = random.randint(0, len(self.color_lst)-1)
            box_color = [int(c) for c in self.color_lst[color_idx]]

            draw.rectangle(((x, y), (x + w, y + h)), outline=tuple(box_color), width=4)
            draw.text((int((x + x + w) / 2), y - 2), str(i[1]), font=self.font, fill=tuple(box_color))

        X_img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.png', X_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        ocr_image = img_base64

        return ocr_image

    def image_to_text(self, content):  # 검출된 텍스트 영역에 다양한 이미지 필터 적용 후 텍스트 추출
        # 추출된 텍스트 영역을 이용하여 각각의 텍스트를 잘라내고 EasyOCR을 다시 적용
        content_3d = cv2.imdecode(content, cv2.IMREAD_COLOR)
        content_3d = cv2.cvtColor(content_3d, cv2.COLOR_BGR2RGB)  # 컬러
        img = cv2.resize(content_3d, (224, 336))

        reader = easyocr.Reader(['ko', 'en'], gpu=True)
        result = reader.readtext(img)

        origin_results = []
        roi_results = []
        roi_gray_results = []
        roi_2_results = []
        roi_equal_results = []
        roi_dil_results = []
        roi_ero_results = []
        roi_canny_results = []

        for (box, text, confidence) in result:
            try:
                origin_results.append(text)
            except:
                pass

            # 각 텍스트 박스의 좌표 추출
            (startX, startY) = box[0]
            (endX, endY) = box[2]

            # 좌표를 정수로 변환
            startX, startY, endX, endY = int(startX), int(startY), int(endX), int(endY)

            # print(startX, startY, endX, endY)
            # 좌표가 음수이거나 이미지 범위를 벗어나는 경우 무시
            if startX < 0 or startY < 0 or endX >= img.shape[1] or endY >= img.shape[0]:
                continue

            # 텍스트 영역 추출
            roi = img[startY:endY, startX:endX]

            ############################################# 이미지 전처리 ###################################################

            # 이미지를 그레이스케일로 변환
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # 적응형 이진화 수행
            roi_adaptive_thresh = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                        cv2.THRESH_BINARY, 15, 2)

            # 히스토그램 평탄화 수행
            roi_equalized = cv2.equalizeHist(roi_gray)

            # 팽창
            kernel = np.ones((3, 3), np.uint8)
            roi_dilation = cv2.dilate(roi, kernel, iterations=1)

            # 침식
            kernel = np.ones((3, 3), np.uint8)
            roi_erosion = cv2.erode(roi, kernel, iterations=1)

            # 캐니
            roi_canny = cv2.Canny(roi, 475, 500, apertureSize=3, L2gradient=True)

            ############################################# 텍스트 인식 ###################################################

            # 잘라낸 텍스트 영역에 EasyOCR을 적용
            roi_result = reader.readtext(roi)
            roi_gray_result = reader.readtext(roi_gray)
            roi_2_result = reader.readtext(roi_adaptive_thresh)
            roi_equal_result = reader.readtext(roi_equalized)
            roi_dil_result = reader.readtext(roi_dilation)
            roi_ero_result = reader.readtext(roi_erosion)
            roi_canny_result = reader.readtext(roi_canny)

            ############################################# 결과 출력 ###################################################

            roi_results_lst = [roi_results, roi_gray_results, roi_2_results, roi_equal_results,
                               roi_dil_results, roi_ero_results, roi_canny_results]

            roi_result_lst = [roi_result, roi_gray_result, roi_2_result, roi_equal_result,
                              roi_dil_result, roi_ero_result, roi_canny_result]

            for i in range(len(roi_result_lst)):
                try:
                    roi_results_lst[i].append(roi_result_lst[i][0][1])
                except:
                    roi_results_lst[i].append('None')

        return origin_results, roi_results, roi_gray_results, roi_2_results, roi_equal_results, roi_dil_results, roi_ero_results, roi_canny_results

    def ocr_pred(self, content):
        text = self.image_to_text(content)
        nouns = self.kkma.nouns(str(text))

        noun_list = []

        # 추출된 명사 출력
        for i2t_noun in nouns:
            if len(i2t_noun) > 1:
                noun_list.append(i2t_noun)

        # 명사 리스트 생성
        noun_list = list(set(noun_list))

        act_freq = 0
        ani_freq = 0
        com_freq = 0
        dra_freq = 0
        hor_freq = 0
        etc_freq = 0

        # 입력한 단어들의 빈도수 확인
        try:
            for word in noun_list:
                if word in self.act_nouns:
                    act_freq += 1
                elif word in self.ani_nouns:
                    ani_freq += 1
                elif word in self.com_nouns:
                    com_freq += 1
                elif word in self.dra_nouns:
                    dra_freq += 1
                elif word in self.hor_nouns:
                    hor_freq += 1
                else:
                    etc_freq += 1
        except:
            pass

        ocr_lst = [act_freq, ani_freq, com_freq, dra_freq, hor_freq]
        ocr_input = np.array(ocr_lst).reshape(1, 5)

        return ocr_lst, ocr_input

    def predict_class(self, color_input, embedding_input, object_input, ocr_input):
        test_input = [color_input, embedding_input, object_input, ocr_input]
        predict = self.final_model.predict(test_input)
        predict_class = tf.math.argmax(predict, axis=1)
        predict_label = self.labels[predict_class.numpy()[0]]
        return predict_label