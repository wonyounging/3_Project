from predictions.PredictModel import PredictModel, read_file
import cv2

act_nouns = read_file('C:/3rd_project/MovieChat/predictions/data/act_nouns.txt')
ani_nouns = read_file('C:/3rd_project/MovieChat/predictions/data/ani_nouns.txt')
com_nouns = read_file('C:/3rd_project/MovieChat/predictions/data/com_nouns.txt')
dra_nouns = read_file('C:/3rd_project/MovieChat/predictions/data/dra_nouns.txt')
hor_nouns = read_file('C:/3rd_project/MovieChat/predictions/data/hor_nouns.txt')
etc_nouns = read_file('C:/3rd_project/MovieChat/predictions/data/etc_nouns.txt')

# 객체 생성
object = PredictModel('C:/3rd_project/MovieChat/predictions/data/embeddings_224.pkl',
             'C:/3rd_project/MovieChat/predictions/data/X_total_color.npy',
             'C:/3rd_project/MovieChat/predictions/data/y_total_color.npy',
             'C:/3rd_project/MovieChat/predictions/model/genre_final_model_ver3.h5',
             'C:/3rd_project/MovieChat/predictions/model/230928_new_best.pt',
             act_nouns, ani_nouns, com_nouns, dra_nouns, hor_nouns, etc_nouns)


def getImage(content):
    ocr_img = object.ocr_image(content) # ocr 이미지
    ocr_lst, ocr_input = object.ocr_pred(content)
    obj_img, obj_col, obj_lst, obj_input = object.object_pred(content)
    emb_imgs, emb_genres, emb_input = object.embedding_pred(content)
    colors, color_input = object.color_pred(content)
    genre = object.predict_class(color_input, emb_input, obj_input, ocr_input)

    json = {
        "Genre" : genre,
        "Main_colors" : colors,
        "Color_input" : color_input,
        "Embedding_images" : emb_imgs,
        "Embedding_images_genres" : emb_genres,
        "Embedding_input" : emb_input,
        "OCR_image": ocr_img,
        "OCR_lst": ocr_lst,
        "OCR_input": ocr_input,
        "Object_image": obj_img,
        "Object_lst" : obj_lst,
        "Object_col": obj_col,
        "Object_input": obj_input,
    }
    return json

# if __name__ == '__main__':
#     msg = getImage(X[0])
#     print(msg)