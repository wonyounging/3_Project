from django.shortcuts import render, redirect
from chat.models import Movie, Member, Chat
import hashlib
from pred.mypred import getImage
import numpy as np
import base64

# UPLOAD_DIR = "C:/workspace/hw_study3/myweb3/board/static/images/"

def home(request):
    if 'userid' not in request.session.keys():
        return render(request, 'pred/login.html')
    else:
        return render(request, 'pred/main.html')

def login(request):
    if request.method == 'POST':
        userid = request.POST['userid']
        passwd = request.POST['passwd']
        passwd = hashlib.sha256(passwd.encode()).hexdigest()
        row = Member.objects.filter(userid=userid, passwd=passwd)
        if len(row) > 0:
            row = Member.objects.filter(userid=userid, passwd=passwd)[0]
            request.session['userid'] = userid
            request.session['name'] = row.name
            return render(request, 'pred/main.html')
        else:
            return render(request, 'pred/login.html',
                          {'msg': '아이디 또는 비밀번호가 일치하지 않습니다.'})

    else:
        return render(request, 'pred/login.html')

def join(request):
    if request.method == 'POST':
        userid = request.POST['userid']
        passwd = request.POST['passwd']
        passwd = hashlib.sha256(passwd.encode()).hexdigest()
        name = request.POST['name']
        address = request.POST['address']
        tel = request.POST['tel']
        Member(userid=userid, passwd=passwd, name=name, address=address, tel=tel).save()
        request.session['userid'] = userid
        request.session['name'] = name
        return render(request, 'pred/main.html')
    else:
        return render(request, 'pred/join.html')

def logout(request):
    request.session.clear()
    return redirect('/')

def insert(request):
    return render(request, 'pred/insert.html')

def result(request):
    if "file" in request.FILES:
        file = request.FILES["file"]
        image_data = np.frombuffer(file.read(), dtype=np.uint8)
        encoded_image = base64.b64encode(image_data).decode("utf-8")

        img = getImage(image_data)

        genre = img['Genre']
        print(f"장르 : {genre}")

        emb_img1 = img['Embedding_images'][0]
        emb_img2 = img['Embedding_images'][1]
        emb_img3 = img['Embedding_images'][2]

        emb_img1_genre = img['Embedding_images_genres'][0]
        emb_img2_genre = img['Embedding_images_genres'][1]
        emb_img3_genre = img['Embedding_images_genres'][2]

        colors = img['Main_colors']

        ocr_img = img['OCR_image']
        ocr_lst = img['OCR_lst']
        obj_img = img['Object_image']
        obj_lst = img['Object_lst']
        obj_col = img['Object_col']

        # obj_lst_col = list(zip(obj_col, obj_lst))
        # 값이 있는 경우만 추출
        obj_lst_col = [(col, val) for col, val in zip(obj_col, obj_lst) if val != 0]

        if len(obj_lst_col)!=0:
            # 내림차순 정렬
            obj_lst_col = sorted(obj_lst_col, key=lambda x: x[1], reverse=True)[:5]
        while len(obj_lst_col)<5:
            obj_lst_col.append(('',''))

        context = {
            'genre': genre,
            'image_data': encoded_image,
            'emb_img1': emb_img1,
            'emb_img2': emb_img2,
            'emb_img3': emb_img3,
            'emb_img1_genre': emb_img1_genre,
            'emb_img2_genre': emb_img2_genre,
            'emb_img3_genre': emb_img3_genre,
            'colors': colors,
            'ocr_img': ocr_img,
            'ocr_lst': ocr_lst,
            'obj_img': obj_img,
            'obj_lst': obj_lst,
            'obj_col': obj_col,
            'obj_lst_col': obj_lst_col,

        }
        return render(request, 'pred/result.html', context)
    else:
        # 파일이 업로드되지 않은 경우에 대한 처리
        return render(request, 'pred/result.html', {'genre': '파일이 업로드되지 않았습니다.'})