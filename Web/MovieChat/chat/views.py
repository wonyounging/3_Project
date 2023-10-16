from django.shortcuts import render, redirect
from chat.models import Movie, Member, Chat
from chat.models import Movie, Member, Chat
import hashlib
from chat.mychatbot import getMessage


def home(request):
    if 'userid' not in request.session.keys():
        return render(request, 'chat/login.html')
    else:
        return render(request, 'chat/main.html')

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
            return render(request, 'chat/main.html')
        else:
            return render(request, 'chat/login.html',
                          {'msg': '아이디 또는 비밀번호가 일치하지 않습니다.'})

    else:
        return render(request, 'chat/login.html')

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
        return render(request, 'chat/main.html')
    else:
        return render(request, 'chat/join.html')

def logout(request):
    request.session.clear()
    return redirect('/')

def order(request):
    return render(request, 'chat/order.html')

def query(request):
    question = request.GET["question"] # 사용자 입력 내용 (질문, 주문 등)
    msg = getMessage(question) # json 형태
    query = msg['Query'] # json key 값으로 value 값 찾기
    answer = msg['Answer']
    intent = msg['Intent']
    Chat(userid=request.session['userid'], query=query, intent=intent).save()
    Chat(userid=request.session['userid'], answer=answer, intent=intent).save()
    items = Chat.objects.filter(userid=request.session['userid']).order_by('-idx') # 오래된 글이 아래로 내려가는 구조
    return render(request, 'chat/result.html', {'items':items})

def delete_chat(request):
    Chat.objects.filter(userid=request.session['userid']).delete() # 대화내용 삭제
    return redirect('order')