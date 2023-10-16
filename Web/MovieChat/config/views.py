from django.shortcuts import render

def home(request):
    #installed_app에 작성한 첫번째 앱인 chat의 index.html 페이지로 출력됨
    return render(request, 'index.html')