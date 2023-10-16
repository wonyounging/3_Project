from pred import views
from django.urls import path

urlpatterns = [
    path('', views.home),
    path('join', views.join),
    path('login', views.login),
    path('logout', views.logout),
    path('insert', views.insert),
    path('result', views.result, name='result'),
]