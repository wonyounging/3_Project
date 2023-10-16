from chat import views
from django.urls import path

urlpatterns = [
    path('', views.home),
    path('join', views.join),
    path('login', views.login),
    path('logout', views.logout),
    path('order', views.order, name='order'),
    path('query', views.query),
    path('delete_chat', views.delete_chat),
]