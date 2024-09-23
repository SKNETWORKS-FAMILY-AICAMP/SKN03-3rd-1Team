from django.urls import path
from . import views

# app.url.py -> request & view를 연결해줌
urlpatterns = [
    # localhost
    path('', views.CustomerTable, name='CustomerTable'),
    path('CustomerTable/', views.CustomerTable, name='CustomerTable'),
    path('Charts/', views.Charts, name='Charts')
]