from django.urls import path
from .views import csv_to_table
from . import views

# app.url.py -> request & view를 연결해줌
urlpatterns = [
    # localhost
    path('',csv_to_table,name="manager-connect")
    path('', views.CustomerTable, name='CustomerTable'),
    path('CustomerTable/', views.CustomerTable, name='CustomerTable'),
    path('Charts/', views.Charts, name='Charts')
]