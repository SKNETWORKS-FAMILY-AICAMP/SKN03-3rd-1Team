from django.urls import path
from .views import csv_to_table

# app.url.py -> request & view를 연결해줌
urlpatterns = [
    # localhost
    path('',csv_to_table,name="manager-connect")
]