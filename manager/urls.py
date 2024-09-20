from django.urls import path
from .views import manager

# app.url.py -> request & view를 연결해줌
urlpatterns = [
    # localhost
    path('',manager,name="manager-connect")
]