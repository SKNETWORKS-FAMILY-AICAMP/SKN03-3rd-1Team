from django.urls import path
from .views import user

# app.url.py -> request & view를 연결해줌
urlpatterns = [
    # localhost
    path('',user,name="user-connect")
]