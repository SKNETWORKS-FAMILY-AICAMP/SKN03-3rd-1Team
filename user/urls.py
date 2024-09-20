from django.urls import path
from . import views

# app.url.py -> request & view를 연결해줌
urlpatterns = [
    # localhost
    path('Email/', views.Profile, name='Email'),
    path('Coupon/', views.Profile, name='Coupon'),
    path('', views.Profile, name='Profile'),
    path('Profile/', views.Profile, name='Profile')
]
