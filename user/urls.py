from django.urls import path
from . import views

# app.url.py -> request & view를 연결해줌
urlpatterns = [
    # localhost
    path('Message/<str:customer_id>/&TotalCharges', views.Message, name='Message'), # 메일함(메인)
    path('Message/<str:customer_id>/&tenure', views.Message, name='Message'),
    path('Message/<str:customer_id>/&MultipleLines_Yes', views.Message, name='Message'),

    path('Message/<str:customer_id>/<int:email_id>/', views.read_msg, name='read_msg'),

    path('Coupon/<str:customer_id>/', views.Coupon, name='Coupon'), # 쿠폰함
    path('Profile/<str:customer_id>/', views.Profile, name='Profile') # 개인 프로필
]
