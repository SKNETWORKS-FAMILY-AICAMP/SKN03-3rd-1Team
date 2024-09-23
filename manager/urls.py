from django.urls import path
from . import views

# app.url.py -> request & view를 연결해줌
urlpatterns = [
    # localhost
    path('', views.CustomerTable, name='CustomerTable'),
    path('CustomerTable/', views.CustomerTable, name='CustomerTable'),
    path('churn-chart/<str:customer_id>/', views.churn_chart, name='churn_chart'),
]