from django.urls import path
from . import views

# app.url.py -> request & view를 연결해줌
urlpatterns = [
    # localhost
    path('', views.CustomerTable, name='CustomerTable'),
    path('CustomerTable/', views.CustomerTable, name='CustomerTable'),
    path('Tenure-churn-chart/<str:customer_id>/', views.churn_chart_tenure, name='churn_chart'),
    path('TotalCharges-churn-chart/<str:customer_id>/', views.churn_chart_totalCharges, name='churn_chart')
]