# from django.shortcuts import render

# def manager(request):
    
#     return render(request, "manager/manager.html")
import pandas as pd
import os
from django.conf import settings
from django.shortcuts import render

def csv_to_table(request):
    # CSV 파일 경로
    csv_file_path = os.path.join(settings.STATICFILES_DIRS[0], 'data', 'Telco-customer-churn.csv')
    
    # pandas로 CSV 파일 읽기
    df = pd.read_csv(csv_file_path)
    
    # 데이터프레임을 HTML로 변환 (템플릿으로 전달하기 위해)
    table_html = df.to_html(index=False)
    
    return render(request, 'manager/manager.html', {'table_html': table_html})

def CustomerTable(request):
    return render(request,'manager/CustomerTable.html')

def Charts(request):
    return render(request, 'manager/Charts.html')
