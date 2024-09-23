from django.shortcuts import render
import pandas as pd
import os
from django.conf import settings
from django.core.paginator import Paginator

import os
import pandas as pd
from django.conf import settings
from django.shortcuts import render
import matplotlib.pyplot as plt
import seaborn as sns
from django.core.paginator import Paginator

def CustomerTable(request):
    try:
        # CSV 파일 경로
        csv_file_path = os.path.join(settings.BASE_DIR, 'static', 'assets', 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
        df = pd.read_csv(csv_file_path)

        # Continuous Data와 Discrete Data 설정
        continuous_columns = ['MonthlyCharges', 'TotalCharges']
        discrete_columns = ['tenure']

        # Binary Data와 Non-Binary Data 설정
        binary_columns = ['PhoneService', 'Dependents', 'gender', 'SeniorCitizen', 'Partner', 'PaperlessBilling']
        non_binary_columns = ['DeviceProtection', 'OnlineBackup', 'StreamingMovies', 'OnlineSecurity', 'TechSupport', 'StreamingTV', 'Contract', 'InternetService', 'MultipleLines', 'PaymentMethod']

        # 원핫 인코딩 적용
        df = pd.get_dummies(df, columns=binary_columns + non_binary_columns)

        # 불필요한 컬럼 제거
        drop_columns = ['No internet service', 'No phone service', 'Month-to-month', '_No', 'Credit card (automatic)']
        for drop_col in drop_columns:
            df = df.drop(df.filter(regex=drop_col), axis=1)

        # TotalCharges 결측값을 0으로 처리하고, 숫자로 변환
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(0, inplace=True)

        # tenure도 숫자로 변환
        df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
        df['tenure'].fillna(0, inplace=True)

        # 컬럼 이름에서 공백을 밑줄로 대체
        df.columns = df.columns.str.replace(' ', '_')

        # 새로운 컬럼 state 생성 및 조건별 버튼 설정
        df['State'] = ''  # state 컬럼에 버튼을 추가할 빈 문자열로 초기화
        df['Service'] = ''

        # 버튼 추가 조건
        df.loc[df['tenure'] >= 6, 'State'] += '<button class="btn btn-sm btn-tenure" style="background-color: #012970; color: white;">Tenure</button> '
        df.loc[df['TotalCharges'] >= 1000, 'State'] += '<button class="btn btn-sm btn-totalcharges" style="background-color: #012970; color: white;">TotalCharges</button> '
        if 'MultipleLines_Yes' in df.columns:
            df.loc[df['MultipleLines_Yes'] == 1, 'State'] += '<button class="btn btn-sm btn-multiplelines" style="background-color: #012970; color: white;">MultipleLines</button> '



        for index, row in df.iterrows():
            button_html = '<div style="display: flex; gap: 10px;">'
            if row['tenure'] >= 6:  # tenure가 6 이상일 때
                df.at[index, 'Service'] += '<a href="/Message/{0}&tenure" class="btn btn-sm btn-tenure-2" style="background-color: #012970; color: white;">Tenure</a> '.format(row['customerID'], row['tenure'])
            
            if row['TotalCharges'] >= 1000:  # TotalCharges가 1000 이상일 때
                df.at[index, 'Service'] += '<a href="/Message/{0}&TotalCharges" class="btn btn-sm btn-totalcharges-2" style="background-color: #012970; color: white;">TotalCharges</a> '.format(row['customerID'], row['TotalCharges'])
            
            if 'MultipleLines_Yes' in df.columns and row['MultipleLines_Yes'] == 1:  # MultipleLines_Yes가 1일 때
                df.at[index, 'Service'] += '<a href="/Message/{0}&MultipleLines_Yes" class="btn btn-sm btn-multiplelines-2" style="background-color: #012970; color: white;">MultipleLines</a> '.format(row['customerID'], row['MultipleLines_Yes'])

        # 필요한 컬럼만 선택
        selected_columns = df[['customerID', 'MonthlyCharges', 'TotalCharges', 'Contract_One_year', 'MultipleLines_Yes', 'tenure', 'State', 'Service']]
    

        search_query = request.GET.get('search', '')
        if search_query:
            mask = selected_columns.apply(lambda row: row.astype(str).str.contains(search_query, case=False).any(), axis=1)
            selected_columns = selected_columns[mask]

        rows = selected_columns.values.tolist()

        per_page = request.GET.get('per-page', 10)
        paginator = Paginator(rows, per_page)
        page_number = request.GET.get('page', 1)
        page_obj = paginator.get_page(page_number)

        columns = selected_columns.columns.tolist()

        current_group = (page_obj.number - 1) // 10
        start_page = current_group * 10 + 1
        end_page = min(start_page + 9, paginator.num_pages)

        return render(request, 'manager/CustomerTable.html', {
            'columns': columns,
            'page_obj': page_obj,
            'page_range': range(start_page, end_page + 1),
            'has_previous': page_obj.has_previous(),
            'has_next': page_obj.has_next(),
            'per_page': per_page,
            'search_query': search_query,
        })

    except FileNotFoundError:
        return render(request, 'manager/CustomerTable.html', {'error': '파일을 찾을 수 없습니다.'})


def Charts(request):
    return render(request, 'manager/Charts.html')

