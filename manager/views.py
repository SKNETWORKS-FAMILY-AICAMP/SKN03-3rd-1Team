from django.shortcuts import render
import pandas as pd
import os
from django.conf import settings
from django.core.paginator import Paginator
import seaborn as sns
import matplotlib.pyplot as plt
from django.http import HttpResponse
from io import BytesIO


def CustomerTable(request):
    try:
        # CSV 파일 경로
        csv_file_path = os.path.join(settings.BASE_DIR, 'static', 'assets', 'data', 'test_pred.csv')
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

        # Churn_Probability를 숫자로 변환하고, 결측값을 0으로 채움
        df['Churn_Probability'] = pd.to_numeric(df['Churn_Probability'], errors='coerce')
        df['Churn_Probability'].fillna(0, inplace=True)

        # 컬럼 이름에서 공백을 밑줄로 대체
        df.columns = df.columns.str.replace(' ', '_')

        # 새로운 컬럼 state 생성 및 조건별 버튼 설정
        df['State'] = ''  # state 컬럼에 버튼을 추가할 빈 문자열로 초기화
        df['Service'] = ''

        # 버튼 추가 조건
        for index, row in df.iterrows():
            if row['tenure'] >= 6:  # tenure가 6 이상일 때
                df.at[index, 'State'] += '<a href="/Tenure-churn-chart/{0}/" class="btn btn-sm btn-tenure" style="background-color: #012970; color: white;">Tenure</a> '.format(row['customerID'])

        for index, row in df.iterrows():
            if row['TotalCharges'] >= 1000:
                df.at[index, 'State'] += '<a href="/TotalCharges-churn-chart/{0}/" class="btn btn-sm btn-totalcharges" style="background-color: #012970; color: white;">TotalCharges</a> '.format(row['customerID'])

        if 'MultipleLines_Yes' in df.columns:
            df.loc[df['MultipleLines_Yes'] == 1, 'State'] += '<button class="btn btn-sm btn-multiplelines" style="background-color: #808080; color: white;">MultipleLines</button>'


        for index, row in df.iterrows():
            button_html = '<div style="display: flex; gap: 10px;">'
            if row['tenure'] >= 6:  # tenure가 6 이상일 때
                df.at[index, 'Service'] += '<a href="Message/{0}&tenure" class="btn btn-sm btn-tenure-2" style="background-color: #012970; color: white;">Tenure</a> '.format(row['customerID'])
            
            if row['TotalCharges'] >= 1000:  # TotalCharges가 1000 이상일 때
                df.at[index, 'Service'] += '<a href="Message/{0}&TotalCharges" class="btn btn-sm btn-totalcharges-2" style="background-color: #012970; color: white;">TotalCharges</a> '.format(row['customerID'])
            
            if 'MultipleLines_Yes' in df.columns and row['MultipleLines_Yes'] == 1:  # MultipleLines_Yes가 1일 때
                df.at[index, 'Service'] += '<a href="Message/{0}&MultipleLines_Yes" class="btn btn-sm btn-multiplelines-2" style="background-color: #012970; color: white;">MultipleLines</a> '.format(row['customerID'])

        # Churn_Probability 0.8 이상인 데이터만 필터링
        filtered_df = df[df['Churn_Probability'] >= 0.8]

        # 필요한 컬럼만 선택
        selected_columns = filtered_df[['customerID', 'tenure', 'TotalCharges','MultipleLines_Yes', 'Churn_Probability','State', 'Service']]
    
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

def churn_chart_tenure(request, customer_id):
    # CSV 파일 경로
    csv_file_path = os.path.join(settings.BASE_DIR, 'static', 'assets', 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df = pd.read_csv(csv_file_path)
    # Churn이 Yes일 때 tenure의 빈도수 계산
    tenure_counts_churn_yes = df[df['Churn'] == 'Yes']['tenure'].value_counts().sort_index()

    # Churn이 No일 때 tenure의 빈도수 계산
    tenure_counts_churn_no = df[df['Churn'] == 'No']['tenure'].value_counts().sort_index()

    # 특정 고객의 tenure 값을 가져옴
    customer_tenure = df.loc[df['customerID'] == customer_id, 'tenure'].values[0]

    # 특정 고객의 Churn 상태를 확인
    customer_churn_status = df.loc[df['customerID'] == customer_id, 'Churn'].values[0]

    # 고객의 Churn 상태에 따라 해당 tenure의 빈도수(y값) 가져오기
    if customer_churn_status == 'Yes':
        customer_frequency = tenure_counts_churn_yes.get(customer_tenure, 0)  # Churn이 Yes일 때 빈도수
    else:
        customer_frequency = tenure_counts_churn_no.get(customer_tenure, 0)  # Churn이 No일 때 빈도수

    # Seaborn 스타일 설정
    sns.set_theme(style="whitegrid")

    # 그래프 설정
    plt.figure(figsize=(10, 6))

    # Churn이 Yes일 때 tenure의 빈도수 시각화
    sns.lineplot(x=tenure_counts_churn_yes.index, y=tenure_counts_churn_yes.values, marker='', color='red', label='Churn = Yes')

    # Churn이 No일 때 tenure의 빈도수 시각화
    sns.lineplot(x=tenure_counts_churn_no.index, y=tenure_counts_churn_no.values, marker='', color='blue', label='Churn = No')

    # 특정 고객의 tenure 값을 빈도수에 맞춰 녹색 점으로 표시
    plt.scatter([customer_tenure], [customer_frequency], color='green', s=100, zorder=5, label=f'Customer({customer_id}) Tenure : ({customer_tenure})')

    # 제목과 축 레이블 설정
    plt.title('Frequency of Tenure by Churn Status', fontsize=16)
    plt.xlabel('Tenure', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)

    # 그래프 꾸미기
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Churn Status', fontsize=12, title_fontsize=14)

    # 그래프를 메모리에 저장
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)

    return HttpResponse(buffer.getvalue(), content_type='image/png')

def churn_chart_totalCharges(request, customer_id):
    # CSV 파일 경로
    csv_file_path = os.path.join(settings.BASE_DIR, 'static', 'assets', 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df = pd.read_csv(csv_file_path)

    # TotalCharges를 숫자형으로 변환하고 결측값 처리
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)

    # 특정 고객의 TotalCharges 값을 가져옴
    customer_totalcharges = df.loc[df['customerID'] == customer_id, 'TotalCharges'].values[0]
    customer_churn = df.loc[df['customerID'] == customer_id, 'Churn'].values[0]  # Churn 상태 가져오기

    sns.set_theme(style="whitegrid")

    # 그래프 설정
    plt.figure(figsize=(10, 6))

    # Churn이 Yes일 때 TotalCharges의 KDE 시각화
    sns.kdeplot(data=df[df['Churn'] == 'Yes'], x='TotalCharges', color='red', label='Churn = Yes', fill=True)

    # Churn이 No일 때 TotalCharges의 KDE 시각화
    sns.kdeplot(data=df[df['Churn'] == 'No'], x='TotalCharges', color='blue', label='Churn = No', fill=True)

    # 고객의 TotalCharges 위치에 점 표시 (Churn 상태에 따라 색상 변경)
    if customer_churn == 'Yes':
        plt.scatter(customer_totalcharges, 0, color='green', s=100, zorder=10, label=f'Customer ID({customer_id}) TotalCharges: {customer_totalcharges} (Churn = Yes)')
    else:
        plt.scatter(customer_totalcharges, 0, color='purple', s=100, zorder=10, label=f'Customer ID({customer_id}) TotalCharges: {customer_totalcharges} (Churn = No)')

    # 제목과 축 레이블 설정
    plt.title('Distribution of TotalCharges by Churn Status', fontsize=16)
    plt.xlabel('TotalCharges', fontsize=14)
    plt.ylabel('Density', fontsize=14)

    # 그래프 꾸미기
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Churn Status', fontsize=12, title_fontsize=14)

    # 그래프를 메모리에 저장
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)

    return HttpResponse(buffer.getvalue(), content_type='image/png')




