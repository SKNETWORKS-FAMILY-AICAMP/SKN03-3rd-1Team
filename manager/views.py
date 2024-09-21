import pandas as pd
import os
from django.conf import settings
from django.shortcuts import render
import matplotlib.pyplot as plt
import seaborn as sns

def generate_graph(tenure, churn_status):
    # Seaborn 스타일 설정
    sns.set(style="whitegrid")

    # Churn 데이터 계산
    tenure_counts_churn_yes = df[df['Churn'] == 'Yes']['tenure'].value_counts().sort_index()
    tenure_counts_churn_no = df[df['Churn'] == 'No']['tenure'].value_counts().sort_index()

    # 그래프 설정
    plt.figure(figsize=(10, 6))

    # Churn이 Yes일 때 tenure의 빈도수 시각화
    sns.lineplot(x=tenure_counts_churn_yes.index, y=tenure_counts_churn_yes.values, color='red', label='Churn = Yes')

    # Churn이 No일 때 tenure의 빈도수 시각화
    sns.lineplot(x=tenure_counts_churn_no.index, y=tenure_counts_churn_no.values, color='blue', label='Churn = No')

    # 고객의 위치 표시
    if churn_status == 'Yes':
        plt.scatter(tenure, 0, color='red', s=100, label='Your Tenure')  # y축 값은 필요에 따라 조정
    else:
        plt.scatter(tenure, 0, color='blue', s=100, label='Your Tenure')  # y축 값은 필요에 따라 조정

    # 제목과 축 레이블 설정
    plt.title('Frequency of Tenure by Churn Status', fontsize=16)
    plt.xlabel('Tenure', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)

    # 그래프 꾸미기
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Churn Status', fontsize=12, title_fontsize=14)

    # 그래프 저장
    image_path = os.path.join(settings.MEDIA_ROOT, 'graphs', f'graph_{tenure}.png')
    plt.savefig(image_path)
    plt.close()
    
    return image_path


def csv_to_table(request):
    # CSV 파일 경로
    csv_file_path = os.path.join(settings.STATICFILES_DIRS[0], 'data', 'Telco-customer-churn.csv')
    
    # pandas로 CSV 파일 읽기
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
    df['state'] = ''  # state 컬럼에 버튼을 추가할 빈 문자열로 초기화

    # 1. tenure가 6 이상이면 'tenure' 버튼 추가
    df.loc[df['tenure'] >= 6, 'state'] += '<button class="button button-tenure">Tenure</button> '

    # 2. TotalCharges가 특정 기준 이상이면 'VIP 고객'으로 간주하고 'TotalCharges' 버튼 추가
    df.loc[df['TotalCharges'] >= 1000, 'state'] += '<button class="button button-totalcharges">TotalCharges</button> '

    # 3. MultipleLines_Yes가 True인 경우 'MultipleLines_Yes' 버튼 추가
    if 'MultipleLines_Yes' in df.columns:
        df.loc[df['MultipleLines_Yes'] == 1, 'state'] += '<button class="button button-multiplelines">MultipleLines</button> '

    # 필요한 컬럼만 선택 (Churn은 제외)
    columns_to_display = ['customerID', 'MonthlyCharges', 'TotalCharges', 'Contract_One_year', 'MultipleLines_Yes', 'DeviceProtection_Yes', 'tenure', 'state']
    df = df[columns_to_display]

    # 데이터프레임을 HTML로 변환 (템플릿으로 전달하기 위해)
    table_html = df.to_html(index=False, escape=False)  # escape=False는 HTML 태그를 처리하기 위해 사용
    
    if request.method == 'POST':
        tenure = int(request.POST.get('tenure'))
        churn_status = request.POST.get('churn_status')
        image_path = generate_graph(tenure, churn_status)
        return render(request, 'manager/graph_popup.html', {'image_path': image_path})

    return render(request, 'manager/manager.html', {'table_html': table_html})
