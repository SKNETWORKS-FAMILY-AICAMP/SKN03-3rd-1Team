import pandas as pd
from user.models import Customer

# CSV 파일 경로
csv_file_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

# CSV 파일 읽기
data = pd.read_csv(csv_file_path)

# 데이터베이스에 추가
for index, row in data.iterrows():
    Customer.objects.create(
        customerID=row['customerID'],
        gender=row['gender'],
        senior_citizen=bool(row['SeniorCitizen']),
        partner=row['Partner'] == 'Yes',
        dependents=row['Dependents'] == 'Yes',
        tenure=row['tenure'],
        phone_service=row['PhoneService'] == 'Yes',
        multiple_lines=row['MultipleLines'],
        internet_service=row['InternetService'],
        online_security=row['OnlineSecurity'] == 'Yes',
        device_protection=row['DeviceProtection'] == 'Yes',
        tech_support=row['TechSupport'] == 'Yes',
        streaming_tv=row['StreamingTV'] == 'Yes',
        streaming_movies=row['StreamingMovies'] == 'Yes',
        contract=row['Contract'],
        paperless_billing=row['PaperlessBilling'] == 'Yes',
        payment_method=row['PaymentMethod'],
        monthly_charges=row['MonthlyCharges'],
        total_charges=row['TotalCharges'],
        churn=row['Churn'] == 'Yes'
    )
