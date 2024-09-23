from django.db import models
from django.utils import timezone

class Customer(models.Model):
    customerID = models.CharField(max_length=50, primary_key=True)
    gender = models.CharField(max_length=6, choices=[('Male', 'Male'), ('Female', 'Female')])
    senior_citizen = models.BooleanField()  # SeniorCitizen은 0 또는 1로 표시되어 있으니 BooleanField로 변환
    partner = models.BooleanField()
    dependents = models.BooleanField()
    tenure = models.IntegerField()  # 계약 기간(개월 수)
    
    # 전화 서비스 관련 필드
    phone_service = models.BooleanField()
    multiple_lines = models.CharField(max_length=20, choices=[('No', 'No'), ('Yes', 'Yes'), ('No phone service', 'No phone service')])
    
    # 인터넷 서비스 관련 필드
    internet_service = models.CharField(max_length=20, choices=[('DSL', 'DSL'), ('Fiber optic', 'Fiber optic'), ('No', 'No')])
    online_security = models.BooleanField()
    device_protection = models.BooleanField()
    tech_support = models.BooleanField()
    streaming_tv = models.BooleanField()
    streaming_movies = models.BooleanField()
    
    # 계약 및 결제 정보
    contract = models.CharField(max_length=20, choices=[('Month-to-month', 'Month-to-month'), ('One year', 'One year'), ('Two year', 'Two year')])
    paperless_billing = models.BooleanField()
    payment_method = models.CharField(max_length=50, choices=[
        ('Electronic check', 'Electronic check'),
        ('Mailed check', 'Mailed check'),
        ('Bank transfer (automatic)', 'Bank transfer (automatic)'),
        ('Credit card (automatic)', 'Credit card (automatic)')
    ])
    
    monthly_charges = models.DecimalField(max_digits=10, decimal_places=2)
    total_charges = models.DecimalField(max_digits=10, decimal_places=2)
    churn = models.BooleanField()  # 고객 이탈 여부

    def __str__(self):
        return self.customerID



class Email(models.Model):
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE, null=True) # ForeingKey 추가
    sender_name = models.CharField(max_length=100)
    subject = models.CharField(max_length=255)
    content = models.TextField(default="축하합니다 ! 쿠폰이 발급되었습니다, ")
    received_date = models.DateTimeField(default=timezone.now)
    is_read = models.BooleanField(default=False)

    def __str__(self):
        return f'{self.sender_name} - {self.subject}'
    



class Coupon(models.Model):
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE)
    code = models.CharField(max_length=50)
    #is_downloaded = models.BooleanField(default=False)  # 다운로드 여부
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.code

