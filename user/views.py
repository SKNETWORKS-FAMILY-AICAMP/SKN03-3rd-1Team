import csv
from django.conf import settings
import os
#from django.shortcuts import render
from django.shortcuts import render, get_object_or_404, redirect
from django.utils import timezone
from .models import Customer, Email

def Message(request, customer_id):
    customer = get_object_or_404(Customer, customerID=customer_id)
    # 이미 특정 주제의 이메일이 생성되었는지 확인
    existing_email = Email.objects.filter(customer=customer, subject='특별한 혜택 안내').exists()
    url_path = request.path
    
    print("url_path : ",url_path)
    # 조건에 맞는 고객에게만 이메일 생성
    if 'tenure' in url_path:
        Email.objects.create(
            sender_name='Admin',
            subject='특별한 혜택 - 6개월 이상 이용하신 고객님께 드리는 할인쿠폰!!',
            content='',
            received_date=timezone.now(),
            is_read=False,
            customer=customer
        )
        
    elif 'TotalCharges' in url_path:
        Email.objects.create(
            sender_name='Admin',
            subject='특별한 혜택 - 우리 통신사를 사랑해주신 고객님께 드리는 할인 혜택!!',
            content='',
            received_date=timezone.now(),
            is_read=False,
            customer=customer
        )
    
    elif 'MultipleLines_Yes' in url_path:
        Email.objects.create(
            sender_name='Admin',
            subject='특별한 혜택 - 기기 2개 이상 개통하신 고객님께 드리는 혜택!!',
            content='',
            received_date=timezone.now(),
            is_read=False,
            customer=customer
        )

    # 해당 고객의 이메일 목록을 필터링
    emails = Email.objects.filter(customer=customer).order_by('-received_date')
    return render(request, 'user/Message.html', {'emails': emails, 'customer': customer})

def read_msg(request, customer_id, email_id):
    customer = Customer.objects.get(customerID=customer_id)
    email = get_object_or_404(Email, id=email_id, customer__customerID=customer_id)
    print("="*20)
    print(email)
    print("="*20)
    #email = get_object_or_404(Email, id=email_id)
    if not email.is_read: 
        email.is_read = True
        email.save()
    """원래 코드"""
    # return render(request, 'user/read_msg.html', {'email': email})
    
    """이걸로 하면, 나와야하는 내용을 간단하게 html로 만들어봄"""
    # return render(request, 'user/test_read_msg.html', {'email': email})
    
    """원래 하려고 했던 코드가 돌아가게 만듦 (단, 내용은 넣지 않음 - test_read_msg.html 을 보면서 넣어야할 듯)"""
    return render(request, 'user/read_msg.html', {'customer': customer, 'email': email})

def Coupon(request, customer_id):
    customer = get_object_or_404(Customer, customerID=customer_id)
    # if email_id.is_downloaded == False:
    #     email_id.save()
    # else:
    #     print("이미 발급된 쿠폰입니다.")
    return render(request, 'user/Coupon.html',  {'customer': customer})

def Profile(request, customer_id):
    customer = get_object_or_404(Customer, customerID=customer_id)
    return render(request, 'user/Profile.html',  {'customer': customer})
