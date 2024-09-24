import csv
from django.conf import settings
import os
#from django.shortcuts import render
from django.shortcuts import render, get_object_or_404, redirect
from django.utils import timezone
from .models import Customer, Email
from django.http import JsonResponse


def download_coupon(request, customer_id):
    # 쿠폰 다운로드 처리 로직 (예: 쿠폰 발급 혹은 다운로드 기록 저장)
    
    # 성공적으로 쿠폰을 처리한 후
    return JsonResponse({'status': 'success'})  # 성공 응답



def Message(request, customer_id):
    customer = get_object_or_404(Customer, customerID=customer_id)
    # 이미 특정 주제의 이메일이 생성되었는지 확인
    existing_email = Email.objects.filter(customer=customer, subject='특별한 혜택 안내').exists()
    url_path = request.path
    
    print("url_path : ",url_path)
    # 조건에 맞는 고객에게만 이메일 생성

    coupon_message = """
    오랫동안 저희 서비스를 사랑해주셔서 감사합니다! 🎉

    고객님께서 6개월 이상 저희 서비스를 이용해주신 것에 대한 감사의 마음으로 특별한 할인 혜택을 준비했습니다!
    모든 상품 및 서비스에서 10% 할인을 제공해드립니다. 앞으로도 저희와 함께 해주시길 바라며, 더 많은 혜택을 드릴 수 있도록 최선을 다하겠습니다.
    """
    coupon_message2 = """
    저희 통신사를 꾸준히 이용해주시고, 소중한 관심을 보내주신 고객님께 깊은 감사를 드립니다. 🍀

    고객님의 꾸준한 이용에 감사하는 의미로 모든 상품 및 서비스에서 15% 할인 혜택을 제공해드립니다.
    앞으로도 저희 서비스를 지속적으로 이용해주시는 고객님께 더 많은 혜택을 드릴 수 있도록 노력하겠습니다.
    """
    coupon_message3 = """
    기기 2개 이상을 개통해주신 고객님께 드리는 특별한 혜택입니다! ⭐
    
    다양한 기기 개통에 감사드리며 모든 액세서리 및 부가 상품에서 20% 할인을 제공해드립니다. 
    저희와 함께 하신 시간을 감사하게 생각하며, 앞으로도 다양한 혜택을 준비하겠습니다.
    """

    if 'tenure' in url_path:
        Email.objects.create(
            sender_name='Admin',
            subject='특별한 혜택 - 6개월 이상 이용하신 고객님께 드리는 할인쿠폰!!',
            content=coupon_message,
            received_date=timezone.now(),
            is_read=False,
            customer=customer
        )
        
    elif 'TotalCharges' in url_path:
        Email.objects.create(
            sender_name='Admin',
            subject='특별한 혜택 - 우리 통신사를 사랑해주신 고객님께 드리는 할인 혜택!!',
            content=coupon_message2,
            received_date=timezone.now(),
            is_read=False,
            customer=customer
        )
    
    elif 'MultipleLines_Yes' in url_path:
        Email.objects.create(
            sender_name='Admin',
            subject='특별한 혜택 - 기기 2개 이상 개통하신 고객님께 드리는 혜택!!',
            content=coupon_message3,
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
