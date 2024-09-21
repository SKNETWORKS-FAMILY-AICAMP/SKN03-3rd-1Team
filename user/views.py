import csv
from django.conf import settings
import os
#from django.shortcuts import render
from django.shortcuts import render, get_object_or_404
from django.utils import timezone
from .models import Customer, Email


# def Message(request):
#     return render(request, 'user/Message.html')

def Message(request, customer_id):
    customer = get_object_or_404(Customer, customerID=customer_id)

    eligible_customers = Customer.objects.filter(gender='Female', email__isnull=True)

    for customer in eligible_customers:
        Email.objects.create(
            sender_name='Admin',
            subject='특별한 혜택 안내',
            received_date=timezone.now(),
            is_read=False,
            customer=customer
        )

    emails = Email.objects.filter(customer=customer).order_by('-received_date')
    return render(request, 'user/Message.html', {'emails': emails, 'customer': customer})




def read_msg(request, customer_id, email_id):
    email = get_object_or_404(Email, id=email_id, customer__customerID=customer_id)
    #email = get_object_or_404(Email, id=email_id)
    if not email.is_read: 
        email.is_read = True
        email.save()
    return render(request, 'user/read_msg.html', {'email': email})





def Profile(request, customer_id):
    customer = get_object_or_404(Customer, customerID=customer_id)
    return render(request, 'user/Profile.html',  {'customer': customer})
