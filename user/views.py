from django.shortcuts import render

def Profile(request):
    return render(request, 'user/Profile.html')