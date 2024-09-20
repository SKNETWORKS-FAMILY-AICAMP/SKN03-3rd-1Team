from django.shortcuts import render

def CustomerTable(request):
    return render(request,'manager/CustomerTable.html')

def Charts(request):
    return render(request, 'manager/Charts.html')
