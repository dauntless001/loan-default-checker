from django.shortcuts import render
from django.views.generic.base import TemplateView
from django.http import JsonResponse
import random
# from predict import predict_Loan_Status
# Create your views here.

DATA_ID = 0
DATA_DEFAULT = 0

INDEX = 0

class IndexView(TemplateView):
    template_name = "index.html"
    file = "../code-ocean.csv"

    def post(self, request, *args, **kwargs):
        data = {
            "msg" : "Loan applicant will default [1]",
            'status' : False
        }

        global DATA_ID
        global DATA_DEFAULT
        
        id = request.POST['id']

        loan_data = {
            'term_month' : request.POST['term_months'],
            'loan_amount' : request.POST['loan_amount'],
            'emp_length' : request.POST['emp_length'],
            'annual_inc' : request.POST['annual_inc'],
            'dti' : request.POST['dti'],
            'delinq_2yrs' : request.POST['delinq_2yrs'],
            'revol_util' : request.POST['revol_util'],
            'total_acc' : request.POST['total_acc'],
            'credit_length_yrs' : request.POST['credit_length_yrs'],
            'int_rate' : request.POST['int_rate'],
            'remain' : request.POST['int_rate'],
            'issue_year' : request.POST['issue_year']
        }

        # default = predict_Loan_Status()
        if DATA_ID != id:
            DATA_DEFAULT = random.randint(0, 1)
            DATA_ID = id
            

        if DATA_DEFAULT == 0:
            data["status"] = True
            data["msg"] = "Loan applicant will not default [0]"

        return JsonResponse(data, safe=False)