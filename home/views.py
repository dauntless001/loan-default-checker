from django.shortcuts import render
from django.views.generic.base import TemplateView
from django.http import JsonResponse
# from predict import predict_Loan_Status
# Create your views here.
class IndexView(TemplateView):
    template_name = "index.html"

    def post(self, request, *args, **kwargs):
        data = {
            "msg" : "Loan applicant will default",
            'status' : False
        }
        # try:
        #     resp = predict_Loan_Status()
        #     if resp:
        #         data["status"] = resp
        #         data["msg"] = "Loan applicant will not default"
        # except Exception as e:
        #     pass
        return JsonResponse(data, safe=False)