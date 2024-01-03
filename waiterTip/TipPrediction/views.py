from django.shortcuts import render
from joblib import load
model = load("../ml_model/rf_model.joblib")

def index(req):
    if req.method == 'POST':
        total_bill = req.POST['total_bill']
        sex = req.POST['sex']
        smoker = req.POST['smoker']
        time = req.POST['time']
        tip_percentage = req.POST['tip_percentage']
        y_pred = model.predict([[total_bill, sex, smoker,time, tip_percentage]]).round(2)
        return render(req, 'index.html', {'result' : y_pred[0]})
    return render(req, "index.html")
