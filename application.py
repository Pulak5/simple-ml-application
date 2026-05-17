from flask import Flask, request,render_template
import pickle
import numpy as np

application = Flask(__name__)
svr=pickle.load(open("models/svr.pkl","rb"))
rscaler=pickle.load(open("models/regressionscaler.pkl","rb"))
cscaler=pickle.load(open("models/classificationscaler.pkl","rb"))
encoder_day=pickle.load(open("models/encoder.pkl","rb"))
logistic=pickle.load(open("models/logistic.pkl","rb"))

@application.route("/")
def home():
    return render_template("index.html")

@application.route("/predict_tips",methods=['GET','POST'])
def predict_tips():
    if request.method=='POST':
        total_bill=float(request.form.get("total_bill"))
        sex=request.form.get("sex")
        smoker=request.form.get("smoker")
        day=request.form.get("day")
        time=request.form.get("time")
        size=int(request.form.get("size"))
        if sex=="Male":
            sex=1
        else:
            sex=0
        if smoker=="Yes":
            smoker=0
        else:
            smoker=1
        if time=="Dinner":
            time=1
        else:
            time=0
        day=encoder_day.transform([[day]]).toarray()
        a=np.array([[total_bill,sex,smoker,time,size,day[0][0],day[0][1],day[0][2],day[0][3]]])
        x=rscaler.transform(a)
        result1=svr.predict(x)
        return render_template('predict_tips.html',result1=result1[0])
        
    else:
        return render_template("predict_tips.html")

@application.route("/predict_smoker",methods=['GET','POST'])
def predict_smoker():
    if request.method=='POST':
        total_bill=float(request.form.get("total_bill"))
        sex=request.form.get("sex")
        tip=float(request.form.get("tip"))
        day=request.form.get("day")
        time=request.form.get("time")
        size=int(request.form.get("size"))
        if sex=="Male":
            sex=1
        else:
            sex=0
        if time=="Dinner":
            time=1
        else:
            time=0
        day=encoder_day.transform([[day]]).toarray()
        a=np.array([[total_bill,tip,sex,time,size,day[0][0],day[0][1],day[0][2],day[0][3]]])
        x=cscaler.transform(a)
        result=logistic.predict(x)
        if result[0]==1:
            result="No"
        else:
            result="Yes"
        return render_template('predict_smoker.html',result=result)
    
    else:    
        return render_template("predict_smoker.html")


if __name__ == "__main__":
    application.run(debug=True,host="0.0.0.0")