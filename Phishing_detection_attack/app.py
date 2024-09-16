from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings, joblib
warnings.filterwarnings('ignore')
from feature import FeatureExtraction

app = Flask(__name__)

# تحميل النموذج من القرص
try:
    gbc = joblib.load("model.pkl")
    print("Model loaded successfully.")
except FileNotFoundError:
    print("The original model file was not found.")
    gbc = None  # تعيين gbc إلى None لمنع الأخطاء إذا حاولت استخدامه في تنبؤ
except Exception as e:
    print(f"An error occurred: {e}")

# مسار الصفحة الرئيسية يدعم GET و POST
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST" and gbc:
        url = request.form["url"]
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1,30) 

        y_pred = gbc.predict(x)[0]
        y_pro_phishing = gbc.predict_proba(x)[0,0]
        y_pro_non_phishing = gbc.predict_proba(x)[0,1]

        return render_template('index.html', prediction=y_pred, xx=round(y_pro_non_phishing,2), url=url)
    else:
        return render_template("chat.html")

# مسار تنبؤ جديد لطلبات POST
@app.route("/predict_route", methods=["POST"])
def predict_route():
    if request.method == "POST" and gbc:
        url = request.form["url"]
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1, 30)

        y_pred = gbc.predict(x)[0]
        y_pro_phishing = gbc.predict_proba(x)[0, 0]
        y_pro_non_phishing = gbc.predict_proba(x)[0, 1]
        
        prediction_message = "It is safe" if y_pred == 1 else "It is not safe"
        return render_template('result.html', prediction=prediction_message, y_pro_phishing=y_pro_phishing*100, y_pro_non_phishing=y_pro_non_phishing*100, url=url)

    return "This route only supports POST requests."

if __name__ == "__main__":
    app.run(debug=True)
