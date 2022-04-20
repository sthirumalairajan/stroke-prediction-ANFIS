#!/usr/bin/env python
# coding: utf-8

# In[2]:


from flask import Flask, render_template, request
import os
import  numpy as np
import pickle

app= Flask(__name__)


# In[3]:


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/result",methods=['POST','GET'])
def result():

    age = float(request.form['age'])
    heart_disease = float(request.form['heart_disease'])
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])
    
    x = np.array([age,heart_disease,avg_glucose_level,bmi]).reshape(1,-1)
    model = pickle.load(open('bestModel.pkl', 'rb'))
    Y_pred = model[0].predict(x)
    op = Y_pred[0][0]

    # for No Stroke Risk
    if op > -400:
        return render_template('index.html', prediction_text='Non-Stroke')
    else:
        return render_template('index.html', prediction_text='"Stroke --> at Very Risk!!! , Immedietly consult the doctor')


if __name__=="__main__":
    app.run(debug=True,port=7385)


# In[ ]:




