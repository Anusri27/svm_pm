import numpy as np
from flask import Flask, request, jsonify, render_template, send_file, Response
import pickle
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
import pymysql
import secrets
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO

conn="mysql+pymysql://{0}:{1}@{2}/{3}".format(secrets.dbuser,secrets.dbpass,secrets.dbhost,secrets.dbname)

app = Flask(__name__)
app.config['SECRET_KEY']='SuperSecretKey'
app.config['SQLALCHEMY_DATABASE_URI']=conn
db=SQLAlchemy(app)
model = pickle.load(open('model.pkl', 'rb'))

SQL_Query = pd.read_sql_query('''select * from vals''', conn)
print(SQL_Query.info())
SQL_Query['time']=pd.to_datetime(SQL_Query['time'])
SQL_Query=SQL_Query.interpolate(method='linear', axis=0).ffill().bfill()
print(SQL_Query.head())
data=SQL_Query[['time','temperature','xaxisvelocity','zaxisvelocity']]
cols = list(data)[1:6]
X= data[cols[1:]]
Y=data[cols[0]]
train_dates =data['time']
forecast_period_dates = pd.date_range(list(train_dates)[-1], periods=17280, freq='5s').tolist()
future = pd.DataFrame(forecast_period_dates)
future.columns = ['time']
future['time']= pd.to_datetime(future['time'])
datess=pd.DataFrame({'time':forecast_period_dates})

Query = pd.read_sql_query('''select * from predictedtemp''', conn)
Query['time']=pd.to_datetime(Query['time'])
dt=Query[['time','predictions']]
acceptable_temp=dt[dt['predictions']<27.5]
unacceptable_temp=dt[dt['predictions']>=27.5]
g=acceptable_temp['time']
h=unacceptable_temp['time']
i=acceptable_temp['predictions']
j=unacceptable_temp['predictions']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    prediction = model.predict(X[-17280:])

    output = round(prediction[0], 2)

    df_op=pd.DataFrame({'predictions' :prediction})
    result=pd.concat([datess,df_op],axis=1)
    result.to_sql('predictedtemp', conn, method='multi',index=True, if_exists='replace')

    return render_template('index.html', prediction_text='Predictions = {} , Saved to MYSQL DATABASE'.format(prediction))

@app.route('/plot.png',methods=['GET','POST'])
def plot():
    fig, ax3 = plt.subplots(figsize=(20,6))
    # ax3.plot(g,i,color='blue',label='Acceptable')
    ax3.scatter(h,j,color='red',label='Unacceptable')
    plt.ylim(0,100)
    ax3.legend()
    ax3.set(xlabel="Date", ylabel="temperature",
           title="Coating Exhaust")
    # canvas= FigureCanvas(fig)
    # img=BytesIO()
    # fig.savefig('images/temp.png')
    # img.seek(0)
    # return render_template('plot.html', url='/images/temp.png')
    output = BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True)
