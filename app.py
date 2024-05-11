from src.MarketingCampaignClustring.pipelines.prediction_pipeline import custom_data,model_prediction
from flask import Flask,request,render_template
app=Flask(__name__,template_folder="template")
@app.route('/')
def home_page():
    return render_template("form.html")
@app.route('/predict',methods=["POST"])
def pred_page():   
        get_data=custom_data(invoice_frequency=float(request.form.get('invoice_frequency')),
                             total_quantity=float(request.form.get('total_quantity')),
                             total_bill=float(request.form.get('total_bill')),
                             day_gap= float(request.form.get('day_gap')),
                             country=request.form.get('country'))
        
        final_data=get_data.get_data_as_dataframe()
        pred=model_prediction()
        x=pred.model_pred_initiate(final_data)
        x=x[0]
        if x==0:
            x='high total_bill,low day_gap or active user.'
        elif x==1:
            x='very low total_bill,high day_gap or user is not active.'
        else:
            x='medium total_bill,medium day_gap,user may active or not.'
        return render_template("result.html",x=x)

if __name__=="__main__":
 app.run()