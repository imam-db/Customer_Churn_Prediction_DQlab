import streamlit as st
import pandas as pd 
import pickle
import numpy as np

with open('best_model_churn.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

def valreplace(data):
        replace_dict = {'Yes' : 1, 'No' : 0, 'Female' : 1, 'Male' : 0}
        return data.replace(replace_dict)

add_selectbox = st.sidebar.selectbox( "How would you like to predict?", ("Online", "Batch"))
st.sidebar.info('**This app is created to predict Customer Churn at DQlab Telco**')

def main():
    st.title("DQLab Telco Customer Churn Prediction") 
    st.header('Customer Data Input')  
    if add_selectbox == 'Online':
        gender = st.radio("**Customer's gender identity**", ('Female', 'Male'))
        SeniorCitizen = st.radio('**Is the customer a senior citizen?**', ('Yes', 'No'))
        Partner = st.radio('**Does the customer have a partner?**', ('Yes', 'No'))
        tenure = st.number_input('**Tenure (in month)**', 0, 130, 0)
        PhoneService = st.radio('**Does the customer have a phone service?**', ('Yes', 'No'))
        StreamingTV = st.radio('**Does the customer have a streaming TV?**', ('Yes', 'No'))
        InternetService = st.radio('**Does the customer have an interner service**', ('Yes', 'No'))
        PaperlessBilling = st.radio('**Does the customer have paperless billing?**', ('Yes', 'No'))
        MonthlyCharges = st.number_input("**Customer's monthly charges**", 0, 170, 0)
        TotalCharges = MonthlyCharges * tenure

        preview_data = pd.DataFrame([[gender, SeniorCitizen, Partner, tenure, PhoneService, StreamingTV, InternetService, PaperlessBilling, MonthlyCharges, TotalCharges]], 
                                    columns = ['gender', 'SeniorCitizen', 'Partner', 'tenure', 'PhoneService', 'StreamingTV', 'InternetService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges'])

        if st.button("Preview Data"):
            st.dataframe(preview_data)

        # Convert value
        data_predict = valreplace(preview_data)

        if st.button('Predict'):
            predicted_churn = loaded_model.predict(data_predict)
            st.success('**Churn : Yes**' if predicted_churn == 1 else '**Churn : No**')

    if add_selectbox == 'Batch':
    # Create a file uploader widget
        file_upload = st.file_uploader("Upload CSV file for predictions", type=["csv"])
        if file_upload is not None:
            new_data = pd.read_csv(file_upload, sep=';')

            if st.button('Predict'):
                data_predict = valreplace(new_data)

                # Create an empty list to store predictions
                predictions = []

                # Loop through each row in the DataFrame and make predictions
                for index, row in data_predict.iterrows():
                    # Assuming 'loaded_model' is already loaded with your model
                    predicted_churn = loaded_model.predict(row.values.reshape(1, -1))
                    predictions.append('Yes' if predicted_churn[0] == 1 else 'No')

                # Add a new column 'Predicted Churn' to the DataFrame
                new_data['Predicted_Churn'] = predictions

                # Display the DataFrame with predictions
                st.dataframe(new_data)
               
                # Churn rate 
                churn_rate = round((new_data['Predicted_Churn'].value_counts()['Yes'] / len(new_data)) * 100, 2)
                st.success(f'**Churn Rate :** {churn_rate}%')
   
if __name__ == "__main__":
    main()
     
   
        
    
    


    
