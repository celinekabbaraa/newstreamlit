
from zoneinfo import available_timezones
import streamlit as st
import numpy as np
import pandas as pd

import functools
from locale import D_FMT

from pickletools import float8
from sqlite3 import DatabaseError
from statistics import multimode
#from msilib import datasizemask
#from pathlib import Path

import streamlit as st
#from st_aggrid import AgGrid
#from st_aggrid.shared import JsCode
#from st_aggrid.grid_options_builder import GridOptionsBuilder
import plotly as plt
import plotly.express as px
#from joblib import load
import pickle
import time

st.set_page_config(
    page_title="Marketing Campain Dashboard & Prediction",
    page_icon="📳",
    layout="wide",
)

st.markdown('<div class="header"> <H1 align="center"><font style="style=color:lightblue; ">  Campain Prediction and Analysis</font></H1></div>', unsafe_allow_html=True)
chart = functools.partial(st.plotly_chart, use_container_width=True)

def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')


def main() -> None:
    
    Dashboard= ['Campaign Performance Dashboard', 'Customer Dashboard', 'Sales Dashboard', 'Predictive model', 'Choose your Dataset']
    choice= st.sidebar.selectbox("Select Dashboard", Dashboard)
    
    if choice == 'Choose your Dataset':
        st.subheader("Upload your Dataset")
        uploaded_data = st.file_uploader(
            "Drag and Drop or Click to Upload", type=".csv", accept_multiple_files=False)
        st.subheader("Choose your Dataset")
        if uploaded_data is None:
            st.info("Using example data. Upload a file above to use your own data!")
            uploaded_data = open("cleaned3.csv", "r")
        else:
            st.success("Uploaded your file!")
            df = pd.read_csv(uploaded_data)
            with st.expander("Raw Dataframe"):
                st.write(df)
            csv = convert_df(df)
    elif choice == 'Sales Dashboard':
        uploaded_data = open("cleaned3.csv", "r")
        df = pd.read_csv(uploaded_data)
        csv = convert_df(df)
        st.subheader("Sales Dashboard")
        income_filter,age_filter,minor_filter = st.columns(3)
        income_range = list(df.Income.unique())
        with income_filter:
            income_filter = st.selectbox("Select the Income range", income_range)
        df = df[df["Income"] == income_filter]
        age_range = list(df.Customer_age.unique())
        with age_filter:
            age_filter = st.selectbox("Select the age range", age_range)
        df = df[df["Customer_age"] == age_filter]
        minor_number = list(df.minor_home.unique())
        with minor_filter:
            minor_filter = st.selectbox("Select the minor at home number", minor_number)
        df = df[df["minor_home"] == minor_filter]
        # create three columns
        kpi1, kpi2, kpi3 = st.columns(3)
        # fill in those three columns with respective metrics or KPIs
        Unique_Customers=df.ID.nunique()
        def Average(l): 
            avg = sum(l) / len(l) 
            return avg
        Average_AOV=Average(df.AOV)
        Total_spending= sum(df.Tot_amnt_spent)
        Average_spending=Average(df.Tot_amnt_spent)
        
        kpi1.metric(
            label="Unique Customers",
            value= Unique_Customers,
             )
        kpi2.metric(
            label="Average Order Value in $",
            value= round(Average_AOV),
            )
        kpi3.metric(
            label="Total Spending Value in $",
            value= round(Total_spending),
            )
        Purchase_day=df.last_purchase_day_type
        fig_col1, fig_col2, fig_col3 = st.columns(3)
        with fig_col1:
            fig = px.histogram(
                data_frame=df, x=Purchase_day, title="Puchase trend by Day"
            )   
            st.write(fig)
        with fig_col2:
            fig = px.bar(df, x=df.Response, y=df.AOV
            )
            st.write(fig)
        with fig_col3:
            fig = px.scatter(df, x="Recency", y="NumWebVisitsMonth", trendline="ols"
            )
            st.write(fig)

        df['MntWines'] = (df['MntWines']* df['Tot_amnt_spent'])
        df['MntFruits'] = (df['MntFruits']* df['Tot_amnt_spent'])
        df['MntMeatProducts'] = (df['MntMeatProducts']* df['Tot_amnt_spent'])
        df['MntFishProducts'] = (df['MntFishProducts']* df['Tot_amnt_spent'])
        df['MntSweetProducts'] = (df['MntSweetProducts']* df['Tot_amnt_spent'])
        df['MntGoldProds'] = (df['MntGoldProds']* df['Tot_amnt_spent'])

        spenditure = df[['MntWines','MntFruits','MntMeatProducts','MntFishProducts', 'MntSweetProducts','MntGoldProds']]

        chart_data = pd.DataFrame(
            df,
            columns=["MntWines", "MntFruits", "MeatProducts","MntFishProducts", "MntSweetProducts","MntGoldProds"])
        st.bar_chart(chart_data)

    elif choice == 'Customer Dashboard':
        uploaded_data = open("cleaned3.csv", "r")
        df = pd.read_csv(uploaded_data)
        csv = convert_df(df)
        st.subheader("Customer Dashboard")
        fig_col5, fig_col6 = st.columns(2)
        with fig_col5:
            fig_col5 = px.histogram(
                data_frame=df, x="Marital_Status", title="Marital Status"
            )
            st.write(fig_col5)
        with fig_col6:
            fig_col6 = px.histogram(
                data_frame=df, x="Education", title="Education Level"
            )
            st.write(fig_col6)
        fig_col11, fig_col12 = st.columns(2)
        with fig_col11:
            fig= px.box(
                df, x="Complain", title='number of complaints in the last two years'
            )
            st.write(fig)
        with fig_col12:
            fig = px.scatter(
                df, x=df.Tot_amnt_spent, y=df.Income, title="Total amount spent per income"
            )
            st.write(fig)



    elif choice == 'Predictive model':
        uploaded_data = open("cleaned3.csv", "r")
        df = pd.read_csv(uploaded_data)
        csv = convert_df(df)
        st.subheader("Predictive model")
        #Predictive analysis
        from sklearn.linear_model import LogisticRegression
        #Feature choice 
        y = df['Response']
        X = df[["minor_home","Tot_amnt_spent"]]
        #Train test models
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        from sklearn.model_selection import train_test_split
        print (X_train.shape,X_test.shape,y_train.shape, y_test.shape)

        #Logistic Regression
        clf_LR = LogisticRegression()
        clf_LR.fit(X_train,y_train)
        y_test_pred = clf_LR.predict(X_test)
        from sklearn.metrics import accuracy_score, confusion_matrix
        confusion_matrix(y_test, y_test_pred)
        accuracy_score(y_test, y_test_pred)
        from sklearn import preprocessing
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_s= scaler.transform(X_train)
        scaler = preprocessing.StandardScaler().fit(X_test)
        X_test_s= scaler.transform(X_test)
        X_test_s

        model = LogisticRegression()
        model.fit(X_train, y_train)

    

        import plotly.graph_objects as go

        fig = go.Figure([
        go.Scatter(x=X_train_s.squeeze(), y=y_train, 
                   name='train', mode='markers'),
        go.Scatter(x=X_test.squeeze(), y=y_test, 
                   name='test', mode='markers'),
        go.Scatter(x=X_test_s, y=y_test_pred, 
                   name='prediction')]
        )
        st.write(fig)

        #KNN 
        from sklearn.neighbors import KNeighborsClassifier
        clf_knn_1 = KNeighborsClassifier(n_neighbors=2)
        clf_knn_1.fit(X_train_s, y_train)
        confusion_matrix(y_test, clf_knn_1.predict(X_test_s))
        accuracy_score(y_test, clf_knn_1.predict(X_test_s))
        from sklearn.model_selection import GridSearchCV
        params = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20,30]}
        
        #Optimized KNN
        grid_search_cv = GridSearchCV(KNeighborsClassifier(), params)
        grid_search_cv.fit(X_train_s, y_train)
        grid_search_cv.best_params_
        optimised_KNN = grid_search_cv.best_estimator_
        y_test_pred = optimised_KNN.predict(X_test_s) 
        confusion_matrix(y_test, y_test_pred)
        accuracy_score(y_test, y_test_pred)

    elif choice == 'Campaign Performance Dashboard':
        uploaded_data = open("cleaned3.csv", "r")
        df = pd.read_csv(uploaded_data)
        csv = convert_df(df)
        st.subheader("Campaign Performance Dashboard")
        income_range = list(df.Income.unique())
        income_filter = st.selectbox("Select the Income range", income_range)
        df = df[df["Income"] == income_filter]
        fig_col7, fig_col8 = st.columns(2)
        with fig_col7:
            fig = px.histogram(
                data_frame=df, x="total_accept", y=df.AOV, title="number of campains accepted"
            )   
            st.write(fig)
        with fig_col8:
            fig = px.bar(
                df, x=df.Response, y=df.Tot_amnt_spent
            )
            st.write(fig)











        
            




        



    


if __name__ == "__main__":

    main()
