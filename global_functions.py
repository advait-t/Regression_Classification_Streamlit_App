import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import random
from scipy import stats
import matplotlib.pyplot as plt
import statistics

from shapash.explainer.smart_explainer import SmartExplainer
from pycaret.classification import *

from imblearn.under_sampling import RandomUnderSampler
from lightgbm import *
from rfpimp import *
import pickle
import base64
import streamlit as st
import os
import zipfile

def highlight_cols(s):
    color = '#FF4B4B'
    return 'background-color: %s' % color

def drop_one_column_feature_importance(X_train, y_train, final_model):
    imp = importances(final_model, X_train, y_train, n_samples=-1)
    fig = px.bar(imp, x='Importance', y=imp.index, orientation='h', color='Importance',
     color_discrete_map={
        "Negative": "indianred",
        "Positive": "seagreen"})
    st.plotly_chart(fig)

def select_data():
    file = st.file_uploader('Upload Dataset', type = ['csv', 'txt'])
    if file is not None:
        data = pd.read_csv(file)
        return data

def show_density_plots(top, column, values, unique_column):
    subset = top[top[column] == unique_column]
    fig = px.histogram(subset, x = values,
                marginal="box",
                hover_data=top.columns, color = column)
    fig.update_layout(title = f'Density Plot for {unique_column}', xaxis_title = unique_column)
    st.plotly_chart(fig)

def download_model(model):
    output_model = pickle.dumps(model)
    b64 = base64.b64encode(output_model).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" download="trained_model.pkl">Download Trained Model</a>'
    st.markdown(href, unsafe_allow_html=True)

def download_configs(model):
    output_model = pickle.dumps(model)
    b64 = base64.b64encode(output_model).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" download="data_configs.pkl">Download Configurations</a>'
    st.markdown(href, unsafe_allow_html=True)

def count_plot_top_features(data, feature, target):
    column = ['feature_1', 'feature_2','feature_3']
    for i in column:
        plot = px.histogram(data, x=i, barmode='group', color = i, title = (f'Output Label {feature},{i}')).update_xaxes(categoryorder="total descending")
        st.plotly_chart(plot)

def ordinal_columns(data, target):
    columns = data.columns
    column_names = []
    values = []
    ordinal_dictionary = {}
    
    for i in range(len(columns)):
        if columns[i] != target:
            unique = list(data[columns[i]].unique())
            unique = [str(k) for k in unique]
            if len(unique)==2:
                column_names.append(columns[i])
                values.append(unique)
                
    for key in column_names:
        for value in values:
            ordinal_dictionary[key] = value
            values.remove(value)
            break
    return ordinal_dictionary

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(
        csv.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/csv;base64,{b64}" download="dataset.csv">Download Top Features File</a>'