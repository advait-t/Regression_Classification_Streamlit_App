
#This Python notebook houses all the general function of EDA and plots


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
# from pycaret.classification import compare_models, create_model, finalize_model, get_config, predict_model, save_config, save_model, setup, tune_model, retrain_model

try:
    from pycaret.classification import compare_models
    from pycaret.classification import create_model
    from pycaret.classification import pull
    from pycaret.classification import create_model
    from pycaret.classification import tune_model
    from pycaret.classification import finalize_model
    from pycaret.classification import get_config
    from pycaret.classification import save_model
    from pycaret.classification import load_model
    from pycaret.classification import download_model
    from pycaret.classification import download_configs
    from pycaret.classification import save_config
    from pycaret.classification import setup
except:
    pass

from imblearn.under_sampling import RandomUnderSampler
from lightgbm import *
from rfpimp import *
import pickle
import base64
import streamlit as st
import os
import zipfile


# These dtypes can be set right at the start which the user knows 
def set_column_data_types(data, categorical_columns=0, continuous_columns=0):
    if categorical_columns == 0 and continuous_columns == 0:
        pass
    else:
        data[categorical_columns] = data[categorical_columns].astype('object')
        data[continuous_columns] = data[continuous_columns].astype('float64')
    return data
    

def missing_values_table(df):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        return mis_val_table_ren_columns

def drop_or_replace_na(data,columns_to_drop = 0):   #replace_na = 0, drop_na_columns = 0,
    if columns_to_drop == 0:
        missing_values = pd.DataFrame(missing_values_table(data))
        for i in range(len(missing_values['% of Total Values'])):
            if missing_values['% of Total Values'][i] <= 7:
                cols = list(missing_values.index.values)
                data.dropna(subset = cols, inplace=True)
    else:
        data = data.drop(columns_to_drop, axis = 1)
        missing_values = pd.DataFrame(missing_values_table(data))
        for i in range(len(missing_values['% of Total Values'])):
            if missing_values['% of Total Values'][i] <= 7:
                cols = list(missing_values.index.values)
                data.dropna(subset = cols, inplace=True)
    return data

def unique_value_count(data):
    unique_counts = pd.DataFrame.from_records([(col, data[col].nunique()) for col in data.columns],columns=['Column_Name', 'Num_Unique']).sort_values(by=['Num_Unique'])
    return unique_counts

def convert_dtype(data,unique=40):
    unique_counts = pd.DataFrame.from_records([(col, data[col].nunique()) for col in data.columns],columns=['Column_Name', 'Num_Unique']).sort_values(by=['Num_Unique'])
    for i in data.columns:
        if data[i].nunique() <= unique:
            data[i] = data[i].astype('object')
        else:
            data[i] = data[i].astype('float64')
    return data

def columns_select(data,exclude):
    data = data.select_dtypes(exclude=[exclude])
    return data

def read_data(path):
    if path.lower().endswith(('.csv')):
        data = pd.read_csv(path)
    else:
        data = pd.read_parquet(path, engine='pyarrow')
    return data

def col_lis(data):
    column_names = data.columns
    return column_names

def categorical_barplots(data, column_name = []):
    
    def without_hue(plot, feature):
        total = len(feature)
        for p in plot.patches:
            percentage = '{:.1f}%'.format(100 * p.get_height()/total)
            x = p.get_x() + p.get_width() / 2 - 0.05
            y = p.get_y() + p.get_height()
            g.annotate(percentage, (x, y), size = 12)
        plt.show()
    
    if len(column_name) == 0:
        data = columns_select(data,'float64')
        data = columns_select(data,'int64')
        column_name = data.columns
    else:
        data = data[column_name]
        column_name = data.columns
        
    for i in range(len(column_name)):
        plt.figure(figsize = (25,10))
        g = sns.countplot(x = data[column_name[i]], order=data[column_name[i]].value_counts().iloc[:20].index)
        g.set(xlabel=column_name[i], ylabel = "Count of Values")
        without_hue(g, data[column_name[i]])
    return data

# Checking for any columns with special characters
# Replacing special characters with NA's
def check_special_char(data):
    special_char = ['?', '!','-','*']
    data = data.replace(special_char, np.nan)
    return data

# Drop any columns not essential for your model. 
# Make a list of column names to be dropped and pass it in the function drop_columns().
def drop_columns(data,column_names):
    data = data.drop(column_names, axis = 1)
    return data

def create_bins(data,old_column_names,new_column_names,drop_old_column,bins, labels):
    bins_items = list(bins.values())[0]
    labels_items = list(labels.values())[0]

    for i in range(len(old_column_names)):
        bins_items = list(bins.values())[i]
        labels_items = list(labels.values())[i]
        data[new_column_names[i]] = pd.cut(data[old_column_names[i]], bins = bins_items, labels = labels_items)
    if drop_old_column == 'yes':
        data=data.drop(old_column_names, axis = 1)
    else:
        pass
    return data


def numeric_xaxis_targetcolumn_yaxis_boxplots(data, target_column, continuous_column = []):
    
    if len(categorical_column) == 0:
        d1 = data[target_column]
        categorical_names = d1.columns
    else:
        d1 = data[categorical_column]
        categorical_names = d1.columns
        
    if len(continuous_column) == 0:
        d2 = columns_select(data,'object')
        continuous_names = d2.columns
    else:
        d2 = data[continuous_column]
        continuous_names = d2.columns
    
    categorical_arr = range(1,len(categorical_names)+1)
    continuous_arr = range(1,len(continuous_names)+1)
    
    for i in range(len(continuous_column)):
        Q1 = np.percentile(d2[continuous_column[i]], 25, 
                       interpolation = 'midpoint') 

        Q3 = np.percentile(d2[continuous_column[i]], 75,
                       interpolation = 'midpoint') 
        IQR = Q3 - Q1 
 

        # Upper bound
        upper = np.where(d2[continuous_column[i]] >= (Q3+1.5*IQR))
        # Lower bound
        lower = np.where(d2[continuous_column[i]] <= (Q1-1.5*IQR))
    for i in range(len(categorical_arr)):
        for j in range(len(continuous_arr)):
            x_col = categorical_names[i]
            y_col = continuous_names[j]
            plt.figure(figsize=(15,5))
            sns.boxplot(x = d1[x_col], y = d2[y_col])
            
        

def numeric_xaxis_numeric_yaxis_scatterplots(data, column_name=[]):
    if len(column_name) == 0:
        d1 = columns_select(data,'object')
        column_names = d1.columns
    else:
        d1 = data[column_name]
        column_names = d1.columns
        
    arr=range(1,len(column_names)+1)
        
    for i in range(len(arr)):
        for j in range(len(arr)):
            if i!=j:
                x_col = column_name[i]
                y_col = column_name[j]
                fig = px.scatter(d1,x =x_col, y=y_col, labels={column_names[i]:column_names[i], column_names[j]:column_names[j]})
                fig.show()

def categorical_xaxis_categorical_yaxis_boxplots(data, categorical_columns, target_column, one_or_zero):
    categorical_columns.append(target_column)
    data = data[categorical_columns]
    column_names = data.columns
    if one_or_zero == 1:
        drop = 0
    else:
        drop = 1
    cols_data = pd.DataFrame()
    for i in range(len(column_names)):
        if column_names[i] != target_column:
            order = pd.crosstab(index = data[column_names[i]],columns=data[target_column])
            order[one_or_zero] = (100. * order[one_or_zero] / order[one_or_zero].sum()).round(2)
            order1 = order.sort_values(by=one_or_zero,ascending=False)
            order1 = order1.drop(drop, axis =1)
            orderf = order1.head(10)
            orderf = orderf.reset_index()
            others = pd.DataFrame(order1[one_or_zero].iloc[11:])
            value = others[one_or_zero].sum().round(2)
            otherss = pd.DataFrame(columns = [one_or_zero])
            otherss.loc[one_or_zero] = value
            app = pd.DataFrame(columns=[one_or_zero,column_names[i]])
            app.loc[0]= [value,'Others']
            orderf = pd.concat([orderf,app], ignore_index = True)
            others = others.reset_index()
            average = []
            one_hot_columns = pd.DataFrame(columns = [column_names[i]])
            #average = statistics.mean(order[one_or_zero])
            average = statistics.mean(orderf[one_or_zero])
            plt.figure(figsize=(15,5))
            g = sns.barplot(x=orderf[column_names[i]], y = orderf[one_or_zero])
            g.axhline(average)
            for bar in g.patches:
                g.annotate(format(bar.get_height(), '.2f'), 
                              (bar.get_x() + bar.get_width() / 2, 
                               bar.get_height()), ha='center', va='center',
                              size=10, xytext=(0, 8),
                             textcoords='offset points')
            px.pyplot(g)
            columns = order[order[one_or_zero]>=average]
            columns = columns.reset_index()
            columns = columns.rename_axis(None, axis = 1)
            columns = columns.drop([drop,one_or_zero], axis = 1)
            columns = dict(columns)
            df = pd.DataFrame.from_dict(columns)
            cols_data = pd.concat([cols_data,df])
    cols_data = cols_data[cols_data.notna()]
    cols_data = cols_data.dropna(axis=1, how='all')

    return cols_data
            

def cramersv(data, display_head,categorical_columns = []):
    if len(categorical_columns) == 0:
        d1 = columns_select(data,'float64')
        column_names = d1.columns
    else:
        d1 = data[categorical_columns]
        column_names = d1.columns
        
    cramers_values = []
    cramers_overall =[]
    crames_list = []
    for i in range(len(column_names)):
        for j in range(len(column_names)):
            if i!=j:
                app_group = pd.crosstab(index=d1[categorical_columns[i]], columns=d1[categorical_columns[j]], margins = False)
                crosstab_print = pd.crosstab(index=d1[categorical_columns[i]], columns=d1[categorical_columns[j]], margins = True)
                (chi2, p, dof, _) = stats.chi2_contingency(app_group)
                cramers_value = (np.sqrt(chi2/(d1.shape[0]*(min(app_group.shape[1],app_group.shape[0])-1)))).round(2)
                cramers_values.append(cramers_value)
                cramers_overall.append(cramers_value)
                crames_list.append([cramers_value,column_names[j], column_names[i]])
                crosstab_print.loc[crosstab_print.index[0], 'Cramers Value'] = cramers_values
                if display_head == 'yes':
                    display(crosstab_print.head())
                cramers_values.clear()
    cramers_list =[]
    cramers_list = pd.DataFrame(crames_list,  columns = ['Cramers Value','X','Y'])
    df = cramers_list.pivot(index='X',columns='Y',values='Cramers Value')
    #display(df)
    sns.heatmap(df, annot=True)

def remove_correlation(data, threshold_value):
    correlated_features = set()
    correlation_matrix = data.corr()
    for i in range(len(correlation_matrix .columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold_value:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)
    print(correlated_features)
    data1 = data.drop(labels=correlated_features, axis=1)
    return data1

def correlation_plot(data): 
    plt.figure(figsize=(10,5))
    sns.heatmap(data.corr(),annot=True)
    

def kde_plot(data, continuous_cols, target):
    for i in range(len(continuous_cols)):
        plt.figure(figsize=(15,5))
        sns.kdeplot(data = data, x = continuous_cols[i], hue = target)

def final_df(data, columns_drop, cols_data0, cols_data1, variables_to_drop_from_selected_columns, final_columns_to_be_dropped):
    data = data.drop(columns_drop, axis = 1)

    sheet0 = cols_data0.drop(variable_to_drop, axis = 1)
    sheet1 = cols_data1.drop(variable_to_drop, axis = 1)

    columns = sheet1.columns
    for i in range(len(columns)):
        df = pd.get_dummies(data[columns[i]])
        cols0 = sheet0[columns[i]]
        cols0 = sheet0[columns[i]][sheet0[columns[i]].notna()]

        cols1 = sheet1[columns[i]]
        cols1 = sheet1[columns[i]][sheet1[columns[i]].notna()]

        cols = cols0.append(cols1)
        remaining = []
        for i in cols:
            if i not in remaining:
                remaining.append(i)

        df = df[remaining]
        data = pd.concat((data,df),1)
    data = data.drop(final_columns_to_be_dropped, axis = 1)
    return(data)

def setup_pycaret(data, output_variable, categorical_features_list, ordinal_features_dictionary):
    rus = RandomUnderSampler(random_state=42, replacement=True)
    setup(data = data,
    session_id = 42,
    fix_imbalance = True,
    fix_imbalance_method= rus,        
    remove_multicollinearity = True,
    multicollinearity_threshold = 0.9, 
    target = output_variable,
    categorical_features = categorical_features_list,#['quantity_lsr_ta_binned'],
    ordinal_features =ordinal_features_dictionary, #{'quantity_lsr_ta_binned':['0','1'],'Facebook':['0','1'],'Whatsapp':['0','1'], 'Youtube':['0','1'],'Facebook Video':['0','1'], 'IP_OTHER':['0','1']},
    fold = 2)
    save_config('configs_data.pkl')

def get_train_test_data():
    X_test = get_config(variable="X_test")
    X_train = get_config(variable="X_train")
    y_train = get_config(variable="y_train")
    y_test = get_config(variable="y_test")
    return(X_test, X_train,y_test,y_train)

def accuracy_f1_summary(y_test, y_pred, model):
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import cohen_kappa_score
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import confusion_matrix

    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: %f' % accuracy)
    
    # precision tp / (tp + fp)
    precision = precision_score(y_test, y_pred)
    print('Precision: %f' % precision)
    
    # recall: tp / (tp + fn)
    recall = recall_score(y_test, y_pred)
    print('Recall: %f' % recall)
    
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, y_pred)
    print('F1 score: %f' % f1)

    # kappa
    kappa = cohen_kappa_score(y_test, y_pred)
    print('Cohens kappa: %f' % kappa)
    
    # confusion matrix
    matrix = confusion_matrix(y_test, y_pred)

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

# Drop one column feature importance
def drop_one_column_feature_importance(X_train, y_train, final_model):
    imp = importances(final_model, X_train, y_train, n_samples=-1)
    fig = px.bar(imp, x='Importance', y=imp.index, orientation='h', color='Importance',
     color_discrete_map={
        "Negative": "indianred",
        "Positive": "seagreen"})
    st.plotly_chart(fig)

def count_plot_top_features(data, feature, target):
    column = ['feature_1', 'feature_2','feature_3']
    for i in column:
        plot = px.histogram(data, x=i, barmode='group', color = i, title = (f'Output Label {feature},{i}')).update_xaxes(categoryorder="total descending")
        st.plotly_chart(plot)

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

def show_density_plots(top, column, values, unique_column):
    subset = top[top[column] == unique_column]
    fig = px.histogram(subset, x = values,
                marginal="box",
                hover_data=top.columns, color = column)
    fig.update_layout(title = f'Density Plot for {unique_column}', xaxis_title = unique_column)
    st.plotly_chart(fig)

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