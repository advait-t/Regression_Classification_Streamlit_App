from classification_modelling_pipeline import *
from shapash.explainer.smart_explainer import SmartExplainer
from all_function import *
from global_functions import *

# saving the shapash pkl file
def save_shap_file(final_model, X_test):
    predictions = predict_model(final_model)
    indexes = X_test.index
    predictions = predictions.set_index(indexes)
    y_pred = predictions['Label']

    SE = SmartExplainer()
    SE.compile(
    y_pred = y_pred,
    x = X_test,
    model=final_model
    )
    temp_dir1 = tempfile.TemporaryDirectory()
    output_file_path = (str(temp_dir1.name)+'/SE_pycaret.pkl')
    download_model(SE.save(output_file_path))

    return SE, predictions

def saving_top_features(SE, predictions, target):
    
    #getting the top contributing features
    X_data = SE.x_pred
    categorical = X_data.select_dtypes(include=['object','int','float64']).columns.tolist()

    indexes = X_data.index
    predictions = predictions.set_index(indexes)

    y_pred = predictions['Label']
    actual = predictions[target]

    summary = SE.to_pandas(proba = False)

    contributions = list(range(1, len(X_data.columns)))

    #making cells NA if categorical features are 0
    for index in range(len(summary)):
        for i in contributions:
            if summary[f'feature_{i}'].iloc[index] in categorical:
                if summary[f'value_{i}'].iloc[index] == 0:
                    summary[f'contribution_{i}'].iloc[index]= None
                    summary[f'value_{i}'].iloc[index] = None
                    summary[f'feature_{i}'].iloc[index] = None
                    
    summary = pd.concat([actual, summary], axis = 1)

    #for each row remove NaNs and create new Series - rows in final df 
    top_3_features = summary.apply(lambda x: pd.Series(x.dropna().values), axis=1)

    #if possible different number of columns like original df is necessary reindex
    top_3_features = top_3_features.reindex(columns=range(len(summary.columns)))

    #assign original columns names
    top_3_features.columns = summary.columns

    #final columns to save
    columns = ['Label',target,'feature_1','feature_2','feature_3','value_1','value_2','value_3','contribution_1','contribution_2','contribution_3']

    #saving to csv
    top_3_features = top_3_features[top_3_features.columns.intersection(columns)]
    top_3_features.to_csv('top_n_features.csv', index = True)
    st.markdown(get_table_download_link(top_3_features), unsafe_allow_html=True)

    #keeping only the relevant columns
    top = pd.read_csv('top_n_features.csv')
    correct_predictions_1 = top[(top[target]==1) & (top['Label']==1)]
    correct_predictions_0 = top[(top[target]==0) & (top['Label']==0)]
    
    # Display header for explainabity section
    st.subheader('Black Box Model Explainability')

    explainability_choice = st.selectbox('Choose Model Explainability Charts', ['Feature Values', 'Top Contributing Features'])

    if explainability_choice == 'Top Contributing Features':
        count_plot_top_features(correct_predictions_1, 1, target)
        count_plot_top_features(correct_predictions_0, 0, target)

    else:
        column = ['feature_1', 'feature_2', 'feature_3']
        values = ['value_1','value_2','value_3']
        feature = st.selectbox('Choose the contributing feature', ['Top Feature','2nd Feature', '3rd Feature'])
        if feature == 'Top Feature':
            column = 'feature_1'
            values = 'value_1'
            unique = top[column].unique()
            unique_column = st.selectbox('Choose column for density charts',unique)
            show_density_plots(top, column, values, unique_column)
        elif feature == '2nd Feature':
            column = 'feature_2'
            values = 'value_2'
            unique = top[column].unique()
            unique_column = st.selectbox('Choose column for density charts',unique)
            show_density_plots(top, column, values, unique_column)
        else:
            column = 'feature_3'
            values = 'value_3'
            unique = top[column].unique()
            unique_column = st.selectbox('Choose column for density charts',unique)
            show_density_plots(top, column, values, unique_column)
