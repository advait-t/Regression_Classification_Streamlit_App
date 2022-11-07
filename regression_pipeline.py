# from pycaret.regression import *

try:
    from pycaret.regression import compare_models
    from pycaret.regression import create_model
    from pycaret.regression import pull
    from pycaret.regression import create_model
    from pycaret.regression import tune_model
    from pycaret.regression import finalize_model
    from pycaret.regression import get_config
    from pycaret.regression import save_model
    from pycaret.regression import load_model
    from pycaret.regression import download_model
    from pycaret.regression import download_configs
    from pycaret.regression import save_config
    from pycaret.regression import setup
except:
    pass

# from typing import final
import all_function
import pandas as pd
import seaborn as sns
from main import *
from explainability_pipeline import *
# from all_function import *
from global_functions import *

def columns_for_model_building_regression(data):
    columns = list(data.columns)

    target = st.selectbox('Select your target variable', columns)
    if target is not None:
        container1 = st.container()
        all1 = st.checkbox("Select all")
        if all1:
            final_columns1 = container1.multiselect("Select columns for model building:", columns, columns)
            final_columns1.append(target)
        else:
            final_columns1 =  container1.multiselect("Select columns for model building:", columns)
            final_columns1.append(target)

        final_columns1 = list(set(final_columns1))
        st.subheader('Dataset Preview')
        st.dataframe(data[final_columns1].style.applymap(highlight_cols, subset=pd.IndexSlice[:, [target]]))

    parameter_to_be_optimized = st.selectbox('Select a parameter you want to optimize', ['MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE'])
    
    return data, final_columns1, parameter_to_be_optimized, target

def regression_model_function(data, final_columns1, parameter_to_be_optimized, target):
    final_model1, data_setup1, best_model, configs_path, best_model_results, model_name = setting_up_automl_regression(data, final_columns1, parameter_to_be_optimized, target)
    X_train, y_train, X_test, y_test = save_feature_importance_regression(final_model1)
    retrain_data = data_set_regression(X_train, y_train, X_test, y_test)
    target_column = y_train.name
    train_set_columns = list(retrain_data.columns)
    SE, predictions = save_shap_file(final_model1, X_test) # called from explainability_pipeline.py
    # saving_top_features(SE, predictions, target_column)
    return train_set_columns, best_model, data_setup1, retrain_data, target_column, best_model_results, model_name, SE, predictions


def setting_up_automl_regression(data, final_columns, parameter_to_be_optimized, target):
    input_data = data[final_columns]
    models = ['huber', 'br', 'ridge', 'lar']
    ordinal_column = ordinal_columns(input_data, target)
    data_setup = pycaret.regression.setup(data = input_data, session_id=42, target = target, fold = 2, ordinal_features = ordinal_column, use_gpu = False, silent = True, remove_multicollinearity = True)
    
    with st.spinner('Setting up Data'):
        temp_dir = tempfile.TemporaryDirectory()
        save_config((str(temp_dir.name)+'/configs.pkl'))
        configs_path = (str(temp_dir.name)+'/configs.pkl')
        download_configs(save_config(configs_path))

    #choosing the best model
    with st.spinner('Building, comparing and selecting the best model......'):
        best_model = pycaret.regression.compare_models(turbo = True, exclude = models)
        best_model_results = pull()
        model_name = best_model_results['Model'][0]
        st.subheader('Model Building Results')
        st.dataframe(best_model_results.style.applymap(highlight_cols, subset = pd.IndexSlice[parameter_to_be_optimized]))
        st.caption('The metrics are on the test set')
        st.caption(f'Compared all the models and selecting the best model for "{parameter_to_be_optimized}"')
        st.caption(f'The best model chosen for "{parameter_to_be_optimized}" is "{model_name}"')
        model = pycaret.regression.create_model(best_model)

    #tuning the model for getting the best F1-score
    with st.spinner('Tuning the best model......'):
        tuned_model = pycaret.regression.tune_model(model, optimize = parameter_to_be_optimized, choose_better = True)

    #finalising the model and saving it
    with st.spinner('Finalising Model'):
        final_model = pycaret.regression.finalize_model(tuned_model)
        # download_model(save_model(final_model, (str(temp_dir.name)+'/final_model')))

    # Printing the model metrics
    eval_model = evaluate_model(final_model)
    evaluated_model = pull()
    st.write(f'{parameter_to_be_optimized} value after optimization :', (evaluated_model[parameter_to_be_optimized][0]))
    st.write(f'Parameters for {model_name}:')
    st.write(final_model)
    return final_model, data_setup, best_model, configs_path, best_model_results, model_name
        
          
def save_feature_importance_regression(final_model):
    #saving the feature importances as a png file
    X_train = get_config(variable="X_train")
    y_train = get_config(variable="y_train")
    X_test = get_config(variable="X_test")
    y_test = get_config(variable="y_test")
    drop_one_column_feature_importance(X_train, y_train, final_model)
    return X_train, y_train, X_test, y_test


def data_set_regression(X_train, y_train, X_test, y_test):
    train_data = pd.concat([X_train, y_train], axis = 1)
    test_data = pd.concat([X_test,y_test], axis = 1)
    data = pd.concat([train_data, test_data], axis = 0)
    return data


def retrain_model_function_regression(retrain_data, final_columns2, parameter_to_be_optimized, target_column, best_model):
    final_model2, data_setup2 = retrain_model_regression(retrain_data, final_columns2, parameter_to_be_optimized, target_column, best_model)
    X_train_new, y_train_new, X_test_new, y_test_new = save_feature_importance_regression(final_model2)
    SE, predictions = save_shap_file(final_model2, X_test_new)
    saving_top_features(SE, predictions, target_column)
    return final_model2

def retrain_model_regression(data, final_columns, parameter_to_be_optimized, target, best_model):

    # setting up pycaret
    with st.spinner('Setting up Data'):
        input_data = data[final_columns]
        data_setup = setup(data = input_data, target = target, session_id=42 ,preprocess = False, html = False, silent = True)

    #choosing the best model
    with st.spinner('Building the best model'):
        model = create_model(best_model)

    #tuning the model for getting the best F1-score
    with st.spinner('Tuning the best model'):
        tuned_model = tune_model(model, optimize = parameter_to_be_optimized, choose_better = True)

    #finalising the model and saving it
    final_model = finalize_model(tuned_model)
    temp_dir1 = tempfile.TemporaryDirectory()
    output_file_path = (str(temp_dir1.name)+'/final_model')
    download_model(save_model(final_model, output_file_path))
    return final_model, data_setup