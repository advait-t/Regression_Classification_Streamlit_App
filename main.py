from pickle import NONE

from matplotlib.pyplot import pink

# from streamlit.state.session_state import SessionState
from classification_modelling_pipeline import *
from regression_pipeline import *
from explainability_pipeline import *
import streamlit as st
# from all_function import *
from global_functions import *

# Function for EDA
def eda():
    st.subheader('Exploratory Data Analysis')
    file = st.file_uploader('Upload Dataset', type = ['csv', 'txt'])
    if file is not None:
        df = pd.read_csv(file)
        all_columns = df.columns.to_list()
        st.dataframe(df.head())

        if st.checkbox("Show Shape"): 
            st.write(df.shape)

        if st.checkbox("Show Columns"):
            st.write(all_columns)

        if st.checkbox("Summary"):
            st.write(df.describe())

        if st.checkbox("Show Selected Columns"):
            selected_columns = st.multiselect("Select Columns",all_columns)
            new_df = df[selected_columns]
            st.dataframe(new_df)

        if st.checkbox("Show Value Counts"):
            selected_columns1 = st.multiselect("Select Column",all_columns)
            #target = st.selectbox('Target Variable', all_columns)
            new_df1 = df[selected_columns1]
            for i in selected_columns1:
                st.write(new_df1[i].value_counts())

        if st.checkbox("Correlation Plot(Matplotlib)"):
            plt.matshow(df.corr())
            st.pyplot()

        if st.checkbox("Correlation Plot(Seaborn)"):
            fig = st.write(sns.heatmap(df.corr(),annot=True))
            st.pyplot(fig)

        if st.checkbox("Pie Plot"):
            all_columns = df.columns.to_list()
            column_to_plot = st.selectbox("Select 1 Column",all_columns)
            pie_plot = df[column_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
            st.write(pie_plot)
            st.pyplot() 




# Function to make plots
def plots():
    st.subheader("Data Visualization")
    data = st.file_uploader("Upload Dataset", type=["csv", "txt"])

    if data is not None:
        df = pd.read_csv(data)
        st.dataframe(df.head())


        if st.checkbox("Show Value Counts"):
            st.write(df.iloc[:,-1].value_counts().plot(kind='bar'))
            st.pyplot()
    
        # Customizable Plot

        all_columns_names = df.columns.tolist()
        type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
        selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)

        if st.button("Generate Plot"):
            st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

            # Plot By Streamlit
            if type_of_plot == 'area':
                cust_data = df[selected_columns_names]
                st.area_chart(cust_data)

            elif type_of_plot == 'bar':
                cust_data = df[selected_columns_names]
                st.bar_chart(cust_data)

            elif type_of_plot == 'line':
                cust_data = df[selected_columns_names]
                st.line_chart(cust_data)

            # Custom Plot 
            elif type_of_plot:
                cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
                st.write(cust_plot)
                st.pyplot() 


def main():
    st.set_page_config(layout="wide")
    st.title('Automated EDA, Model Building and Explainability')
    st.subheader('Automation of Model Building and Explainability of Models')
    side_bar = ['EDA','Visualisation','Classification Model Building', 'Regression Model Building']
    choice = st.sidebar.selectbox('Select your task',side_bar)
    if choice == 'EDA':
        eda()
    elif choice == 'Visualisation':
        plots()
    elif choice == 'Classification Model Building':
        file = select_data()
        if file is not None:
            st.subheader('Uploaded Dataset')
            st.dataframe(file)
            data, final_columns1, parameter_to_be_optimized, target = columns_for_model_building_classification(file)
            button_state = st.checkbox('Build Model')
            list_of_variables = ['best_model', 'target_column','shapash','predictions','best_model_results','parameter_to_be_optimised', 'model_name', 'train_set_columns', 'retrain_data']
            count = 0

            if button_state:
                if any(x in st.session_state for x in list_of_variables):
                    st.write('Model is already built')
                    parameter_to_be_optimized = st.session_state.parameter_to_be_optimized
                    best_model = st.session_state.best_model
                    model_name = st.session_state.model_name
                    best_model_results = st.session_state.best_model_results
                    train_set_columns = st.session_state.train_set_columns
                    retrain_data = st.session_state.retrain_data
                    target_column = st.session_state.target_column
                    shapash = st.session_state.shapash
                    predictions = st.session_state.predictions
                    X_train = st.session_state.X_train
                    y_train = st.session_state.y_train
                    # data_setup = st.session_state.data_setup1
                else:
                    train_set_columns, best_model, data_setup1, retrain_data, target_column, best_model_results, model_name, shapash, predictions, X_train, y_train = classifiction_model_function(data, final_columns1, parameter_to_be_optimized, target)
                    count = count + 1
                
                if count != 1:
                    st.subheader('Model Building Results')
                    st.dataframe(best_model_results)
                    st.caption('Metrics on test set')
                    st.caption(f'Model Selected for "{parameter_to_be_optimized}" is {model_name}')
                    st.write('* Parameters for Best Model:')
                    st.write(best_model)
                    temp_dir = tempfile.TemporaryDirectory()
                    download_model(save_model(best_model, (str(temp_dir.name)+'/final_model')))
                
                drop_one_column_feature_importance(X_train, y_train, best_model)
                saving_top_features(shapash, predictions, target_column)
                # st.write(data_setup1)
                
                st.header('Graphs for the built model')
                graphs = ['auc','threshold','pr','confusion_matrix','error','class_report','feature','feature_all']
                selected_graph = st.selectbox('Graphs', graphs)
                plot_model(best_model, plot = selected_graph, display_format='streamlit')
                
                st.session_state.best_model = best_model
                st.session_state.best_model_results = best_model_results
                st.session_state.parameter_to_be_optimized = parameter_to_be_optimized
                st.session_state.model_name = model_name
                st.session_state.train_set_columns = train_set_columns
                st.session_state.retrain_data = retrain_data
                st.session_state.target_column = target_column
                st.session_state.shapash = shapash
                st.session_state.predictions = predictions
                st.session_state.X_train = X_train
                st.session_state.y_train = y_train
                # st.session_state.data_setup = data_setup1

                rebuild = st.selectbox('Do you want to remove any features and rebuild the model?', ['No', 'Yes'])
                if rebuild == 'Yes':
                    container2 = st.container()
                    all2 = st.checkbox("Select all columns")
                    if all2:
                        final_columns2 = container2.multiselect("Select columns for final model building:", train_set_columns, train_set_columns)
                        
                        if st.checkbox('Rebuild Model'):
                            final_model2 = retrain_model_function_classification(retrain_data, final_columns2, parameter_to_be_optimized, target_column, best_model)
                    else:
                        final_columns2 =  container2.multiselect("Select columns for final model building:", train_set_columns)
                        
                        if st.checkbox('Rebuild Model'):
                            final_model2 = retrain_model_function_classification(retrain_data, final_columns2, parameter_to_be_optimized, target_column, best_model)
                if rebuild == 'No':
                    st.write('Thank You, Model Building Done.')
    else:
        file = select_data()
        if file is not None:
            count = 0
            st.dataframe(file)
            st.caption('Uploaded Dataset')
            data_regression, final_columns1_regression, parameter_to_be_optimized_regression, target_regression = columns_for_model_building_regression(file)
            button_state_regression = st.checkbox('Build Model')
            list_of_variables_regression = ['best_model', 'target_column','shapash','predictions','best_model_results','parameter_to_be_optimised', 'model_name', 'train_set_columns', 'retrain_data']
            
            if button_state_regression:
                if any(x in st.session_state for x in list_of_variables_regression):
                    st.write('Model is already built')
                    parameter_to_be_optimized_regression = st.session_state.parameter_to_be_optimized_regression
                    best_model_regression = st.session_state.best_model_regression
                    model_name_regression = st.session_state.model_name_regression
                    best_model_results_regression = st.session_state.best_model_results_regression
                    train_set_columns_regression = st.session_state.train_set_columns_regression
                    retrain_data_regression = st.session_state.retrain_data_regression
                    target_column_regression = st.session_state.target_column_regression
                    shapash_regression = st.session_state.shapash_regression
                    predictions_regression = st.session_state.predictions_regression
                else:
                    train_set_columns_regression, best_model_regression, data_setup1_regression, retrain_data_regression, target_column_regression, best_model_results_regression, model_name_regression, shapash_regression, predictions_regression = regression_model_function(data_regression, final_columns1_regression, parameter_to_be_optimized_regression, target_regression)
                    count = count + 1

                if count != 1:
                    st.subheader('Model Building Results')
                    st.dataframe(best_model_results_regression)
                    st.caption('Metrics on test set')
                    st.caption(f'Model Selected for "{parameter_to_be_optimized_regression}" is {model_name_regression}')
                    st.write('* Parameters for Best Model:')
                    st.write(best_model_regression)
                    temp_dir = tempfile.TemporaryDirectory()
                    download_model(save_model(best_model_regression, (str(temp_dir.name)+'/final_model')))

                saving_top_features(shapash_regression, predictions_regression, target_column_regression)
                # st.dataframe(best_model_results_regression)
                # st.caption(f'Model Selected for "{parameter_to_be_optimized_regression}" is {model_name_regression}')
                
                # graphs = ['residuals','error', 'cooks', 'rfe', 'learning', 'vc', 'manifold', 'feature', 'parameter']
                # selected_graph = st.selectbox('Graphs', graphs)
                # plot_model(best_model_regression, plot = selected_graph, display_format='streamlit')

                st.session_state.best_model_regression = best_model_regression
                st.session_state.best_model_results_regression = best_model_results_regression
                st.session_state.parameter_to_be_optimized_regression = parameter_to_be_optimized_regression
                st.session_state.model_name_regression = model_name_regression
                st.session_state.train_set_columns_regression = train_set_columns_regression
                st.session_state.retrain_data_regression = retrain_data_regression
                st.session_state.target_column_regression = target_column_regression
                st.session_state.shapash_regression = shapash_regression
                st.session_state.predictions_regression = predictions_regression

                rebuild = st.selectbox('Do you want to remove any features and rebuild the model?', ['No', 'Yes'])
                if rebuild == 'Yes':
                    container2 = st.container()
                    all2 = st.checkbox("Select all columns")
                    if all2:
                        final_columns2 = container2.multiselect("Select columns for final model building:", train_set_columns_regression, train_set_columns_regression)
                        
                        if st.checkbox('Rebuild Model'):
                            final_model2_regression = retrain_model_function_regression(retrain_data_regression, final_columns2, parameter_to_be_optimized_regression, target_column_regression, best_model_regression)
                            graphs = ['residuals','error', 'cooks', 'rfe', 'learning', 'vc', 'manifold', 'feature', 'parameter']
                            selected_graph1 = st.selectbox('Graphs', graphs)
                            plot_model(final_model2_regression, plot = selected_graph1, display_format='streamlit')
                    else:
                        final_columns2_regression =  container2.multiselect("Select columns for final model building:", train_set_columns_regression)
                        
                        if st.checkbox('Rebuild Model'):
                            final_model2_regression = retrain_model_function_regression(retrain_data_regression, final_columns2_regression, parameter_to_be_optimized_regression, target_column_regression, best_model_regression)
                            graphs = ['residuals','error', 'cooks', 'rfe', 'learning', 'vc', 'manifold', 'feature', 'parameter']
                            selected_graph2 = st.selectbox('Graphs', graphs)
                            plot_model(final_model2_regression, plot = selected_graph2, display_format='streamlit')
                if rebuild == 'No':
                    st.write('Thank You, Model Building Done.')

if __name__ == '__main__':
	main()