import pandas as pd
from ember.preprocessing import Preprocessor 
import numpy as np
from ember.impute import GeneralImputer
from ember.optimize import GridSelector, BayesSelector
from sklearn.pipeline import make_pipeline
from ember.utils import DtypeSelector
from ember.preprocessing import Preprocessor, GeneralEncoder, GeneralScaler
import streamlit as st
from skopt.plots import plot_convergence
from ember.optimize import BaesianSklearnSelector
import pickle
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

    """
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

def main():
    """
       Main function for using Ember library in web browser
    """
    train_button = None
    
    objective = st.selectbox(
                                'Please select objective',
                                ('regression', 'classification')
                            )
    iters = int(st.text_input("Please provide maximal number of itterations",10))
    file = st.file_uploader("Upload file", type=['csv','xlm'])
    show_file = st.empty()
    if not file:
        show_file.info("Please upload a file of type: " + ", ".join(['xlm','csv']))
        return
    else:
        data = pd.read_csv(file)
        st.write(data.head())
        train_button = st.button("Start training")
    
    if train_button:
        target = 'class'
        X = data.drop(columns = [target])
        y = data[target]
        columns = X.columns

        target_preprocessor = Preprocessor()
        target_preprocessor.add_branch('target')


        if y.dtype == np.object:

            target_preprocessor.add_transformer_to_branch('target', GeneralImputer('Simple', 'most_frequent'))
            target_preprocessor.add_transformer_to_branch('target', GeneralEncoder('LE'))
        else:
            if objective == 'classification':
                target_preprocessor.add_transformer_to_branch('target', GeneralImputer('Simple', 'most_frequent'))
            elif objective == 'regression':
                target_preprocessor.add_transformer_to_branch('target', GeneralImputer('Simple', 'mean'))
            else:
                pass
                        
                ## features pipeline ##

        feature_preprocessor = Preprocessor()

        is_number = len(X.select_dtypes(include=np.number).columns.tolist()) > 0
        is_object = len(X.select_dtypes(include=np.object).columns.tolist()) > 0

        if is_object:
            feature_preprocessor.add_branch("categorical")
            feature_preprocessor.add_transformer_to_branch("categorical", DtypeSelector(np.object))
            feature_preprocessor.add_transformer_to_branch("categorical", GeneralImputer('Simple', strategy='most_frequent'))
            feature_preprocessor.add_transformer_to_branch("categorical", GeneralEncoder(kind = 'OHE'))


        if is_number:
            feature_preprocessor.add_branch('numerical')
            feature_preprocessor.add_transformer_to_branch("numerical", DtypeSelector(np.number))
            feature_preprocessor.add_transformer_to_branch("numerical", GeneralImputer('Simple'))
            feature_preprocessor.add_transformer_to_branch("numerical", GeneralScaler('SS'))

                
        feature_preprocessor = feature_preprocessor.merge()
        target_preprocessor = target_preprocessor.merge()

        y = np.array(y).reshape(-1,1)
        y = target_preprocessor.fit_transform(y).ravel()

        X = feature_preprocessor.fit_transform(X)
        #X = np.array(X)
        print("Starting selection")
        sns.set_context("paper", font_scale=1)  
        bss = BaesianSklearnSelector(objective,iters, cv = 3)
        fig,results,model = bss.fit(X,y)
        st.pyplot(fig)

        feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_,columns)), columns=['Value','Feature'])
        print(feature_imp.head(3))
        sns.set_context("paper", font_scale=4)  
        fig2 = plt.figure(figsize=(20, 20))
        fig2.suptitle("Feature importance")
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
        st.pyplot(fig2)

        if st.button('Download model'):
            # nie dziala to
            tmp_download_link = download_link(model, 'model.ember', 'Click here to download your text!')
            st.markdown(tmp_download_link, unsafe_allow_html=True)


main()