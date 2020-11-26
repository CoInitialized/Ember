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

def main():
    """
       Main function for using Ember library in web browser
    """
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
    data = pd.read_csv(file)
    target = 'class'
    X = data.drop(columns = [target])
    y = data[target]

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
    bss = BaesianSklearnSelector(objective,iters)
    fig,results,model = bss.fit(X,y)
    st.pyplot(fig)


main()