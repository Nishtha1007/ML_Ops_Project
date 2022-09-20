import streamlit as st
import pandas as pd
import numpy as np
import sys
import subprocess
def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install',package])

import_or_install("PIL")
from PIL import Image
import_or_install("requests")
import requests
im = Image.open('uic.png')
import_or_install("seaborn")
import seaborn as sns
import_or_install("sklearn")
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, precision_recall_fscore_support, roc_curve, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import  DecisionTreeClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
import_or_install("imblearn")
from imblearn.over_sampling import SMOTE



import matplotlib.pyplot as plt



st.set_page_config(
    page_title="Diabetes Predictability",
    page_icon=im,
    layout="wide",
    initial_sidebar_state="expanded"
)

header=st.container()
dataset=st.container()
data_manipulation=st.container()
modeling=st.container()
model_fitting=st.container()

st.set_option('deprecation.showPyplotGlobalUse', False)
    

st.markdown(
   """ <style> 
   .font2 {
font-family:"serif";
    font-size: 160%;
    color:#000000;    
    line-height: 100%;
    background-color: #FBCEB1;
    padding: 0.4em;
    letter-spacing: -0.05em;
    word-break: normal;
    border-radius: 10px;
    } 
      .font11 {
font-family:"serif";
    font-size: 250%;
    color:#FFFFFF;    
    line-height: 100%;
    background-color: #06038D;
    padding: 0.4em;
    letter-spacing: -0.05em;
    word-break: normal;
    border-radius: 10px;
    } 
    .font1 {
font-family:"serif";
    font-size: 160%;
    color:#00FFFF;    
    line-height: 100%;
    background-color: #36454F;
    padding: 0.4em;
    letter-spacing: -0.05em;
    word-break: normal;
    border-radius: 5px;
}
        .font3 {
font-family:"serif";
    font-size: 100%;
    color:#000000;    
    font-weight: 300;
    line-height: 100%;
    background-color: #FFFFFF;
    padding: 0.4em;
    letter-spacing: -0.05em;
    word-break: normal;
    border-radius: 5px;
    /font-style: italic;
}
</style> 

""",
        unsafe_allow_html=True,
    )


def plot_metrics(metrics_list,y_test, y_pred):
    if "Confusion Matrix" in metrics_list:
        cm = confusion_matrix(y_test, y_pred)
        cmn = cm/len(y_test)
        fig, ax = plt.subplots(figsize=(15,10))
        sns.heatmap(cmn, annot=True, fmt='.2f',ax=ax)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show(block=False)
        st.pyplot(fig)

        
st.sidebar.title("Interactive") 

models=st.sidebar.multiselect("How would you like the data to be modeled?",("Naive Bayes", "Logistic Regression","KNN"))

def data():
    medical_df = pd.read_csv("diabetic_data.csv")
    medical_df = medical_df.replace('?', np.nan)
    return medical_df

with header:
    col1, col2 = st.columns(2)
    with col1:
        image = Image.open('diabetes.jpg')
        st.image(image, width=380)
    with col2:
        st.write("Diabetes is one of the most prevalent chronic diseases worldwide. Detecting diabetes in the early stage will help a patient to get better treatment and lifestyle. In this paper, we built several machine learning models to predict hospital readmission that is within 30 days among diabetic patients. The main idea is to provide a comprehensive data solution to the re-admission problem at the healthcare institutions to embark on a significant improvement in in-patient diabetic care.")
    st.markdown("""# ML Ops Project""")
    st.markdown('<p class="font11">Welcome to Project</p>',unsafe_allow_html=True)
    
    

with dataset:
    medical_df=data()
    st.markdown('<p class="font2">Diabetes Data</p>', unsafe_allow_html=True)
    st.markdown('<p class="font3">Few Lines of Data', unsafe_allow_html=True)
    st.write(medical_df.head(5))  
    st.markdown('<p class="font3">Percentage of Missing Data across columns', unsafe_allow_html=True)
    percent_missing = medical_df.isnull().sum() * 100 / len(medical_df)
    missing_value_df = pd.DataFrame({'percent_missing': percent_missing})
    st.write(missing_value_df)
    st.markdown('<p class="font3">Readmission Categories', unsafe_allow_html=True)
    st.write(medical_df.readmitted.value_counts())
    st.markdown('<p class="font3">Visualizing the target label distribution using a bar chart', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(15,8))
    sns.countplot(x='readmitted', data=medical_df,order=medical_df['readmitted'].value_counts(ascending=False).index,ax=ax)
    plt.title('Count of patients by their readmission rate (target label)')
    st.pyplot(fig)

    explode = (0, 0.1, 0)
    pie_df=medical_df.readmitted.value_counts().rename_axis('unique_values').reset_index(name='counts')
    
    st.markdown('<p class="font3">Calculating the proportion of patients  fall under each category(label)', unsafe_allow_html=True)
    fig1, ax1 = plt.subplots(figsize=(15,8))
    ax1.pie(pie_df.counts, explode=explode, labels=pie_df.unique_values, autopct='%1.2f%%',
        shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig1)
    st.markdown('<p class="font2">Checking the distribution of different variables', unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(15,8))
    sns.countplot(x = "admission_type_id", data = medical_df)
    plt.title("Distribution of Admission IDs")
    st.pyplot(fig)
    # Refined Adimission Type
    fig, ax = plt.subplots(figsize=(15,8))
    mapped = {1.0:"Emergency",
              2.0:"Emergency",
              3.0:"Elective",
              4.0:"New Born",
              5.0:np.nan,
              6.0:np.nan,
              7.0:"Trauma Center",
              8.0:np.nan}

    medical_df.admission_type_id = medical_df.admission_type_id.replace(mapped)
    sns.countplot(x = "admission_type_id", data = medical_df,ax=ax)
    plt.title("Distribution of Refined Admission IDs")
    st.pyplot(fig)
    
    fig, ax = plt.subplots(figsize=(15,8))
    sns.countplot(x=medical_df.race, data = medical_df,ax=ax)
    plt.xticks(rotation=90)
    plt.title("Number of Race values")
    st.pyplot(fig)
    
    fig, ax = plt.subplots(figsize=(15,8))
    sns.countplot(x="age", data = medical_df,ax=ax)
    plt.title("Age")
    plt.xticks(rotation = 90)
    st.pyplot(fig)
    
    fig, ax = plt.subplots(figsize=(15,8))
    sns.countplot(x="time_in_hospital", data = medical_df,
              order = medical_df.time_in_hospital.value_counts().index,ax=ax)
    plt.title("Time in Hospital")
    st.pyplot(fig)
    
    st.markdown('<p class="font2">Data Preprocessing- Binary Classification: 1 if diabetes within 30 days else 0', unsafe_allow_html=True)
    medical_df['readmitted'] = medical_df['readmitted'].map({'NO': 0, '<30': 1, '>30': 0})
    
    fig, ax = plt.subplots(figsize=(15,8))
    sns.countplot(x = "readmitted", data = medical_df,ax=ax)
    plt.title("Distribution of Target Values")
    st.pyplot(fig)
    
    fig, ax = plt.subplots(figsize=(15,5))
    sns.heatmap(medical_df.corr(), vmin=-1, vmax=1, annot=True, cmap='Blues',ax=ax)
    plt.title("Correlation Heatmap")
    st.pyplot(fig)
    
@st.cache(suppress_st_warning=True)  
def data_preprocess(medical_df):
    medical_df['admission_type_id'] = medical_df.admission_type_id.astype('object', copy=False)
    medical_df['admission_source_id'] = medical_df.admission_source_id.astype('object', copy=False)
    medical_df['discharge_disposition_id'] = medical_df.discharge_disposition_id.astype('object', copy=False)
    mapped_discharge = {1:"Discharged to Home",
                    6:"Discharged to Home",
                    8:"Discharged to Home",
                    13:"Discharged to Home",
                    19:"Discharged to Home",
                    18:np.nan,25:np.nan,26:np.nan,
                    2:"Other",3:"Other",4:"Other",
                    5:"Other",7:"Other",9:"Other",
                    10:"Other",11:"Other",12:"Other",
                    14:"Other",15:"Other",16:"Other",
                    17:"Other",20:"Other",21:"Other",
                    22:"Other",23:"Other",24:"Other",
                    27:"Other",28:"Other",29:"Other",30:"Other"}

    medical_df["discharge_disposition_id"] = medical_df["discharge_disposition_id"].replace(mapped_discharge)
    
    mapped_adm = {1:"Referral",2:"Referral",3:"Referral",
              4:"Other",5:"Other",6:"Other",10:"Other",22:"Other",25:"Other",
              9:"Other",8:"Other",14:"Other",13:"Other",11:"Other",
              15:np.nan,17:np.nan,20:np.nan,21:np.nan,
              7:"Emergency"}
    medical_df.admission_source_id = medical_df.admission_source_id.replace(mapped_adm)
    medical_df=medical_df.drop(['encounter_id'],axis=1)
    medical_df = medical_df.drop(['patient_nbr'],axis=1)
    medical_df = medical_df.drop(columns=['examide', 'citoglipton'])
    medical_df = medical_df.drop(columns=['diag_1', 'diag_2', 'diag_3'])
    # Based on weak correlation results
    medical_df = medical_df.drop(columns=['nateglinide', 'chlorpropamide', 'acetohexamide', 'glyburide',
                                                               'tolbutamide', 'miglitol', 'troglitazone', 'tolazamide', 
                                                               'glyburide-metformin', 'glipizide-metformin', 
                                                                'glimepiride-pioglitazone', 'metformin-rosiglitazone', 
                                                               'metformin-pioglitazone'])
    # Dealing with missing values
    medical_df = medical_df.drop(columns=['weight', 'payer_code', 'medical_specialty'])
    # The features: 'race' has a lot of missing values in the rows so getting rid of the missing values
    medical_df = medical_df.dropna(how='any', subset=['race'])
    
    Col = medical_df.columns[medical_df.dtypes.apply(lambda c: np.issubdtype(c, np.number))]
    
    scaler = MinMaxScaler()
    medical_df[Col.difference(['readmitted'])] = scaler.fit_transform(medical_df[Col.difference(['readmitted'])])
    for col in medical_df[['race', 'gender', 'age', 'discharge_disposition_id','admission_type_id', 'admission_source_id',
                                     'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide',
                                    'glimepiride', 'glipizide' , 'pioglitazone', 'rosiglitazone', 'acarbose', 'insulin',
                                    'change','diabetesMed']]:
      medical_df = pd.get_dummies(medical_df, columns=[col], dtype= 'int64', prefix=col, drop_first = True)
    return medical_df
with data_manipulation:
    medical_df=data_preprocess(medical_df)
    
@st.cache(suppress_st_warning=True) 
def split(medical_df,test_size):
    X = medical_df.drop(['readmitted'],axis=1).values   # independant features
    Y = medical_df['readmitted'].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size, random_state=42, shuffle=True)
    return X_train, X_test, Y_train, Y_test 

@st.cache(suppress_st_warning=True)
def smote(X_train, Y_train):
    # One thing to note is new samples are only added to the traning set. This is to ensure that the model generalizes well on the unseen data
    smt = SMOTE()
    X_train,Y_train=smt.fit_resample(X_train,Y_train) 
    return X_train, Y_train

@st.cache(suppress_st_warning=True)
def pca(X_train,X_test,variation):
    pca=PCA(variation)
    pca.fit(X_train)
    X_train=pca.transform(X_train)
    X_test=pca.transform(X_test)
    return X_train, X_test

with modeling:
    st.markdown('<p class="font2">Splitting into Training and Test Data</p>', unsafe_allow_html=True)
    test_size=st.slider("What would be the test_size?", min_value=0.1,max_value=1.0,value=0.3,step=0.05)
    X_train, X_test, Y_train, Y_test = split(medical_df,test_size)
    X_train, Y_train= smote(X_train, Y_train)
    st.markdown('<p class="font2">PCA is used to reduce the number of variables/features(dimensions) in a dataset</p>', unsafe_allow_html=True)
    variation=st.slider("Amount of variation to be captured?", min_value=0.1,max_value=1.0,value=0.9,step=0.05)
    X_train, X_test= pca(X_train,X_test,variation)
    st.markdown('<p class="font3">Training Data shape', unsafe_allow_html=True)
    st.write(X_train.shape)
    
@st.cache(suppress_st_warning=True)     
def naive_bayes(X_train, Y_train,X_test,var_smoothing):
    gnb = GaussianNB(var_smoothing=var_smoothing)
    gnb.fit(X_train, Y_train)
    nb_pred_test = gnb.predict(X_test)
    nb_pred_train = gnb.predict(X_train)
    train_acc = accuracy_score(Y_train, nb_pred_train)
    test_acc = accuracy_score(Y_test, nb_pred_test)
    test_err = 1-test_acc
    train_err = 1 - train_acc
    return gnb,train_acc,train_err,test_acc,test_err,nb_pred_test,nb_pred_train
@st.cache(suppress_st_warning=True)     
def logic_reg(X_train, Y_train,X_test,solver,C):
    lr_model = LogisticRegression(solver=solver,C=C,random_state=0)
    # fit the model
    lr_model.fit(X_train, Y_train)
    lr_pred_test = lr_model.predict(X_test)
    lr_pred_train = lr_model.predict(X_train)
    lr_train_acc = accuracy_score(Y_train, lr_pred_train)
    lr_test_acc = accuracy_score(Y_test, lr_pred_test)
    lr_test_err = 1 - lr_test_acc
    lr_train_err = 1 - lr_train_acc
    return lr_model,lr_pred_test,lr_pred_train,lr_train_acc,lr_test_acc,lr_test_err,lr_train_err
@st.cache(suppress_st_warning=True)     
def knn_reg(X_train, Y_train,X_test,n_neighbors):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, Y_train)
    knn_pred_train = knn.predict(X_train)
    knn_pred_test = knn.predict(X_test)
    knn_train_acc = accuracy_score(Y_train, knn_pred_train)
    knn_test_acc = accuracy_score(Y_test, knn_pred_test)
    knn_test_err = 1 - knn_test_acc
    knn_train_err = 1 - knn_train_acc
    return knn,knn_pred_train,knn_pred_test,knn_train_acc,knn_test_acc,knn_test_err,knn_train_err
with model_fitting:
    if "Naive Bayes" in models:
        gnb,train_acc,train_err,test_acc,test_err,nb_pred_test,nb_pred_train=naive_bayes(X_train, Y_train,X_test,var_smoothing=0.001)
        st.markdown('<p class="font2">Gaussian Naive Bayes Performance</p>', unsafe_allow_html=True)
        st.write('Train Acurracy:', round(train_acc,2))
        st.write('Training error is:', round(train_err,2))
        st.write('Confusion matrix for training data')
        plot_metrics(["Confusion Matrix"],Y_train,nb_pred_train)
        st.write('Test Acurracy:', round(test_acc,2))
        st.write("Test error is:",round(test_err,2))
        st.write('Confusion_matrix for the test data')
        plot_metrics(["Confusion Matrix"],Y_test,nb_pred_test)
    if "Logistic Regression" in models:
        st.markdown('<p class="font2">Applying Logistic Regression to Model</p>', unsafe_allow_html=True)
        solver=st.select_slider('Select solver type',options=['liblinear','saga','newton-cg','lbfgs'],value=('liblinear'))
        C=st.slider("What would be the value of C(Inverse of regularization strength)?", min_value=0.1,max_value=2.0,value=0.2,step=0.1)   
        lr_model,lr_pred_test,lr_pred_train,lr_train_acc,lr_test_acc,lr_test_err,lr_train_err=logic_reg(X_train, Y_train,X_test,solver,C)
        st.markdown('<p class="font2">Logistic Regression Performance</p>', unsafe_allow_html=True)
        st.write('Train Acurracy:', round(lr_train_acc,2))
        st.write('Training error is:', round(lr_train_err,2))
        st.write('Confusion matrix for training data')
        plot_metrics(["Confusion Matrix"],Y_train,lr_pred_train)
        st.write('Test Acurracy:', round(lr_test_acc,2))
        st.write("Test error is:",round(lr_test_err,2))
        st.write('Confusion_matrix for the test data')
        plot_metrics(["Confusion Matrix"],Y_test,lr_pred_test)   
    if "KNN" in models:
        st.markdown('<p class="font2">Applying K Nearest Neighbours to Model</p>', unsafe_allow_html=True)
        n_neighbors=st.slider("What would be the number of neighbours?", min_value=1,max_value=8,value=3,step=1)   
        knn,knn_pred_train,knn_pred_test,knn_train_acc,knn_test_acc,knn_test_err,knn_train_err=knn_reg(X_train, Y_train,X_test,n_neighbors)
        st.markdown('<p class="font2">KNN Performance</p>', unsafe_allow_html=True)
        st.write('Train Acurracy:', round(knn_train_acc,2))
        st.write('Training error is:', round(knn_train_err,2))
        st.write('Confusion matrix for training data')
        plot_metrics(["Confusion Matrix"],Y_train,knn_pred_train)
        st.write('Test Acurracy:', round(knn_test_acc,2))
        st.write("Test error is:",round(knn_test_err,2))
        st.write('Confusion_matrix for the test data')
        plot_metrics(["Confusion Matrix"],Y_test,knn_pred_test)       
    
    
    
     
