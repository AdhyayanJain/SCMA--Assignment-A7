import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from textblob import TextBlob
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from io import BytesIO
import requests





# Function for file upload and preview
def upload_and_preview():
    st.write("## Upload and Preview Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.dataframe(df.head())
        return df
    return None

# Function for data cleaning
def data_cleaning(df):
    st.write("### Data Cleaning")
    if df is not None:
        if st.checkbox("Show Missing Values", key="show_missing"):
            st.write(df.isnull().sum())
        if st.checkbox("Drop Missing Values", key="drop_missing"):
            df.dropna(inplace=True)
            st.write("## Missing values dropped")
        if st.checkbox("Fill Missing Values", key="fill_missing"):
            fill_value = st.selectbox("Select Fill Method", ["Mean", "Median", "Mode"], key="fill_method")
            for col in df.columns:
                if fill_value == "Mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif fill_value == "Median":
                    df[col].fillna(df[col].median(), inplace=True)
                elif fill_value == "Mode":
                    df[col].fillna(df[col].mode()[0], inplace=True)
            st.write("## Missing values filled")
        st.write("### Cleaned Data Preview")
        st.dataframe(df.head())
    return df

# Function for univariate analysis
def univariate_analysis(df):
    st.write("### Univariate Analysis")
    if df is not None:
        column = st.selectbox("Select Column", df.columns, key="univariate_column")
        if column:
            st.write("## Distribution Plot")
            fig = px.histogram(df, x=column)
            st.plotly_chart(fig)

# Function for bivariate analysis
def bivariate_analysis(df):
    st.write("### Bivariate Analysis")
    if df is not None:
        columns = df.columns.tolist()
        x_column = st.selectbox("Select X Column", columns, key="bivariate_x")
        y_column = st.selectbox("Select Y Column", columns, key="bivariate_y")
        if x_column and y_column:
            st.write("## Scatter Plot")
            fig = px.scatter(df, x=x_column, y=y_column)
            st.plotly_chart(fig)

# Function for multivariate analysis
def multivariate_analysis(df):
    st.write("### Multivariate Analysis")
    if df is not None:
        st.write("## Pairplot")
        sns.pairplot(df)
        st.pyplot()

# Function for outlier detection
def outlier_detection(df):
    st.write("### Outlier Detection")
    if df is not None:
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        selected_column = st.selectbox("Select Column", numeric_columns, key="outlier_column")
        if selected_column:
            st.write("## Box Plot")
            fig = px.box(df, y=selected_column)
            st.plotly_chart(fig)

# Function for clustering analysis
def clustering_analysis(df):
    st.write("### Clustering Analysis")
    if df is not None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
        selected_columns = st.multiselect("Select Columns", columns, key="cluster_columns")
        n_clusters = st.slider("Select Number of Clusters", 2, 10, 3, key="n_clusters")
        if selected_columns:
            X = df[selected_columns].values
            kmeans = KMeans(n_clusters=n_clusters)
            df['Cluster'] = kmeans.fit_predict(X)
            st.write("## Cluster Plot")
            fig = px.scatter_matrix(df, dimensions=selected_columns, color="Cluster")
            st.plotly_chart(fig)

# Function for PCA analysis
def pca_analysis(df):
    st.write("### PCA Analysis")
    if df is not None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
        selected_columns = st.multiselect("Select Columns", columns, key="pca_columns")
        n_components = st.slider("Select Number of Components", 2, min(10, len(columns)), 2, key="n_components")
        if selected_columns:
            X = df[selected_columns].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(X_scaled)
            pca_df = pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(n_components)])
            st.write("## PCA Plot")
            fig = px.scatter(pca_df, x="PC1", y="PC2")
            st.plotly_chart(fig)

# Function for sentiment analysis
def sentiment_analysis():
    st.write("### Sentiment Analysis")
    user_text = st.text_area("Enter Text for Sentiment Analysis")
    if user_text:
        blob = TextBlob(user_text)
        st.write(f"## Sentiment Polarity: {blob.sentiment.polarity}")
        st.write(f"## Sentiment Subjectivity: {blob.sentiment.subjectivity}")

# Function for customer segmentation
def customer_segmentation(df):
    st.write("### Customer Segmentation")
    if df is not None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
        selected_columns = st.multiselect("Select Columns", columns, key="cust_seg_columns")
        n_segments = st.slider("Select Number of Segments", 2, 10, 3, key="cust_seg_n")
        if selected_columns:
            X = df[selected_columns].values
            kmeans = KMeans(n_segments)
            df["Segment"] = kmeans.fit_predict(X)
            st.write("## Segment Plot")
            fig = px.scatter_matrix(df, dimensions=selected_columns, color="Segment")
            st.plotly_chart(fig)

# Function for real-time stock analysis
def real_time_stock_analysis():
    st.write("### Real-time Stock Analysis")
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, GOOG)", key="stock_symbol")
    if stock_symbol:
        st.write(f"## Stock Price for {stock_symbol}")
        stock_data = yf.download(stock_symbol, period="1y")
        st.line_chart(stock_data['Close'])
        st.write("## Technical Indicators")
        st.write("## Moving Averages")
        stock_data['SMA'] = stock_data['Close'].rolling(window=20).mean()
        stock_data['EMA'] = stock_data['Close'].ewm(span=20, adjust=False).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close'))
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA'], mode='lines', name='SMA'))
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['EMA'], mode='lines', name='EMA'))
        st.plotly_chart(fig)
        


# Function for feature engineering
def feature_engineering(df):
    st.write("### Feature Engineering")
    if df is not None:
        columns = df.columns.tolist()
        st.write("## Create New Features")
        new_feature = st.text_input("Enter New Feature Expression (e.g., col1 * col2)", key="new_feature")
        if new_feature:
            df.eval(new_feature, inplace=True)
            st.write("## New Feature Added")
            st.dataframe(df.head())
        st.write("## Transform Features")
        transform_column = st.selectbox("Select Column to Transform", columns, key="transform_column")
        transform_method = st.selectbox("Select Transformation Method", ["Log", "Square Root"], key="transform_method")
        if transform_column:
            if transform_method == "Log":
                df[transform_column] = np.log(df[transform_column])
            elif transform_method == "Square Root":
                df[transform_column] = np.sqrt(df[transform_column])
            st.write(f"## Transformed {transform_column}")
            st.dataframe(df.head())

# Function for data preprocessing
def data_preprocessing(df):
    st.write("### Data Preprocessing")
    if df is not None:
        if st.checkbox("Handle Missing Values", key="handle_missing"):
            method = st.selectbox("Select Method", ["Drop Rows", "Fill Mean", "Fill Median", "Fill Mode"], key="missing_method")
            if method == "Drop Rows":
                df.dropna(inplace=True)
            elif method == "Fill Mean":
                df.fillna(df.mean(), inplace=True)
            elif method == "Fill Median":
                df.fillna(df.median(), inplace=True)
            elif method == "Fill Mode":
                df.fillna(df.mode().iloc[0], inplace=True)
        if st.checkbox("Scale Features", key="scale_features"):
            scaler = StandardScaler()
            df[df.columns] = scaler.fit_transform(df[df.columns])
        st.write("## Preprocessed Data Preview")
        st.dataframe(df.head())
    return df

def heart_disease_prediction():
    st.write("### Heart Disease Prediction")
    # Option to download sample dataset
    st.write("#### Download Sample Dataset")
    sample_data_url = "https://raw.githubusercontent.com/AdhyayanJain/SCMA--Assignment-A7/main/heart.csv"
    if st.button("Download Sample Dataset"):
        sample_data = download_sample_data(sample_data_url)
        st.download_button(
            label="Download CSV",
            data=sample_data,
            file_name="sample_insurance_data.csv",
            mime="text/csv"
        )
    uploaded_file = st.file_uploader("Upload Heart Disease Dataset (CSV)", type="csv", key="heart_disease_file")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("## Data Preview")
        st.dataframe(df.head())

        # Preprocess the data
        st.write("## Data Preprocessing")
        # Handle categorical variables if any
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_columns:
            df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
        
        st.write("### Preprocessed Data Preview")
        st.dataframe(df.head())
        
        if st.button("Train Model", key="train_heart_disease"):
            X = df.drop("target", axis=1)
            y = df["target"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"## Model Accuracy: {accuracy}")
            
            # Save the model and scaler for prediction use
            st.session_state.heart_disease_model = model
            st.session_state.heart_disease_scaler = scaler
            st.session_state.heart_disease_columns = X.columns

        if 'heart_disease_model' in st.session_state:
            st.write("### Enter Data for Prediction")
            
            user_input = {}
            for col in st.session_state.heart_disease_columns:
                if col == "age":
                    user_input[col] = st.number_input(f'Enter {col} (in years)', value=0, help="Enter your age in years.")
                elif col == "sex":
                    user_input[col] = st.selectbox(f'Enter {col}', options=[0, 1], help="Enter 1 for male and 0 for female.")
                elif col == "cp":
                    user_input[col] = st.selectbox(f'Enter {col}', options=[0, 1, 2, 3], help="Chest pain type (0-3).")
                elif col == "trestbps":
                    user_input[col] = st.number_input(f'Enter {col} (in mm Hg)', value=0, help="Resting blood pressure (in mm Hg).")
                elif col == "chol":
                    user_input[col] = st.number_input(f'Enter {col} (in mg/dl)', value=0, help="Serum cholesterol (in mg/dl).")
                elif col == "fbs":
                    user_input[col] = st.selectbox(f'Enter {col}', options=[0, 1], help="Fasting blood sugar > 120 mg/dl (1 = true; 0 = false).")
                elif col == "restecg":
                    user_input[col] = st.selectbox(f'Enter {col}', options=[0, 1, 2], help="Resting electrocardiographic results (0-2).")
                elif col == "thalach":
                    user_input[col] = st.number_input(f'Enter {col}', value=0, help="Maximum heart rate achieved.")
                elif col == "exang":
                    user_input[col] = st.selectbox(f'Enter {col}', options=[0, 1], help="Exercise induced angina (1 = yes; 0 = no).")
                elif col == "oldpeak":
                    user_input[col] = st.number_input(f'Enter {col}', value=0.0, help="ST depression induced by exercise relative to rest.")
                elif col == "slope":
                    user_input[col] = st.selectbox(f'Enter {col}', options=[0, 1, 2], help="The slope of the peak exercise ST segment (0-2).")
                elif col == "ca":
                    user_input[col] = st.selectbox(f'Enter {col}', options=[0, 1, 2, 3, 4], help="Number of major vessels (0-4) colored by flourosopy.")
                elif col == "thal":
                    user_input[col] = st.selectbox(f'Enter {col}', options=[0, 1, 2, 3], help="Thalassemia (0 = normal; 1 = fixed defect; 2 = reversable defect).")
                else:
                    user_input[col] = st.number_input(f'Enter {col}', value=0)

            user_input_df = pd.DataFrame(user_input, index=[0])
            user_input_df = st.session_state.heart_disease_scaler.transform(user_input_df)
            
            if st.button("Predict Heart Disease"):
                prediction = st.session_state.heart_disease_model.predict(user_input_df)
                st.write(f"### Predicted Heart Disease: {'Yes' if prediction[0] == 1 else 'No'}")


def breast_cancer_prediction():
    st.write("### Breast Cancer Prediction")

    # Option to download sample dataset
    st.write("#### Download Sample Dataset")
    sample_data_url = "https://raw.githubusercontent.com/AdhyayanJain/SCMA--Assignment-A7/main/cancer.csv"
    if st.button("Download Sample Dataset"):
        sample_data = download_sample_data(sample_data_url)
        st.download_button(
            label="Download CSV",
            data=sample_data,
            file_name="sample_insurance_data.csv",
            mime="text/csv"
        )
    uploaded_file = st.file_uploader("Upload Breast Cancer Dataset (CSV)", type="csv", key="breast_cancer_file")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("## Data Preview")
        st.dataframe(df.head())

        # Drop the 'id' and 'diagnosis' columns for training
        X = df.drop(["id", "diagnosis"], axis=1)
        y = df["diagnosis"].map({"M": 1, "B": 0})

        # Preprocess the data
        st.write("## Data Preprocessing")
        
        # Handle missing values
        imputer = SimpleImputer(strategy="mean")
        X = imputer.fit_transform(X)

        st.write("### Preprocessed Data Preview")
        st.dataframe(X[:5])

        if st.button("Train Model", key="train_breast_cancer"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"## Model Accuracy: {accuracy}")
            
            # Save the model and scaler for prediction use
            st.session_state.breast_cancer_model = model
            st.session_state.breast_cancer_scaler = scaler
            st.session_state.breast_cancer_columns = df.drop(["id", "diagnosis"], axis=1).columns

        if 'breast_cancer_model' in st.session_state:
            st.write("### Enter Data for Prediction")
            
            user_input = {}
            for col in st.session_state.breast_cancer_columns:
                if col == "radius_mean":
                    user_input[col] = st.number_input(f'Enter {col}', value=14.0, help="Mean radius of the cells.")
                elif col == "texture_mean":
                    user_input[col] = st.number_input(f'Enter {col}', value=20.0, help="Mean texture (standard deviation of gray-scale values).")
                elif col == "perimeter_mean":
                    user_input[col] = st.number_input(f'Enter {col}', value=90.0, help="Mean perimeter of the cells.")
                elif col == "area_mean":
                    user_input[col] = st.number_input(f'Enter {col}', value=600.0, help="Mean area of the cells.")
                elif col == "smoothness_mean":
                    user_input[col] = st.number_input(f'Enter {col}', value=0.1, help="Mean smoothness (local variation in radius lengths).")
                elif col == "compactness_mean":
                    user_input[col] = st.number_input(f'Enter {col}', value=0.2, help="Mean compactness of the cells.")
                elif col == "concavity_mean":
                    user_input[col] = st.number_input(f'Enter {col}', value=0.3, help="Mean concavity of the cells.")
                elif col == "concave points_mean":
                    user_input[col] = st.number_input(f'Enter {col}', value=0.1, help="Mean number of concave points on the cell nuclei.")
                elif col == "symmetry_mean":
                    user_input[col] = st.number_input(f'Enter {col}', value=0.2, help="Mean symmetry of the cells.")
                elif col == "fractal_dimension_mean":
                    user_input[col] = st.number_input(f'Enter {col}', value=0.05, help="Mean fractal dimension of the cells.")
                elif col == "radius_se":
                    user_input[col] = st.number_input(f'Enter {col}', value=0.5, help="Standard error of the cell radius.")
                elif col == "texture_se":
                    user_input[col] = st.number_input(f'Enter {col}', value=1.0, help="Standard error of the cell texture.")
                elif col == "perimeter_se":
                    user_input[col] = st.number_input(f'Enter {col}', value=3.0, help="Standard error of the cell perimeter.")
                elif col == "area_se":
                    user_input[col] = st.number_input(f'Enter {col}', value=40.0, help="Standard error of the cell area.")
                elif col == "smoothness_se":
                    user_input[col] = st.number_input(f'Enter {col}', value=0.01, help="Standard error of the cell smoothness.")
                elif col == "compactness_se":
                    user_input[col] = st.number_input(f'Enter {col}', value=0.03, help="Standard error of the cell compactness.")
                elif col == "concavity_se":
                    user_input[col] = st.number_input(f'Enter {col}', value=0.04, help="Standard error of the cell concavity.")
                elif col == "concave points_se":
                    user_input[col] = st.number_input(f'Enter {col}', value=0.02, help="Standard error of the number of concave points on the cell nuclei.")
                elif col == "symmetry_se":
                    user_input[col] = st.number_input(f'Enter {col}', value=0.02, help="Standard error of the cell symmetry.")
                elif col == "fractal_dimension_se":
                    user_input[col] = st.number_input(f'Enter {col}', value=0.004, help="Standard error of the cell fractal dimension.")
                elif col == "radius_worst":
                    user_input[col] = st.number_input(f'Enter {col}', value=17.0, help="Worst (largest) value of cell radius.")
                elif col == "texture_worst":
                    user_input[col] = st.number_input(f'Enter {col}', value=25.0, help="Worst (largest) value of cell texture.")
                elif col == "perimeter_worst":
                    user_input[col] = st.number_input(f'Enter {col}', value=110.0, help="Worst (largest) value of cell perimeter.")
                elif col == "area_worst":
                    user_input[col] = st.number_input(f'Enter {col}', value=900.0, help="Worst (largest) value of cell area.")
                elif col == "smoothness_worst":
                    user_input[col] = st.number_input(f'Enter {col}', value=0.15, help="Worst (largest) value of cell smoothness.")
                elif col == "compactness_worst":
                    user_input[col] = st.number_input(f'Enter {col}', value=0.3, help="Worst (largest) value of cell compactness.")
                elif col == "concavity_worst":
                    user_input[col] = st.number_input(f'Enter {col}', value=0.4, help="Worst (largest) value of cell concavity.")
                elif col == "concave points_worst":
                    user_input[col] = st.number_input(f'Enter {col}', value=0.2, help="Worst (largest) value of cell concave points.")
                elif col == "symmetry_worst":
                    user_input[col] = st.number_input(f'Enter {col}', value=0.3, help="Worst (largest) value of cell symmetry.")
                elif col == "fractal_dimension_worst":
                    user_input[col] = st.number_input(f'Enter {col}', value=0.08, help="Worst (largest) value of cell fractal dimension.")
            
            if st.button("Predict"):
                input_df = pd.DataFrame([user_input])
                input_df = st.session_state.breast_cancer_scaler.transform(input_df)
                prediction = st.session_state.breast_cancer_model.predict(input_df)[0]
                prediction_proba = st.session_state.breast_cancer_model.predict_proba(input_df)[0]
                result = "Malignant" if prediction == 1 else "Benign"
                st.write(f"### Prediction: {result}")
                st.write(f"### Prediction Probability: {prediction_proba}")

def insurance_price_prediction():
    st.write("### Insurance Price Prediction")
    # Option to download sample dataset
    st.write("#### Download Sample Dataset")
    sample_data_url = "https://raw.githubusercontent.com/AdhyayanJain/SCMA--Assignment-A7/main/insurance.csv"
    if st.button("Download Sample Dataset"):
        sample_data = download_sample_data(sample_data_url)
        st.download_button(
            label="Download CSV",
            data=sample_data,
            file_name="sample_insurance_data.csv",
            mime="text/csv"
        )
    uploaded_file = st.file_uploader("Upload Insurance Price Dataset (CSV)", type="csv", key="insurance_price_file")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("## Data Preview")
        st.dataframe(df.head())

        # Preprocess the data
        st.write("## Data Preprocessing")
        # Handle categorical variables
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_columns:
            df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
        
        st.write("### Preprocessed Data Preview")
        st.dataframe(df.head())
        
        if st.button("Train Model", key="train_insurance_price"):
            X = df.drop("charges", axis=1)
            y = df["charges"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            st.write(f"## Model RMSE: {rmse}")
            
            # Save the model and scaler for prediction use
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.columns = X.columns

        if 'model' in st.session_state:
            st.write("### Enter Data for Prediction")
            
            user_input = {}
            for col in st.session_state.columns:
                if 'age' in col:
                    user_input[col] = st.number_input(f'Enter {col}', min_value=0, max_value=120, value=30)
                elif 'bmi' in col:
                    user_input[col] = st.number_input(f'Enter {col}', min_value=0.0, max_value=50.0, value=25.0)
                elif 'children' in col:
                    user_input[col] = st.number_input(f'Enter {col}', min_value=0, max_value=10, value=0)
                elif 'sex_male' in col:
                    user_input[col] = st.selectbox('Sex', options=['female', 'male'])
                    user_input[col] = 1 if user_input[col] == 'male' else 0
                else:
                    user_input[col] = st.number_input(f'Enter {col}', value=0)
            
            user_input_df = pd.DataFrame(user_input, index=[0])
            user_input_df = st.session_state.scaler.transform(user_input_df)
            
            if st.button("Predict Insurance Price"):
                prediction = st.session_state.model.predict(user_input_df)
                st.write(f"### Predicted Insurance Price: {prediction[0]}")

# Main App
def main():
    st.title("Interactive Data Analysis and Machine Learning App")

    st.sidebar.title("Navigation")
    menu = [
        "Upload Data", "EDA", "Feature Engineering", "Sentiment Analysis", 
        "Customer Segmentation", "Real-time Stock Analysis", "Heart Disease Prediction", 
        "Breast Cancer Prediction", "Insurance Price Prediction"
    ]
    choice = st.sidebar.selectbox("Select an Option", menu)

    st.sidebar.markdown("---")
    st.sidebar.write("Developed by [Your Name](https://github.com/YourGitHub)")

    if choice == "Upload Data":
        st.header("Upload Data")
        df = upload_and_preview()

    elif choice == "EDA":
        st.header("Exploratory Data Analysis (EDA)")
        df = upload_and_preview()
        if df is not None:
            eda_options(df)

    elif choice == "Feature Engineering":
        st.header("Feature Engineering")
        df = upload_and_preview()
        if df is not None:
            feature_engineering(df)

    elif choice == "Sentiment Analysis":
        st.header("Sentiment Analysis")
        sentiment_analysis()

    elif choice == "Customer Segmentation":
        st.header("Customer Segmentation")
        df = upload_and_preview()
        if df is not None:
            customer_segmentation(df)

    elif choice == "Real-time Stock Analysis":
        st.header("Real-time Stock Analysis")
        real_time_stock_analysis()

    elif choice == "Heart Disease Prediction":
        st.header("Heart Disease Prediction")
        heart_disease_prediction()

    elif choice == "Breast Cancer Prediction":
        st.header("Breast Cancer Prediction")
        breast_cancer_prediction()

    elif choice == "Insurance Price Prediction":
        st.header("Insurance Price Prediction")
        insurance_price_prediction()

def eda_options(df):
    st.sidebar.subheader("EDA Options")
    eda_option = st.sidebar.selectbox("Choose EDA", ["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis", "Outlier Detection"])
    if eda_option == "Univariate Analysis":
        univariate_analysis(df)
    elif eda_option == "Bivariate Analysis":
        bivariate_analysis(df)
    elif eda_option == "Multivariate Analysis":
        multivariate_analysis(df)
    elif eda_option == "Outlier Detection":
        outlier_detection(df)

if __name__ == '__main__':
    main()
