from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Initialize the Flask app
app = Flask(__name__)

# Paths to the saved models
MODEL_DIR = 'D:/RGT/Code/Project/Intrusion-Detection-System/models/'
AVAILABLE_MODELS = {
    'Bagging Ensemble': 'Bagging_Classifier.pkl',
    'BernoulliNB': 'Bernoulli_Naive_Bayes.pkl',
    'CNN': 'cnn_model.h5',
    'Decision Trees': 'Decision_Tree.pkl',
    'LightGBM': 'LightGBM.pkl',
    'Linear SVC': 'Linear_SVC.pkl',
    'Logistic Regression': 'Logistic_Regression.pkl',
    'MLP': 'mlp_model.h5',
    'Random Forest': 'Random_Forest.pkl',
    'Stacking Ensemble': 'Stacking_Classifier.pkl',
    'SVM/GradientBoosting Ensemble': 'svm_gb.pkl',
    'Voting Ensemble': 'Voting_Classifier.pkl',
    'XGBoost': 'XGBoost.pkl'
}

# Default columns if the file has no header
default_columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]

# Function to preprocess the data
def preprocess_data(df):
    try:
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_encoded = pd.get_dummies(df, columns=categorical_columns)
        scaler = StandardScaler()
        df_encoded[numerical_columns] = scaler.fit_transform(df_encoded[numerical_columns])
        
        return df_encoded
    except Exception as e:
        print(f"Error in preprocess_data: {e}")
        return None

# Define routes and views
@app.route('/')
def index():
    return render_template('index.html', models=AVAILABLE_MODELS)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    try:
        if request.method == 'POST':
            if 'file' in request.files:
                file = request.files['file']
                if file:
                    file_path = os.path.join('uploads', file.filename)
                    file.save(file_path)
                    
                    df = pd.read_csv(file_path, delimiter=',', header=None)
                    with open(file_path, 'r') as f:
                        first_line = f.readline()
                        has_header = not first_line.strip().replace(',', '').isnumeric()
                    
                    if not has_header:
                        df = pd.read_csv(file_path, names=default_columns[:len(df.columns)])
                    
                    return render_template('upload.html', data=df.values.tolist(), columns=df.columns.tolist(), file_path=file_path, models=AVAILABLE_MODELS)
            
            elif 'manual_entry' in request.form:
                input_data = {col: request.form[col] for col in default_columns}
                df = pd.DataFrame([input_data])
                processed_data = preprocess_data(df)
                
                if processed_data is not None:
                    selected_model = request.form['model']
                    model_path = os.path.join(MODEL_DIR, AVAILABLE_MODELS[selected_model])
                    model = joblib.load(model_path)
                    predictions = model.predict(processed_data)
                    prediction_labels = ['Normal' if pred == 0 else 'Attack' for pred in predictions]
                    
                    return render_template('result.html', data=prediction_labels, original_data=df.to_html())
                else:
                    flash('Failed to process manual entry data.', 'error')
            
            elif 'selected_rows' in request.form:
                selected_indices = request.form.getlist('selected_rows')
                file_path = request.form['file_path']
                selected_model = request.form['model']
                df = pd.read_csv(file_path)
                
                processed_data = preprocess_data(df)
                if processed_data is not None:
                    selected_data = processed_data.iloc[list(map(int, selected_indices))]
                    
                    model_path = os.path.join(MODEL_DIR, AVAILABLE_MODELS[selected_model])
                    model = joblib.load(model_path)
                    predictions = model.predict(selected_data)
                    prediction_labels = ['Normal' if pred == 0 else 'Attack' for pred in predictions]
                    
                    return render_template('result.html', data=prediction_labels, original_data=df.iloc[list(map(int, selected_indices))].to_html())
                else:
                    flash('Failed to process selected rows.', 'error')

    except Exception as e:
        flash(f'Error processing request: {e}', 'error')
    
    return redirect(url_for('index'))


@app.route('/manual_entry', methods=['GET'])
def manual_entry():
    return render_template('manual_entry.html', columns=default_columns, models=AVAILABLE_MODELS)

@app.route('/predict_selected', methods=['POST'])
def predict_selected():
    try:
        if 'selected_rows' in request.form:
            selected_indices = request.form.getlist('selected_rows')
            file_path = request.form['file_path']
            selected_model = request.form['model']
            df = pd.read_csv(file_path)
            
            processed_data = preprocess_data(df)
            if processed_data is not None:
                selected_data = processed_data.iloc[list(map(int, selected_indices))]
                
                model_path = os.path.join(MODEL_DIR, AVAILABLE_MODELS[selected_model])
                model = joblib.load(model_path)
                predictions = model.predict(selected_data)
                prediction_labels = ['Normal' if pred == 0 else 'Attack' for pred in predictions]
                
                return render_template('results.html', data=prediction_labels, original_data=df.iloc[list(map(int, selected_indices))].to_html())
            else:
                return redirect(url_for('index'))
    except Exception as e:
        return redirect(url_for('index'))
    return redirect(url_for('index'))

@app.route('/predict_all', methods=['POST'])
def predict_all():
    try:
        file_path = request.form['file_path']
        selected_model = request.form['model']
        df = pd.read_csv(file_path)
        
        processed_data = preprocess_data(df)
        if processed_data is not None:
            predictions = model.predict(processed_data)
            prediction_labels = ['Normal' if pred == 0 else 'Attack' for pred in predictions]
            
            return render_template('results.html', data=prediction_labels, original_data=df.to_html())
        else:
            return redirect(url_for('index'))
    except Exception as e:
        return redirect(url_for('index'))

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    try:
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        
        app.run(debug=True, use_reloader=False, port=8000)
    except Exception as e:
        print(f"Failed to start the server: {e}")
