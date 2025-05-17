#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Model Testing

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox
import os


# ## Step 1: Load Multiple Test Datasets

# In[2]:


def load_test_datasets():
    test_datasets = []
    
    root = tk.Tk()
    root.withdraw()
    
    while True:
        file_path = filedialog.askopenfilename(
            title="Select test dataset files (CSV or Excel)",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx;*.xls"), ("All files", "*.*")]
        )
        
        if not file_path:  # User cancelled selection
            break
            
        # Load dataset based on file extension
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            print(f"Unsupported file format: {file_path}")
            continue
            
        test_datasets.append(df)
        print(f"Loaded test dataset from {file_path} with shape {df.shape}")
        
        # Ask if user wants to add more datasets
        root = tk.Tk()
        root.withdraw()
        add_more = messagebox.askyesno("Add More", "Do you want to add another test dataset?")
        if not add_more:
            break
    
    if not test_datasets:
        print("No test datasets loaded. Please run the function again.")
        return None
    
    print(f"\nLoaded {len(test_datasets)} test datasets.")
    return test_datasets


# In[3]:


test_datasets = load_test_datasets()


# ## Step 2: Combine Test Datasets

# In[4]:


def combine_test_datasets(test_datasets):
    if not test_datasets:
        print("No test datasets to combine.")
        return None
    
    if len(test_datasets) == 1:
        print("Only one test dataset loaded. No need to combine.")
        return test_datasets[0]
    
    try:
        print("Combining multiple test datasets...")
        combined_test_df = pd.concat(test_datasets, ignore_index=True)
        print(f"Combined test dataset shape: {combined_test_df.shape}")
        return combined_test_df
    except Exception as e:
        print(f"Error combining test datasets: {e}")
        return None


# In[5]:


combined_test_df = combine_test_datasets(test_datasets)
if combined_test_df is not None:
    print("First few rows of combined test data:")
    # display(combined_test_df.head())


# ## Step 3: Load the Trained Model

# In[6]:


def load_trained_model():
    root = tk.Tk()
    root.withdraw()
    
    model_path = filedialog.askopenfilename(
        title="Select the saved model file",
        filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
    )
    
    if not model_path:
        print("No model file selected.")
        return None
    
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


# In[7]:


trained_model = load_trained_model()


# ## Step 4: Preprocess and Test

# In[8]:


def preprocess_and_test(combined_df, model):
    if combined_df is None or model is None:
        print("Missing combined test data or model. Cannot proceed.")
        return None, None, None
    
    print("\nPreprocessing test data...")
    
    # Get target column from user
    target_column = input("Enter the name of the target column: ")
    
    if target_column not in combined_df.columns:
        print(f"Target column '{target_column}' not found in the dataset.")
        print(f"Available columns: {', '.join(combined_df.columns)}")
        return None, None, None
    
    # Split features and target
    X_test = combined_df.drop(columns=[target_column])
    y_test = combined_df[target_column]
    
    # Identify categorical and numerical columns
    categorical_columns = X_test.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Handle categorical features (one-hot encoding)
    if categorical_columns:
        print(f"Performing one-hot encoding on categorical columns: {categorical_columns}")
        X_test = pd.get_dummies(X_test, columns=categorical_columns, drop_first=True)
    
    print(f"Processed test features shape: {X_test.shape}")
    
    # Make predictions
    try:
        print("\nRunning model predictions...")
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()
        
        return X_test, y_test, y_pred
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return X_test, y_test, None


# In[9]:


X_test, y_test, predictions = preprocess_and_test(combined_test_df, trained_model)


# ## Step 5: Save Test Results

# In[10]:


def save_results(X_test, y_test, predictions):
    if predictions is None:
        print("No predictions available to save.")
        return
    
    root = tk.Tk()
    root.withdraw()
    
    save_option = input("Do you want to save the test results? (yes/no): ").lower()
    
    if save_option in ['yes', 'y']:
        save_path = filedialog.asksaveasfilename(
            title="Save test results",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            defaultextension=".csv"
        )
        
        if save_path:
            # Create results dataframe with predictions
            results_df = pd.DataFrame({
                'Actual': y_test.values,
                'Predicted': predictions
            })
            
            # Save to CSV
            results_df.to_csv(save_path, index=False)
            print(f"Test results saved to {save_path}")


# In[11]:


if predictions is not None:
    save_results(X_test, y_test, predictions)


# ## Summary of Testing Process

# In[12]:


def print_test_summary(test_datasets, combined_df, X_test, y_test, predictions):
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    print(f"\nNumber of test datasets loaded: {len(test_datasets) if test_datasets else 0}")
    
    if combined_df is not None:
        print(f"Combined test dataset shape: {combined_df.shape}")
    
    if X_test is not None and y_test is not None:
        print(f"\nTest features shape: {X_test.shape}")
        print(f"Test target shape: {y_test.shape}")
    
    if predictions is not None:
        accuracy = accuracy_score(y_test, predictions)
        print(f"\nTest Accuracy: {accuracy:.4f}")
    
    print("\n" + "="*50)


# In[13]:


print_test_summary(test_datasets, combined_test_df, X_test, y_test, predictions)