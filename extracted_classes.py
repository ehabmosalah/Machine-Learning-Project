# ======================== IMPORTS ========================
from collections import Counter
from joblib import dump
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVC
from sklearn.svm import SVR
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import pycountry
import re
import seaborn as sns

# ======================== CLASSES ========================
class CorrelationFilter:

    def __init__(self, threshold=0.85):
        self.threshold = threshold
        self.to_drop = set()

    def fit(self, data):
        numerical_data = data.select_dtypes(include=[np.number])
        corr_matrix = numerical_data.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > self.threshold:
                    colname = corr_matrix.columns[i]
                    self.to_drop.add(colname)

    def transform(self, data):
        return data.drop(columns=list(self.to_drop & set(data.columns)), errors='ignore')

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
class CategoryReducer:

    def __init__(self, category_columns, top_n=15):
        self.category_columns = category_columns
        self.top_n = top_n
        self.top_categories = None

    def fit(self, data):
        self.top_categories = data[self.category_columns].sum().sort_values(ascending=False).head(self.top_n).index.tolist()

    def transform(self, data):
        df_top = data[self.top_categories].copy()
        other_columns = list(set(self.category_columns) - set(self.top_categories))
        df_top['Other'] = data[other_columns].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
        data = data.drop(columns=self.category_columns)
        data = pd.concat([data, df_top], axis=1)
        return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
class AgeTransformer:

    def __init__(self, current_year=2025):
        self.current_year = current_year
        self.age_mode = None

    def fit(self, data):
        data = data.copy()
        data['Year Founded'] = pd.to_numeric(data['Year Founded'], errors='coerce')
        data['age'] = self.current_year - data['Year Founded']
        mode_series = data['age'].mode()
        if not mode_series.empty:
            self.age_mode = mode_series[0]
        else:
            self.age_mode = 5

    def transform(self, data):
        """Applies the transformation: computes age and fills missing values."""
        data = data.copy()
        data['Year Founded'] = pd.to_numeric(data['Year Founded'], errors='coerce')
        data['age'] = self.current_year - data['Year Founded']
        data.drop(columns=['Year Founded'], inplace=True)
        data['age'].fillna(self.age_mode, inplace=True)
        return data
class IPOAgeTransformer:

    def __init__(self, current_year=2025):
        self.current_year = current_year

    def fit(self, data):
        return self

    def transform(self, data):
        df = data.copy()
        df['IPO'] = pd.to_numeric(df['IPO'], errors='coerce')
        df['age IPO'] = self.current_year - df['IPO']
        df['age IPO'].replace(np.nan, -1, inplace=True)
        if 'IPO' in df.columns:
            df.drop(columns=['IPO'], inplace=True)
        return df

    def fit_transform(self, data):
        return self.fit(data).transform(data)
class EmployeeDataCleaner:

    def __init__(self):
        self.employee_mode = None
        self.mean_without_zeros = None

    def fit(self, data):
        self.employee_mode = data['Number of Employees (year of last update)'].mode()[0] if not data['Number of Employees (year of last update)'].mode().empty else 0
        self.mean_without_zeros = data.loc[data['Number of Employees'] > 0, 'Number of Employees'].mean()
        return self

    def transform(self, data):
        df = data.copy()
        if 'Number of Employees (year of last update)' in df.columns:
            df['Number of Employees (year of last update)'].fillna(self.employee_mode, inplace=True)
        if 'Number of Employees' in df.columns:
            df['Number of Employees'] = df['Number of Employees'].apply(lambda x: 0 if pd.isna(x) or x < 0 else x)
            df['Number of Employees'] = df['Number of Employees'].replace(0, self.mean_without_zeros)
        return df

    def fit_transform(self, data):
        return self.fit(data).transform(data)
class TaglineCategoryGuesser:

    def __init__(self):
        self.category_keywords = {'Artificial Intelligence': ['ai', 'machine learning', 'deep learning', 'neural network'], 'Mobile': ['mobile', 'android', 'ios', 'app store', 'smartphone'], 'E-Commerce': ['ecommerce', 'e-commerce', 'shopping', 'online store'], 'FinTech': ['finance', 'banking', 'payments', 'fintech', 'crypto', 'blockchain'], 'Healthcare': ['health', 'medical', 'hospital', 'doctor', 'pharma'], 'Social Media': ['social network', 'community', 'messaging', 'chat'], 'Gaming': ['game', 'gaming', 'video game', 'esports'], 'Cloud': ['cloud', 'saas', 'paas', 'infrastructure'], 'EdTech': ['education', 'learning', 'students', 'teaching', 'school'], 'Data Analytics': ['analytics', 'data science', 'big data', 'insights']}

    def guess_category_from_tagline(self, tagline):
        tagline = str(tagline).lower()
        matched = [cat for (cat, keywords) in self.category_keywords.items() if any((keyword in tagline for keyword in keywords))]
        if len(matched) == 0:
            matched = ['Software', 'Advertising']
        elif len(matched) == 1:
            matched.append('Software')
        return ', '.join(matched)

    def fit(self, data):
        return self

    def transform(self, data):
        df = data.copy()
        df['Tagline'] = df['Tagline'].fillna('')
        df['Market Categories'] = df['Market Categories'].fillna('Unknown')
        df['Market Categories'] = df.apply(lambda row: self.guess_category_from_tagline(row['Tagline']) if str(row['Market Categories']).strip().lower() in ['unknown', 'nan', 'none', ''] else row['Market Categories'], axis=1)
        return df

    def fit_transform(self, data):
        return self.fit(data).transform(data)
class MarketCategoryGeneralizer:

    def __init__(self):
        self.category_mapping = {'Software': 'Technology & Software', 'Advertising': 'Advertising & Marketing', 'E-Commerce': 'E-Commerce & Online Services', 'Mobile': 'Mobile & Consumer Electronics', 'Games': 'Games & Entertainment', 'Social Media': 'Social Networking & Communication', 'Cloud': 'Technology & Software', 'Finance': 'Finance & Payments', 'Healthcare': 'Healthcare & Wellness', 'Semiconductors': 'Technology Hardware', 'Data Analytics': 'Analytics & Data Science', 'Search': 'Advertising & Marketing', 'Video': 'Games & Entertainment', 'Networking': 'Telecom & Networks', 'Messaging': 'Social Networking & Communication', 'Education': 'Education & Learning', 'News': 'Media & News', 'Photo Sharing': 'Digital Media & Content', 'Mobile Payments': 'Finance & Payments', 'Robotics': 'Games & Entertainment', 'Music': 'Games & Entertainment', 'Photo Editing': 'Digital Media & Content', 'Online Rental': 'E-Commerce & Online Services', 'Location Based Services': 'Telecom & Networks', 'Enterprise Software': 'Technology & Software', 'Video Streaming': 'Games & Entertainment', 'PaaS': 'Technology & Software', 'SaaS': 'Technology & Software', 'Health and Wellness': 'Healthcare & Wellness', 'Web Hosting': 'Technology & Software', 'Internet of Things': 'IoT (Internet of Things)', 'Cloud Security': 'Technology & Software', 'Virtual Currency': 'Finance & Payments', 'Search Marketing': 'Advertising & Marketing', 'Mobile Social': 'Social Networking & Communication', 'Retail': 'Retail & Fashion', 'Consulting': 'Others & Miscellaneous', 'Aerospace': 'Others & Miscellaneous', 'Food Delivery': 'Consumer Goods & Services', 'Fashion': 'Retail & Fashion', 'Wine And Spirits': 'Consumer Goods & Services', 'Streaming': 'Games & Entertainment', 'Task Management': 'Others & Miscellaneous', 'Video Chat': 'Social Networking & Communication', 'Personalization': 'Advertising & Marketing', 'Shopping': 'E-Commerce & Online Services', 'Local': 'E-Commerce & Online Services', 'News': 'Media & News', 'Fraud Detection': 'Advertising & Marketing', 'Image Recognition': 'Technology Hardware', 'Virtualization': 'Games & Entertainment', 'Analytics': 'Analytics & Data Science', 'Video on Demand': 'Games & Entertainment', 'Mobile Payments': 'Finance & Payments', 'Marketing Automation': 'Advertising & Marketing', 'Consumer Electronics': 'Mobile & Consumer Electronics', 'Video Games': 'Games & Entertainment', 'Public Relations': 'Advertising & Marketing'}

    def map_categories(self, row):
        categories = str(row).split(',')
        generalized = []
        for cat in categories:
            cat = cat.strip()
            if cat in self.category_mapping:
                generalized.append(self.category_mapping[cat])
            else:
                generalized.append('Others & Miscellaneous')
        return ', '.join(set(generalized))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['Generalized Market Categories'] = df['Market Categories'].fillna('').apply(self.map_categories)
        return df

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
class CountryRegionFiller:

    def __init__(self):
        self.countries = [country.name for country in pycountry.countries]
        self.regions = ['California', 'New York', 'Texas', 'Basel', 'Utah', 'Île-de-France', 'Bavaria', 'Ontario', 'Switzerland', 'United States', 'France', 'Great Britain', 'Israel', 'Sweden', 'Canada', 'Germany', 'Japan', 'India', 'Denmark', 'China', 'Spain', 'Netherlands', 'Finland', 'Australia', 'Ireland', 'United Stats of AMerica', 'United Arab Emirates', 'Quebec']

    def find_place(self, text, place_list):
        for place in place_list:
            if re.search('\\b' + re.escape(place) + '\\b', str(text)):
                return place
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        for (idx, row) in df[df['Country (HQ)'].isnull() | df['State / Region (HQ)'].isnull()].iterrows():
            desc = row['Description']
            if pd.isnull(desc):
                continue
            country = self.find_place(desc, self.countries)
            region = self.find_place(desc, self.regions)
            if pd.isnull(row['Country (HQ)']) and country:
                df.at[idx, 'Country (HQ)'] = country
            if pd.isnull(row['State / Region (HQ)']) and region:
                df.at[idx, 'State / Region (HQ)'] = region
        return df

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
class CategoricalFillerAndEncoder:

    def __init__(self, columns):
        self.columns = columns
        self.modes = {}
        self.label_encoders = {}
        self.label_maps = {}

    def fit(self, X, y=None):
        for col in self.columns:
            mode_val = X[col].mode()[0]
            self.modes[col] = mode_val
            le = LabelEncoder()
            filled = X[col].fillna(mode_val).astype(str)
            le.fit(filled)
            self.label_encoders[col] = le
            self.label_maps[col] = {label: i for (i, label) in enumerate(le.classes_)}
        return self

    def transform(self, X):
        df = X.copy()
        for col in self.columns:
            mode_val = self.modes[col]
            label_map = self.label_maps[col]
            df[col] = df[col].fillna(mode_val).astype(str)
            df[col + '_LabelEncoded'] = df[col].map(lambda x: label_map.get(x, -1))
        return df

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
class CustomEncoder(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.status_cols_ = None
        self.terms_cols_ = None

    def fit(self, df):
        self.status_cols_ = df['Status'].unique()
        self.terms_cols_ = df['Terms'].unique()
        return self

    def transform(self, df):
        df = df.copy()
        df = pd.get_dummies(df, columns=['Status'], drop_first=False)
        df = pd.get_dummies(df, columns=['Terms'], drop_first=False)
        if 'Terms_Cash, Stock' in df.columns:
            cash_stock_mask = df['Terms_Cash, Stock'] == 1
            df.loc[cash_stock_mask, 'Terms_Cash'] = 1
            df.loc[cash_stock_mask, 'Terms_Stock'] = 1
            df = df.drop('Terms_Cash, Stock', axis=1)
        expected_cols = [f'Status_{s}' for s in self.status_cols_] + [f'Terms_{t}' for t in self.terms_cols_ if t != 'Cash, Stock']
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        return df[expected_cols]