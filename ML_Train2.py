import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pycountry
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor
from sklearn.svm import SVC

data =pd.read_csv('data/Acquiring Tech Companies.csv')

data.isna().sum()

data

data=data.drop(['CrunchBase Profile','Image','Homepage','Twitter','API'],axis=1)

data

data.dtypes

data['Number of Employees'] = data['Number of Employees'].replace({',': ''}, regex=True)
data['Number of Employees'] = data['Number of Employees'].fillna(0).astype(int)

for col in data.columns:
    if data[col].dtype == 'object':
        try:
            data[col] = data[col].astype(int)
        except:
            pass  # ignore if it fails, i.e., for non-numeric text

data.dtypes

# Replace "Not yet" with NaN first
data['IPO'] = data['IPO'].replace("Not yet", np.nan)

# Create a new binary feature
data['Is_Public'] = data['IPO'].notna().astype(int)


data.drop_duplicates(inplace=True)


data.count()

from sklearn.preprocessing import MultiLabelBinarizer

# Step 1: Split the comma-separated values into lists
data['Market Categories'] = data['Market Categories'].fillna('')  # Handle NaN
data['Market Categories List'] = data['Market Categories'].apply(lambda x: [i.strip() for i in x.split(',') if i.strip()])

# Step 2: Use MultiLabelBinarizer to create one-hot columns
mlb = MultiLabelBinarizer()
mlb.fit(data['Market Categories'].fillna('').str.split(','))
category_dummies = pd.DataFrame(mlb.fit_transform(data['Market Categories List']),
                                columns=mlb.classes_,
                                index=data.index)

# Step 3: Concatenate the result with the original dataframe (or drop the original column if you want)
data = pd.concat([data, category_dummies], axis=1)

# Optional: Drop the old columns if not needed
data.drop(columns=['Market Categories', 'Market Categories List'], inplace=True)


# Save the MultiLabelBinarizer
import pickle
# Save the MultiLabelBinarizer to a file
with open('mlb_acquiring.pkl', 'wb') as f:
 pickle.dump(mlb, f)

data.columns

numerical_data = data.select_dtypes(include=[np.number])

# Select object (categorical) columns
categorical_data = data.select_dtypes(include=['object'])

# Display the first few rows of each
print("Numerical Data:")
print(numerical_data.head())

print("\nCategorical Data:")
print(categorical_data.head())

'''correlation_matrix = numerical_data.corr()

# Set a threshold for correlation (e.g., 0.85)
threshold = 0.85

# Create a set to store features to drop
to_drop = set()

# Loop through the upper triangle of the correlation matrix
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:  # Check if correlation is above threshold
            colname = correlation_matrix.columns[i]
            to_drop.add(colname)  # Mark column for removal

# Drop the columns from the dataset
data = data.drop(columns=to_drop)'''


class CorrelationFilter:
    def __init__(self, threshold=0.85):
        self.threshold = threshold
        self.to_drop = set()  # Stores columns to drop
        self.fitted = False   # Tracks if fit() was called

    def fit(self, data):
        """Identifies highly correlated columns to drop."""
        numerical_data = data.select_dtypes(include=[np.number])
        corr_matrix = numerical_data.corr()

        self.to_drop = set()  # Reset in case fit() is called again

        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > self.threshold:
                    colname = corr_matrix.columns[i]
                    self.to_drop.add(colname)

        self.fitted = True
        return self

    def transform(self, data):
        """Drops columns identified in fit()."""
        if not self.fitted:
            raise RuntimeError("Call fit() before transform()!")
        
        cols_to_drop = list(self.to_drop & set(data.columns))
        return data.drop(columns=cols_to_drop, errors='ignore')

    def fit_transform(self, data):
        """Combines fit() and transform()."""
        self.fit(data)
        return self.transform(data)

    def get_columns_to_drop(self):
        """Returns the list of columns to be dropped."""
        if not self.fitted:
            raise RuntimeError("Call fit() first!")
        return list(self.to_drop)

# Initialize and fit
corr_filter = CorrelationFilter(threshold=0.85)
data = corr_filter.fit_transform(data)

# Get columns to drop (for reference)
columns_to_drop = corr_filter.get_columns_to_drop()
print("Columns to drop:", columns_to_drop)

# Save the filter (includes `to_drop` list)
with open("correlation_filter.pkl", "wb") as f:
    pickle.dump(corr_filter, f)

null_indexes = data[data['State / Region (HQ)'].isnull()].index
print(null_indexes)


data['State / Region (HQ)'][21]='Finnmark'

data['State / Region (HQ)'][26]='Gyeonggi-do'

data['State / Region (HQ)'][28]='Baden-Württemberg'

null_indexes = data[data['City (HQ)'].isnull()].index
print(null_indexes)


data['City (HQ)'][28]='Walldorf'

data['City (HQ)'][34]='West Berkshire'

data.columns

'''category_columns = ['Advertising Platforms', 'All Markets', 'All Students', 'Big Data',
       'Blogging Platforms', 'Cloud Computing', 'Communications Hardware',
       'Computers', 'Consumer Goods', 'Creative', 'Curated Web', 'E-Commerce',
       'Electronics', 'Email', 'Enterprise Software', 'Hardware',
       'Hardware + Software', 'Information Technology', 'Messaging', 'Mobile',
       'Networking', 'Photography', 'RIM', 'Search', 'Security',
       'Semiconductors', 'Social Bookmarking', 'Social Media',
       'Social Recruiting', 'Software', 'Storage', 'Telecommunications',
       'Video Games', 'Web Hosting']
# Step 2: Find top 15 most frequent (most used) category columns
top_15 = data[category_columns].sum().sort_values(ascending=False).head(20).index.tolist()

# Step 3: Keep top 15 and compute 'Other' column
df_top = data[top_15].copy()

# Create 'Other' column from all the rest
other_columns = list(set(category_columns) - set(top_15))
df_top['Other'] = data[other_columns].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

# Step 4: Drop the original one-hot category columns
data.drop(columns=category_columns, inplace=True)

# Step 5: Add the reduced one-hot set back
data = pd.concat([data, df_top], axis=1)'''

class CategoryReducer:
    def __init__(self, category_columns, top_n=15):
        self.category_columns = category_columns
        self.top_n = top_n
        self.top_categories = None  # Will store the top categories from training

    def fit(self, data):
        # Identify and store the top N categories (only during training)
        self.top_categories = (
            data[self.category_columns]
            .sum()
            .sort_values(ascending=False)
            .head(self.top_n)
            .index.tolist()
        )
        return self  # For sklearn compatibility

    def transform(self, data):
        if self.top_categories is None:
            raise RuntimeError("Call fit() before transform()!")

        # Keep only the top categories (from training)
        df_top = data[self.top_categories].copy()

        # Sum remaining categories into "Other"
        other_columns = list(set(self.category_columns) - set(self.top_categories))
        df_top['Other'] = data[other_columns].sum(axis=1).clip(upper=1)  # Ensures 0 or 1

        # Drop original columns and concatenate reduced set
        data = data.drop(columns=self.category_columns)
        return pd.concat([data, df_top], axis=1)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


category_columns = ['Advertising Platforms', 'All Markets', 'All Students', 'Big Data',
       'Blogging Platforms', 'Cloud Computing', 'Communications Hardware',
       'Computers', 'Consumer Goods', 'Creative', 'Curated Web', 'E-Commerce',
       'Electronics', 'Email', 'Enterprise Software', 'Hardware',
       'Hardware + Software', 'Information Technology', 'Messaging', 'Mobile',
       'Networking', 'Photography', 'RIM', 'Search', 'Security',
       'Semiconductors', 'Social Bookmarking', 'Social Media',
       'Social Recruiting', 'Software', 'Storage', 'Telecommunications',
       'Video Games', 'Web Hosting']

# Initialize and apply reducer
reducer = CategoryReducer(category_columns, top_n=15)
data = reducer.fit_transform(data)

import pickle

with open("category_reducer_acquiring.pkl", "wb") as f:
    pickle.dump(reducer, f)



data

'''data['Year Founded'] = pd.to_numeric(data['Year Founded'], errors='coerce')

# Then, calculate age (assuming current year is 2025)
data['age'] = 2025 - data['Year Founded']
data.drop(columns=['Year Founded'], inplace=True)'''

class AgeTransformer:
    def __init__(self, current_year=2025, min_age=0, max_age=100):
        self.current_year = current_year
        self.age_mode = None
        self.min_age = min_age  # Prevent negative ages
        self.max_age = max_age  # Prevent unrealistic ages

    def fit(self, data):
        """Calculates and stores the mode of ages from training data."""
        if 'Year Founded' not in data.columns:
            raise ValueError("Column 'Year Founded' not found in data.")
        
        data = data.copy()
        data['Year Founded'] = pd.to_numeric(data['Year Founded'], errors='coerce')
        data['age'] = (self.current_year - data['Year Founded']).clip(self.min_age, self.max_age)
        
        mode_series = data['age'].mode()
        self.age_mode = mode_series[0] if not mode_series.empty else 5  # Default fallback

    def transform(self, data):
        """Applies age calculation and fills missing values with the training mode."""
        if 'Year Founded' not in data.columns:
            raise ValueError("Column 'Year Founded' not found in data.")
        
        data = data.copy()
        data['Year Founded'] = pd.to_numeric(data['Year Founded'], errors='coerce')
        data['age'] = (self.current_year - data['Year Founded']).clip(self.min_age, self.max_age)
        data['age'].fillna(self.age_mode, inplace=True)
        data.drop(columns=['Year Founded'], inplace=True)
        return data

transformer = AgeTransformer()
transformer.fit(data)
data = transformer.transform(data)



#save 
with open("Age_column_acquiring.pkl", 'wb') as f:
    pickle.dump(transformer, f)

class IPOAgeTransformer:
    def __init__(self, current_year=2025, unknown_placeholder="Unknown"):
        self.current_year = current_year
        self.unknown_placeholder = unknown_placeholder  # Replace NaN values

    def fit(self, data):
        """Stateless (no training needed). For pipeline compatibility."""
        return self

    def transform(self, data):
        """Computes IPO age and replaces missing values."""
        if 'IPO' not in data.columns:
            raise ValueError("Column 'IPO' not found in data.")

        df = data.copy()
        df['IPO'] = pd.to_numeric(df['IPO'], errors='coerce')
        
        # Compute age (clamp negative values to 0)
        df['age IPO'] = (self.current_year - df['IPO']).clip(lower=0)
        
        # Replace missing ages with placeholder
        df['age IPO'] = df['age IPO'].replace(
            np.nan, self.unknown_placeholder
        )
        
        # Drop original IPO column
        df.drop(columns=['IPO'], inplace=True, errors='ignore')
        
        return df

    def fit_transform(self, data):
        return self.transform(data)  # fit() is stateless

ipo_transformer = IPOAgeTransformer(current_year=2025, unknown_placeholder="Unknown")
data = ipo_transformer.fit_transform(data)


with open("ipo_transformer_acquiring.pkl", "wb") as f:
    pickle.dump(ipo_transformer, f)





"""data['IPO'] = pd.to_numeric(data['IPO'], errors='coerce')

# Then, calculate age (assuming current year is 2025)
data['age IPO'] = 2025 - data['IPO']
data.drop(columns=['IPO'], inplace=True)"""

#data['age IPO'].replace(np.nan, -1, inplace=True)

data.isna().sum()

data.drop(columns=['Address (HQ)'], inplace=True)


'''data['Number of Employees (year of last update)'].fillna(
    data['Number of Employees (year of last update)'].mode()[0], inplace=True
)

data.dtypes'''

'''mean_without_zeros = data.loc[data['Number of Employees'] != 0, 'Number of Employees'].mean()

# Step 2: Replace 0s with that mean
data['Number of Employees'] = data['Number of Employees'].replace(0, mean_without_zeros)'''

class EmployeeDataCleaner:
    def __init__(self):
        self.employee_mode = None
        self.mean_without_zeros = None
        self.fitted = False  # Safety flag

    def fit(self, data):
        """Compute and store statistics from training data."""
        # Validate columns
        required_columns = [
            'Number of Employees (year of last update)',
            'Number of Employees'
        ]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data.")

        # Compute mode for 'Number of Employees (year of last update)'
        mode_series = data['Number of Employees (year of last update)'].mode()
        self.employee_mode = mode_series[0] if not mode_series.empty else 0

        # Compute mean (excluding zeros/negatives) for 'Number of Employees'
        non_zero_employees = data.loc[
            data['Number of Employees'] > 0, 'Number of Employees'
        ]
        self.mean_without_zeros = non_zero_employees.mean()

        # Fallback if all values are zero/NaN
        if pd.isna(self.mean_without_zeros):
            self.mean_without_zeros = data['Number of Employees'].median()  # or a global default

        self.fitted = True
        return self

    def transform(self, data):
        """Apply cleaning using statistics from fit()."""
        if not self.fitted:
            raise RuntimeError("Call fit() before transform()!")

        df = data.copy()

        # Fill missing values with training mode
        if 'Number of Employees (year of last update)' in df.columns:
            df['Number of Employees (year of last update)'].fillna(
                self.employee_mode, inplace=True
            )

        # Handle nulls/negatives and replace zeros with training mean
        if 'Number of Employees' in df.columns:
            df['Number of Employees'] = np.where(
                df['Number of Employees'].isna() | (df['Number of Employees'] < 0),
                0,
                df['Number of Employees']
            )
            df['Number of Employees'] = df['Number of Employees'].replace(
                0, self.mean_without_zeros
            )

        return df

    def fit_transform(self, data):
        return self.fit(data).transform(data)

employee_cleaner = EmployeeDataCleaner()
data = employee_cleaner.fit_transform(data)


with open("employee_cleaner_acquiring.pkl", "wb") as f:
    pickle.dump(employee_cleaner, f)



data.info()

# for col in data.columns:
#     if data[col].dtype == 'object':  # or you can use: if data[col].dtype == 'O'
#         le = LabelEncoder()
#         data[col] = data[col].astype(str)  # ensure all values are strings
#         data[col] = le.fit_transform(data[col])

data

from collections import Counter
import pandas as pd
import pickle

class BoardMembersTransformer:
    def __init__(self, min_count=5):
        self.min_count = min_count  # Keep only members appearing ≥ min_count times
        self.common_members = None  # Stores frequent members from training
        self.member_counts_ = None  # Optional: track raw counts

    def fit(self, data):
        """Identify frequently occurring board members from training data."""
        if 'Board Members' not in data.columns:
            raise ValueError("Column 'Board Members' not found in data.")

        all_members = []
        for cell in data['Board Members'].dropna():
            members = [name.strip() for name in str(cell).split(',')]
            all_members.extend(members)

        # Count occurrences and filter by min_count
        self.member_counts_ = Counter(all_members)
        self.common_members = {
            name for name, count in self.member_counts_.items() 
            if count >= self.min_count
        }
        return self

    def transform(self, data):
        """Convert board members into binary features for common members."""
        if self.common_members is None:
            raise RuntimeError("Call fit() before transform()!")

        df = data.copy()
        
        # Create binary columns for each common member
        for member in self.common_members:
            df[f'Board Member: {member}'] = df['Board Members'].apply(
                lambda x: 1 if pd.notna(x) and member in str(x) else 0
            )

        # Optional: Add a summary feature (total members or binary "has members")
        df['Has Board Members'] = df['Board Members'].notna().astype(int)
        
        # Drop original column
        df.drop(columns=['Board Members'], inplace=True, errors='ignore')
        return df

    def fit_transform(self, data):
        return self.fit(data).transform(data)

transformer = BoardMembersTransformer(min_count=5)
data = transformer.fit_transform(data)

# Save the transformer (includes common_members)
with open("board_members_transformer.pkl", "wb") as f:
    pickle.dump(transformer, f)

data.isna().sum()

from collections import Counter

class FoundersTransformer:
    def __init__(self, min_count=3):
        self.min_count = min_count  # Keep founders appearing ≥ min_count times
        self.common_founders = None  # Stores frequent founders from training
        self.founder_counts_ = None  # Optional: track raw counts

    def fit(self, data):
        """Identify frequently occurring founders from training data."""
        if 'Founders' not in data.columns:
            raise ValueError("Column 'Founders' not found in data.")

        all_founders = []
        for cell in data['Founders'].dropna():
            founders = [name.strip() for name in str(cell).split(',')]
            all_founders.extend(founders)

        # Count occurrences and filter by min_count
        self.founder_counts_ = Counter(all_founders)
        self.common_founders = {
            name for name, count in self.founder_counts_.items() 
            if count >= self.min_count
        }
        return self

    def transform(self, data):
        """Convert founders into binary features for common founders."""
        if self.common_founders is None:
            raise RuntimeError("Call fit() before transform()!")

        df = data.copy()
        
        # Create binary columns for each common founder
        for founder in self.common_founders:
            df[f'Founder: {founder}'] = df['Founders'].apply(
                lambda x: 1 if pd.notna(x) and founder in str(x) else 0
            )

        # Optional: Add a summary feature
        df['Has Founders'] = df['Founders'].notna().astype(int)
        
        # Drop original column
        df.drop(columns=['Founders'], inplace=True, errors='ignore')
        return df

    def fit_transform(self, data):
        return self.fit(data).transform(data)

transformer = FoundersTransformer(min_count=3)
data = transformer.fit_transform(data)

# Save the transformer (includes common_founders)
with open("founders_transformer.pkl", "wb") as f:
    pickle.dump(transformer, f)

data

data['Text_Combined'] = data['Tagline'].fillna('') + ' ' + data['Description'].fillna('')


data['Text_Combined']

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra whitespace
    return text

data['Text_Clean'] = data['Text_Combined'].apply(clean_text)

data['Text_Clean']


tfidf = TfidfVectorizer(max_features=150)
tfidf_matrix = tfidf.fit_transform(data['Text_Clean'])

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
data = pd.concat([data.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

 # Save TF-IDF vectorizer
import pickle
# Save the TF-IDF vectorizer to a file
with open('tfidf_acquiring.pkl', 'wb') as f:
            pickle.dump(tfidf, f)

data.shape

data.drop(columns=['Tagline','Description'], inplace=True)

data.drop(columns=['Text_Clean'], inplace=True)

import re
acquisition_df = pd.read_csv("data/Acquisitions.csv")
acquired_df = pd.read_csv(r"data/Acquired Tech Companies.csv")


#acquired_df.head()
acquired_df.columns
print(acquired_df.isnull().sum())

print(acquired_df.duplicated().sum())
#print("status column unique values",acquisition_df["Status"].unique())
#print("Terms column unique values",acquisition_df["Terms"].unique())

acquired_df.info()
# task1 from the address get the country then remove the address




def extract_year(text):
    if not isinstance(text, str) or not text.strip():
        return None

    # Look for 'founded' followed by anything then a 4-digit year
    match = re.search(r'\bfounded\b.*?(\d{4})', text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    return None

print(acquired_df['Year Founded'].isnull().sum())
# Apply the function to 'Tagline' and 'Description' where 'Year Founded' is null
acquired_df['Year Founded'] = acquired_df.apply(
    lambda row: extract_year(row['Tagline']) if pd.isnull(row['Year Founded']) else row['Year Founded'], axis=1
)

# If 'Year Founded' is still null, try using the Description column
acquired_df['Year Founded'] = acquired_df.apply(
    lambda row: extract_year(row['Description']) if pd.isnull(row['Year Founded']) else row['Year Founded'], axis=1
)
print(acquired_df['Year Founded'].isnull().sum())

'''acquired_df['Year Founded'].fillna(acquired_df['Year Founded'].median(), inplace=True)
print(acquired_df['Year Founded'].isnull().sum())'''


# --- Step 4.5: Fill missing 'Market Categories' from 'Tagline' ---

# Fill NaN values with 'Unknown'
'''acquired_df['Market Categories'] = acquired_df['Market Categories'].fillna('Unknown')

# Define keyword-based categories
category_keywords = {
    "Artificial Intelligence": ["ai", "machine learning", "deep learning", "neural network"],
    "Mobile": ["mobile", "android", "ios", "app store", "smartphone"],
    "E-Commerce": ["ecommerce", "e-commerce", "shopping", "online store"],
    "FinTech": ["finance", "banking", "payments", "fintech", "crypto", "blockchain"],
    "Healthcare": ["health", "medical", "hospital", "doctor", "pharma"],
    "Social Media": ["social network", "community", "messaging", "chat"],
    "Gaming": ["game", "gaming", "video game", "esports"],
    "Cloud": ["cloud", "saas", "paas", "infrastructure"],
    "EdTech": ["education", "learning", "students", "teaching", "school"],
    "Data Analytics": ["analytics", "data science", "big data", "insights"]
}

# Optional: preview unique existing categories
split_categories = acquired_df['Market Categories'].apply(lambda x: [i.strip() for i in x.split(',') if i.strip()])
all_categories = sorted(set(cat for sublist in split_categories for cat in sublist))
print("Sample existing categories:", all_categories[:20])
print("Total unique categories:", len(all_categories))

# Function to guess category from tagline

def guess_category_from_tagline(tagline):
    tagline = str(tagline).lower()
    matched = [cat for cat, keywords in category_keywords.items()
               if any(keyword in tagline for keyword in keywords)]
    return ', '.join(matched) if matched else "Unknown"


def guess_category_from_tagline(tagline):
    tagline = str(tagline).lower()
    matched = [cat for cat, keywords in category_keywords.items()
               if any(keyword in tagline for keyword in keywords)]

    # Ensure at least 2 values are returned
    if len(matched) == 0:
        matched = ["Software", "Advertising"]
    elif len(matched) == 1:
        matched.append("Software")

    return ', '.join(matched)
# Fill only where category is unknown
acquired_df['Market Categories'] = acquired_df.apply(
    lambda row: guess_category_from_tagline(row['Tagline']) if row['Market Categories'] in ["Unknown", "nan", "", None] else row['Market Categories'],
    axis=1
)'''
#--------------------------------------------------------------------
# Define the mapping of specific categories to generalized categories
'''category_mapping = {
    'Software': 'Technology & Software',
    'Advertising': 'Advertising & Marketing',
    'E-Commerce': 'E-Commerce & Online Services',
    'Mobile': 'Mobile & Consumer Electronics',
    'Games': 'Games & Entertainment',
    'Social Media': 'Social Networking & Communication',
    'Cloud': 'Technology & Software',
    'Finance': 'Finance & Payments',
    'Healthcare': 'Healthcare & Wellness',
    'Semiconductors': 'Technology Hardware',
    'Data Analytics': 'Analytics & Data Science',
    'Search': 'Advertising & Marketing',
    'Video': 'Games & Entertainment',
    'Networking': 'Telecom & Networks',
    'Messaging': 'Social Networking & Communication',
    'Education': 'Education & Learning',
    'News': 'Media & News',
    'Photo Sharing': 'Digital Media & Content',
    'Mobile Payments': 'Finance & Payments',
    'Robotics': 'Games & Entertainment',
    'Music': 'Games & Entertainment',
    'Photo Editing': 'Digital Media & Content',
    'Online Rental': 'E-Commerce & Online Services',
    'Location Based Services': 'Telecom & Networks',
    'Enterprise Software': 'Technology & Software',
    'Video Streaming': 'Games & Entertainment',
    'PaaS': 'Technology & Software',
    'SaaS': 'Technology & Software',
    'Health and Wellness': 'Healthcare & Wellness',
    'Web Hosting': 'Technology & Software',
    'Internet of Things': 'IoT (Internet of Things)',
    'Cloud Security': 'Technology & Software',
    'Virtual Currency': 'Finance & Payments',
    'Search Marketing': 'Advertising & Marketing',
    'Mobile Social': 'Social Networking & Communication',
    'Retail': 'Retail & Fashion',
    'Consulting': 'Others & Miscellaneous',
    'Aerospace': 'Others & Miscellaneous',
    'Food Delivery': 'Consumer Goods & Services',
    'Fashion': 'Retail & Fashion',
    'Wine And Spirits': 'Consumer Goods & Services',
    'Streaming': 'Games & Entertainment',
    'Task Management': 'Others & Miscellaneous',
    'Video Chat': 'Social Networking & Communication',
    'Personalization': 'Advertising & Marketing',
    'Shopping': 'E-Commerce & Online Services',
    'Local': 'E-Commerce & Online Services',
    'News': 'Media & News',
    'Fraud Detection': 'Advertising & Marketing',
    'Image Recognition': 'Technology Hardware',
    'Virtualization': 'Games & Entertainment',
    'Analytics': 'Analytics & Data Science',
    'Video on Demand': 'Games & Entertainment',
    'Mobile Payments': 'Finance & Payments',
    'Marketing Automation': 'Advertising & Marketing',
    'Consumer Electronics': 'Mobile & Consumer Electronics',
    'Video Games': 'Games & Entertainment',
    'Public Relations': 'Advertising & Marketing'
}

# Add any other specific categories to the mapping above

# Function to map specific categories to the generalized ones
def map_to_generalized_category(row):
    # Split the current market categories and map each category
    categories = row.split(',')  # Split by comma
    generalized_categories = []

    for category in categories:
        category = category.strip()  # Remove any extra spaces
        if category in category_mapping:
            generalized_categories.append(category_mapping[category])
        else:
            generalized_categories.append('Others & Miscellaneous')  # In case of unknown categories

    return ', '.join(set(generalized_categories))  # Return unique generalized categories as a string

# Apply the mapping function to the 'Market Categories' column
acquired_df['Generalized Market Categories'] = acquired_df['Market Categories'].apply(map_to_generalized_category)

# Print the full output without truncation
print(acquired_df[['Market Categories', 'Generalized Market Categories']].to_string(index=False))

acquired_df.loc[acquired_df['Company']=="Emagic"]
acquired_df.info()

print(acquired_df.isnull().sum())'''
#--------------------------------------------------------------------------------------------------------
# Prepare country list
'''countries = [country.name for country in pycountry.countries]
regions = ['California', 'New York', 'Texas', 'Basel', 'Utah', 'Île-de-France', 'Bavaria', 'Ontario',
           'Switzerland', 'United States', 'France', 'Great Britain',
       'Israel', 'Sweden', 'Canada', 'Germany', 'Japan', 'India',
       'Denmark', 'China', 'Spain', 'Netherlands', 'Finland', 'Australia',
       'Ireland', 'United Stats of AMerica', 'United Arab Emirates'
           'Quebec',]  # extend as needed

# Helper function to find first match
def find_place(text, place_list):
    for place in place_list:
        if re.search(r'\b' + re.escape(place) + r'\b', str(text)):
            return place
    return None

# Loop through rows with nulls
for idx, row in acquired_df[acquired_df['Country (HQ)'].isnull() | acquired_df['State / Region (HQ)'].isnull()].iterrows():
    desc = row['Description']
    if pd.isnull(desc):
        continue
    # Try to extract country and region
    country = find_place(desc, countries)
    region = find_place(desc, regions)
    if pd.isnull(row['Country (HQ)']) and country:
        acquired_df.at[idx, 'Country (HQ)'] = country
    if pd.isnull(row['State / Region (HQ)']) and region:
        acquired_df.at[idx, 'State / Region (HQ)'] = region


columns_to_drop = ['Image', 'CrunchBase Profile', 'Homepage', 'Twitter','Address (HQ)','API','Description','Tagline','Market Categories']
acquired_df.drop(columns=columns_to_drop, inplace=True)
print(acquired_df.columns)
'''
#-----------------------------------------------------------
'''import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Fill nulls in categorical columns using mode
for col in ['City (HQ)', 'State / Region (HQ)', 'Country (HQ)']:
    mode_value = acquired_df[col].mode()[0]
    acquired_df[col].fillna(mode_value, inplace=True)

# Label Encoding
label_encoders = {}
for col in ['City (HQ)', 'State / Region (HQ)', 'Country (HQ)']:
    le = LabelEncoder()
    acquired_df[col + '_LabelEncoded'] = le.fit_transform(acquired_df[col].astype(str))
    label_encoders[col] = le

# One-hot encoding
df_one_hot = pd.get_dummies(acquired_df[['City (HQ)', 'State / Region (HQ)', 'Country (HQ)']],
                            prefix=['City', 'State', 'Country'])

# Drop the original columns
acquired_df.drop(columns=['City (HQ)', 'State / Region (HQ)', 'Country (HQ)'], inplace=True)

# Combine everything
df_final = pd.concat([acquired_df, df_one_hot], axis=1)'''

#--------------------------------------------------

# View the result
#print(df_final.head())

#---------------------------------------------------------------------------------------------------------
# from sklearn.preprocessing import MultiLabelBinarizer
# import pandas as pd

# # Step 1: Convert the 'Generalized Market Categories' column to a list of categories
# acquired_df['MultiCategoryList'] = acquired_df['Generalized Market Categories'].apply(
#     lambda x: [cat.strip() for cat in x.split(',')] if pd.notnull(x) else []
# )

# # Step 2: Initialize the MultiLabelBinarizer
# mlb = MultiLabelBinarizer()

# # Step 3: Transform the list column to one-hot encoded format
# multi_hot_encoded = mlb.fit_transform(acquired_df['MultiCategoryList'])

# # Step 4: Convert to DataFrame with proper column names
# df_multi_hot = pd.DataFrame(multi_hot_encoded, columns=mlb.classes_, index=acquired_df.index)

# # Step 5: Drop the helper column
# acquired_df.drop(columns=['MultiCategoryList'], inplace=True)

# # Step 6: Merge back the encoded columns
# acquired_df = pd.concat([acquired_df, df_multi_hot], axis=1)

# # Final result
# print(acquired_df.head())


# acquired_df.columns


class TaglineCategoryGuesser:
    def __init__(self):
        self.category_keywords = {
            "Artificial Intelligence": ["ai", "machine learning", "deep learning", "neural network"],
            "Mobile": ["mobile", "android", "ios", "app store", "smartphone"],
            "E-Commerce": ["ecommerce", "e-commerce", "shopping", "online store"],
            "FinTech": ["finance", "banking", "payments", "fintech", "crypto", "blockchain"],
            "Healthcare": ["health", "medical", "hospital", "doctor", "pharma"],
            "Social Media": ["social network", "community", "messaging", "chat"],
            "Gaming": ["game", "gaming", "video game", "esports"],
            "Cloud": ["cloud", "saas", "paas", "infrastructure"],
            "EdTech": ["education", "learning", "students", "teaching", "school"],
            "Data Analytics": ["analytics", "data science", "big data", "insights"]
        }

    def guess_category_from_tagline(self, tagline):
        tagline = str(tagline).lower()
        matched = [cat for cat, keywords in self.category_keywords.items()
                   if any(keyword in tagline for keyword in keywords)]

        if len(matched) == 0:
            matched = ["Software", "Advertising"]
        elif len(matched) == 1:
            matched.append("Software")

        return ', '.join(matched)

    def fit(self, data):
        return self

    def transform(self, data):
        df = data.copy()

        # Fill missing values with empty string for 'Tagline'
        df['Tagline'] = df['Tagline'].fillna('')

        # Fill missing Market Categories
        df['Market Categories'] = df['Market Categories'].fillna('Unknown')

        # Apply guessing only on unknowns or blanks
        df['Market Categories'] = df.apply(
            lambda row: self.guess_category_from_tagline(row['Tagline'])
            if str(row['Market Categories']).strip().lower() in ["unknown", "nan", "none", ""]
            else row['Market Categories'],
            axis=1
        )

        return df

    def fit_transform(self, data):
        return self.fit(data).transform(data)


guesser = TaglineCategoryGuesser()
acquired_df = guesser.fit_transform(acquired_df)


with open("tagline_guesser_acquired.pkl", "wb") as f:
    pickle.dump(guesser, f)



class MarketCategoryGeneralizer:
    def __init__(self):
        self.category_mapping = {
    'Software': 'Technology & Software',
    'Advertising': 'Advertising & Marketing',
    'E-Commerce': 'E-Commerce & Online Services',
    'Mobile': 'Mobile & Consumer Electronics',
    'Games': 'Games & Entertainment',
    'Social Media': 'Social Networking & Communication',
    'Cloud': 'Technology & Software',
    'Finance': 'Finance & Payments',
    'Healthcare': 'Healthcare & Wellness',
    'Semiconductors': 'Technology Hardware',
    'Data Analytics': 'Analytics & Data Science',
    'Search': 'Advertising & Marketing',
    'Video': 'Games & Entertainment',
    'Networking': 'Telecom & Networks',
    'Messaging': 'Social Networking & Communication',
    'Education': 'Education & Learning',
    'News': 'Media & News',
    'Photo Sharing': 'Digital Media & Content',
    'Mobile Payments': 'Finance & Payments',
    'Robotics': 'Games & Entertainment',
    'Music': 'Games & Entertainment',
    'Photo Editing': 'Digital Media & Content',
    'Online Rental': 'E-Commerce & Online Services',
    'Location Based Services': 'Telecom & Networks',
    'Enterprise Software': 'Technology & Software',
    'Video Streaming': 'Games & Entertainment',
    'PaaS': 'Technology & Software',
    'SaaS': 'Technology & Software',
    'Health and Wellness': 'Healthcare & Wellness',
    'Web Hosting': 'Technology & Software',
    'Internet of Things': 'IoT (Internet of Things)',
    'Cloud Security': 'Technology & Software',
    'Virtual Currency': 'Finance & Payments',
    'Search Marketing': 'Advertising & Marketing',
    'Mobile Social': 'Social Networking & Communication',
    'Retail': 'Retail & Fashion',
    'Consulting': 'Others & Miscellaneous',
    'Aerospace': 'Others & Miscellaneous',
    'Food Delivery': 'Consumer Goods & Services',
    'Fashion': 'Retail & Fashion',
    'Wine And Spirits': 'Consumer Goods & Services',
    'Streaming': 'Games & Entertainment',
    'Task Management': 'Others & Miscellaneous',
    'Video Chat': 'Social Networking & Communication',
    'Personalization': 'Advertising & Marketing',
    'Shopping': 'E-Commerce & Online Services',
    'Local': 'E-Commerce & Online Services',
    'News': 'Media & News',
    'Fraud Detection': 'Advertising & Marketing',
    'Image Recognition': 'Technology Hardware',
    'Virtualization': 'Games & Entertainment',
    'Analytics': 'Analytics & Data Science',
    'Video on Demand': 'Games & Entertainment',
    'Mobile Payments': 'Finance & Payments',
    'Marketing Automation': 'Advertising & Marketing',
    'Consumer Electronics': 'Mobile & Consumer Electronics',
    'Video Games': 'Games & Entertainment',
    'Public Relations': 'Advertising & Marketing'
    }

    def map_categories(self, row):
        categories = str(row).split(',')
        generalized = []
        for cat in categories:
            cat = cat.strip()
            if cat in self.category_mapping:
                generalized.append(self.category_mapping[cat])
            else:
                generalized.append("Others & Miscellaneous")
        return ', '.join(set(generalized))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['Generalized Market Categories'] = df['Market Categories'].fillna('').apply(self.map_categories)
        return df

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


generalizer = MarketCategoryGeneralizer()
acquired_df = generalizer.fit_transform(acquired_df)



with open("category_generalizer_acquired.pkl", "wb") as f:
    pickle.dump(generalizer, f)


class CountryRegionFiller:
    def __init__(self):
        self.countries = [country.name for country in pycountry.countries]
        self.regions = [
            'California', 'New York', 'Texas', 'Basel', 'Utah', 'Île-de-France', 'Bavaria', 'Ontario',
            'Switzerland', 'United States', 'France', 'Great Britain', 'Israel', 'Sweden', 'Canada',
            'Germany', 'Japan', 'India', 'Denmark', 'China', 'Spain', 'Netherlands', 'Finland',
            'Australia', 'Ireland', 'United Stats of AMerica', 'United Arab Emirates', 'Quebec'
        ]

    def find_place(self, text, place_list):
        for place in place_list:
            if re.search(r'\b' + re.escape(place) + r'\b', str(text)):
                return place
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        for idx, row in df[df['Country (HQ)'].isnull() | df['State / Region (HQ)'].isnull()].iterrows():
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


filler = CountryRegionFiller()
acquired_df = filler.fit_transform(acquired_df)
with open("country_region_filler_acquired.pkl", "wb") as f:
    pickle.dump(filler, f)


columns_to_drop = ['Image', 'CrunchBase Profile', 'Homepage', 'Twitter','Address (HQ)','API','Description','Tagline','Market Categories']
acquired_df.drop(columns=columns_to_drop, inplace=True)
print(acquired_df.columns)



class CategoricalFillerAndEncoder:
    def __init__(self, columns):
        self.columns = columns
        self.modes = {}
        self.label_encoders = {}
        self.label_maps = {}  # Store manual mapping of classes

    def fit(self, X, y=None):
        for col in self.columns:
            mode_val = X[col].mode()[0]
            self.modes[col] = mode_val

            le = LabelEncoder()
            filled = X[col].fillna(mode_val).astype(str)
            le.fit(filled)
            self.label_encoders[col] = le
            self.label_maps[col] = {label: i for i, label in enumerate(le.classes_)}
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



cols = ['City (HQ)', 'State / Region (HQ)', 'Country (HQ)']
encoder = CategoricalFillerAndEncoder(columns=cols)
acquired_df = encoder.fit_transform(acquired_df)

with open("categorical_encoder_acquired.pkl", "wb") as f:
    pickle.dump(encoder, f)


# Save the MultiLabelBinarizer
import pickle
# Save the MultiLabelBinarizer to a file
with open('mlb_acquired.pkl', 'wb') as f:
    pickle.dump(mlb, f)

acquired_df.drop(columns=['Generalized Market Categories'], inplace=True)

acquired_df.columns

acquired_df.drop(columns=['Year Founded'], inplace=True)

acquired_df['Acquired by'].fillna('Salesforce', inplace=True)


acquired_df.isna().sum()

### One-Hot Encoding
 #One-hot encode the 'Status' column
#df = pd.get_dummies(df, columns=['Status'], drop_first=False)
# One-hot encode the 'Terms' column
##df = pd.get_dummies(df, columns=['Terms'], drop_first=False)
#df.head()
### Drop unnecessary columns
# Apply transformation to remove 'Cash,Stock' from the 'Terms' column
#combined_mask = df['Terms_Cash, Stock'] == True
#df.loc[combined_mask, 'Terms_Cash'] = True
#df.loc[combined_mask, 'Terms_Stock'] = True

# Drop combined column
#df = df.drop('Terms_Cash, Stock', axis=1)

# Importing Libraries

# Import and Overview the data

df = pd.read_csv('data/Acquisitions.csv')
df.head()
df.info()
# Check for missing values
print(df.isnull().sum())

value_counts_1 = df['Status'].value_counts()
value_counts_2 = df['Terms'].value_counts()

# Print unique values and their frequencies
print(value_counts_1)
print('------------------')
print(value_counts_2)


### Filling Null Values
# Get mode of the column
mode_value = df['Status'].mode()[0]

# Fill NaN values with the mode
df['Status'].fillna(mode_value, inplace=True)



# Create a custom encoder class that mimics your logic
import pandas as pd
import pickle
from sklearn.base import BaseEstimator, TransformerMixin

class CustomEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.status_cols_ = None
        self.terms_cols_ = None
        
    def fit(self, df):
        # Learn all possible categories
        self.status_cols_ = df['Status'].unique()
        self.terms_cols_ = df['Terms'].unique()
        return self
        
    def transform(self, df):
        df = df.copy()
        
        # One-hot encode Status
        df = pd.get_dummies(df, columns=['Status'], drop_first=False)
        
        # One-hot encode Terms (with special handling)
        df = pd.get_dummies(df, columns=['Terms'], drop_first=False)
        
        # Handle "Cash, Stock" special case
        if 'Terms_Cash, Stock' in df.columns:
            cash_stock_mask = df['Terms_Cash, Stock'] == 1
            df.loc[cash_stock_mask, 'Terms_Cash'] = 1
            df.loc[cash_stock_mask, 'Terms_Stock'] = 1
            df = df.drop('Terms_Cash, Stock', axis=1)
            
        # Ensure all expected columns exist
        expected_cols = [f'Status_{s}' for s in self.status_cols_] + \
                       [f'Terms_{t}' for t in self.terms_cols_ if t != 'Cash, Stock']
        
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
                
        return df[expected_cols]

# Create and save encoder
encoder = CustomEncoder()
encoder.fit(df)  # Your original dataframe
# Save the custom encoder
with open('custom_acquisitions_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)



#Drop unuseful columns that are not needed for the analysis
df.drop(columns=["Acquisition Profile", "News", "News Link"], inplace=True)
# Recheck after conversion
print("\n🔹 Data Types After Converting 'Price':")
print(df.dtypes)
## Extract Date Features
# Convert 'Deal announced on' to datetime
df['Deal_date'] = pd.to_datetime(df['Deal announced on'], dayfirst=True, errors='coerce')


# Extract day, month, and day of week
df['Deal_day'] = df['Deal_date'].dt.day
df['Deal_month'] = df['Deal_date'].dt.month
df['Deal_dayofweek'] = df['Deal_date'].dt.dayofweek  # Monday=0, Sunday=6

# Drop combined column
df = df.drop('Deal announced on', axis=1)
df = df.drop('Deal_date', axis=1)
df.info()
df.head()

null_indices = df[df.isna().any(axis=1)].index

# Display the indices with null values
print(null_indices)

for column in df.columns:
    mode_value = df[column].mode()[0]  # Get the mode value of the column
    df[column].fillna(mode_value, inplace=True)

# Check the result
print(df.isna().sum())

def find_company_column(df):
    for col in df.columns:
        if "company" in col.lower():
            return col
    raise ValueError("No company name column found.")

# Start with acquisition data
final_df = df.copy()

# Mapping for left_on column and its corresponding dataset
merge_targets = [
    ('Acquired Company', acquired_df, '_Acquired'),
    ('Acquiring Company', data, '_Acquiring')
]

# Perform merges in loop
for left_key, company_data, suffix in merge_targets:
    company_col = find_company_column(company_data)

    # Strip and lower case the company names for matching
    final_df[left_key] = final_df[left_key].str.strip().str.lower()
    company_data[company_col] = company_data[company_col].str.strip().str.lower()

    # Merge
    final_df = final_df.merge(
        company_data,
        how='left',
        left_on=left_key,
        right_on=company_col,
        suffixes=('', suffix)
    )

    # Drop the extra company column if you want
    final_df.drop(columns=[company_col], inplace=True, errors='ignore')



# Done!
print(final_df.head())

final_df.shape

null_indices = final_df[final_df.isna().any(axis=1)].index

# Print them
print(null_indices)
final_df.dropna(axis=0, inplace=True)

import pickle

# Calculate and save modes
mode_values = {col: final_df[col].mode()[0] if not final_df[col].mode().empty else None 
               for col in final_df.columns}
with open('mode_acquisitions_imputer.pkl', 'wb') as f:
    pickle.dump(mode_values, f)

# Load and apply
#with open('mode_acquisitions_imputer.pkl', 'rb') as f:
   # modes = pickle.load(f)

#for col, mode_val in modes.items():
   ## if col in df.columns and mode_val is not None:
       # df[col].fillna(mode_val, inplace=True)

final_df.shape

first_column = acquired_df.columns[0]

# Define the value you're searching for
search_value = "hotjobs"  # or any value you want to look for

# Find rows where the first column matches the value
matching_rows = acquired_df[acquired_df[first_column].str.strip().str.lower() == search_value.strip().lower()]
print(matching_rows)



for x in final_df.isna().sum():
    print(x)

from sklearn.preprocessing import LabelEncoder

# Make a copy so you don’t overwrite your original
encoded_df = final_df.copy()

# Loop over all columns
for col in encoded_df.columns:
    if encoded_df[col].dtype in ['object', 'bool']:  # Include strings and booleans
        le = LabelEncoder()
        encoded_df[col] = encoded_df[col].astype(str)  # Convert all values to string
        encoded_df[col] = le.fit_transform(encoded_df[col])


############################################The encoder pereformed on all columns #########################################
#that mean we have to save the encoder for each column
# Save label encoder
import pickle
# Save the LabelEncoder to a file
with open('target_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)


encoded_df

for col in encoded_df.columns:
    if (encoded_df[col].nunique() <= 1) or (encoded_df[col].eq(0).all()):
        encoded_df.drop(columns=col, inplace=True)

encoded_df.shape

encoded_df['Deal size class']

for x in df.isna().sum():
    print(x)

for x in encoded_df.columns:
    print(x)

from sklearn.decomposition import PCA

# ... [Your existing code up to the correlation matrix calculation] ...

# Calculate correlation matrix
correlation_matrix = encoded_df.corr()

# Get correlation of each feature with 'Deal size class'
price_correlation = correlation_matrix['Deal size class']

# Set the correlation threshold (e.g., absolute value of correlation > 0.15)
threshold = 0.15

# Exclude target variable and raw categorical columns
exclude_columns = ['Deal size class', 'City (HQ)', 'State / Region (HQ)', 'Country (HQ)']
selected_features = [col for col in price_correlation[price_correlation.abs() >= threshold].index.tolist() 
                    if col.lower() not in [x.lower() for x in exclude_columns]]

# Debug: Print selected features to confirm exclusions
print("Selected Features (excluding 'Deal size class' and raw categorical columns):", selected_features)

# Save selected features
with open('selected_features.pkl', 'wb') as f:
    pickle.dump(selected_features, f)

# Create filtered dataset (X) using selected features
X = encoded_df[selected_features]
y = encoded_df['Deal size class']

# Proceed with scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Save scaler
with open('final_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Split data
X_train, X_test, y_train_log, y_test_log = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Apply PCA
pca = PCA(n_components=0.95)  # Retain 95% of variance
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Save PCA
with open('final_pca.pkl', 'wb') as f:
    pickle.dump(pca, f)

# ... [Rest of your script for model training, evaluation, etc. remains unchanged] ...


from sklearn.model_selection import cross_val_score, train_test_split


y_train_log

C_values = [0.1, 1, 10]
for c in C_values:
    model = SVC(C=c, kernel='rbf', random_state=42)
    scores = cross_val_score(model, X_train, y_train_log, cv=5)
    print(f"C={c}, Accuracy={scores.mean():.3f}")


from sklearn.metrics import accuracy_score


# Track the best model
best_model = None
best_gamma = None
best_test_acc = 0
best_cv_acc = 0


C_values = [0.1, 1, 10]
for c in C_values:
    model = SVC(C=c, kernel='rbf', random_state=42)

    # Cross-validation on training data
    cv_scores = cross_val_score(model, X_train_pca, y_train_log, cv=5)
    print(f"C={c}, CV Accuracy={cv_scores.mean():.3f}")

    # Train on full training set
    model.fit(X_train_pca, y_train_log)

    # Evaluate on test set
    y_pred = model.predict(X_test_pca)
    test_acc = accuracy_score(y_test_log, y_pred)
    print(f"C={c}, Test Accuracy={test_acc:.3f}\n")
    

C_fixed = 1
kernel_types = ['linear', 'rbf', 'poly', 'sigmoid']

for kernel in kernel_types:
    model = SVC(C=C_fixed, kernel=kernel, random_state=42)

    # Cross-validation on training set
    cv_scores = cross_val_score(model, X_train_pca, y_train_log, cv=5)
    print(f"Kernel='{kernel}', CV Accuracy={cv_scores.mean():.3f}")

    # Train on full training set
    model.fit(X_train_pca, y_train_log)

    # Predict on test set
    y_pred = model.predict(X_test_pca)
    test_acc = accuracy_score(y_test_log, y_pred)
    print(f"Kernel='{kernel}', Test Accuracy={test_acc:.3f}\n")

C_fixed = 1
kernel_fixed = 'sigmoid'
gamma_values = ['scale', 'auto', 0.01, 0.1, 1]

for gamma in gamma_values:
    model = SVC(C=C_fixed, kernel=kernel_fixed, gamma=gamma, random_state=42)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_pca, y_train_log, cv=5)
    print(f"gamma={gamma}, CV Accuracy={cv_scores.mean():.3f}")

    # Train on full training set
    model.fit(X_train_pca, y_train_log)

    # Evaluate on test set
    y_pred = model.predict(X_test_pca)
    test_acc = accuracy_score(y_test_log, y_pred)
    print(f"gamma={gamma}, Test Accuracy={test_acc:.3f}\n")


C_values = [0.1, 1, 10]
for c in C_values:
    model = SVC(C=c, kernel='rbf', random_state=42)

    # Cross-validation on training data
    cv_scores = cross_val_score(model, X_train, y_train_log, cv=5)
    print(f"C={c}, CV Accuracy={cv_scores.mean():.3f}")

    # Train on full training set
    model.fit(X_train, y_train_log)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test_log, y_pred)
    print(f"C={c}, Test Accuracy={test_acc:.3f}\n")

C_fixed = 1
kernel_types = ['linear', 'rbf', 'poly', 'sigmoid']

for kernel in kernel_types:
    model = SVC(C=C_fixed, kernel=kernel, random_state=42)

    # Cross-validation on training set
    cv_scores = cross_val_score(model, X_train, y_train_log, cv=5)
    print(f"Kernel='{kernel}', CV Accuracy={cv_scores.mean():.3f}")

    # Train on full training set
    model.fit(X_train, y_train_log)

    # Predict on test set
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test_log, y_pred)
    print(f"Kernel='{kernel}', Test Accuracy={test_acc:.3f}\n")

C_fixed = 1
kernel_fixed = 'sigmoid'
gamma_values = ['scale', 'auto', 0.01, 0.1, 1,10]

for gamma in gamma_values:
    model = SVC(C=C_fixed, kernel=kernel_fixed, gamma=gamma, random_state=42)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train_log, cv=5)
    print(f"gamma={gamma}, CV Accuracy={cv_scores.mean():.3f}")

    # Train on full training set
    model.fit(X_train, y_train_log)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test_log, y_pred)
    print(f"gamma={gamma}, Test Accuracy={test_acc:.3f}\n")

# Based on your results
best_model = SVC(C=1, kernel='sigmoid', gamma='auto', random_state=42)
best_model.fit(X_train, y_train_log)

# Save SVC model
import pickle
with open('best_svc_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score



# Example values to test
n_estimators_list = [50, 100, 200,300]

# Fixed parameters
max_depth_fixed = 10

for n in n_estimators_list:
    model = RandomForestClassifier(n_estimators=n, max_depth=max_depth_fixed, random_state=42)

    # Cross-validation on training set
    cv_scores = cross_val_score(model, X_train, y_train_log, cv=5)
    print(f"n_estimators={n}, CV Accuracy={cv_scores.mean():.3f}")

    # Train on full training set
    model.fit(X_train, y_train_log)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test_log, y_pred)
    print(f"n_estimators={n}, Test Accuracy={test_acc:.3f}\n")

n_estimators_list =  200

# Fixed parameters
max_depth_fixed = [10, 20, 50,100]

for n in max_depth_fixed:
    model = RandomForestClassifier(n_estimators=n_estimators_list, max_depth=n, random_state=42)

    # Cross-validation on training set
    cv_scores = cross_val_score(model, X_train, y_train_log, cv=5)
    print(f"n_estimators={n}, CV Accuracy={cv_scores.mean():.3f}")

    # Train on full training set
    model.fit(X_train, y_train_log)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test_log, y_pred)
    print(f"n_estimators={n}, Test Accuracy={test_acc:.3f}\n")

n_estimators_list =  200

# Fixed parameters
max_depth_fixed = 10
min=[10,8,20,50]
for n in min:
    model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=n,random_state=42)

    # Cross-validation on training set
    cv_scores = cross_val_score(model, X_train, y_train_log, cv=5)
    print(f"n_estimators={n}, CV Accuracy={cv_scores.mean():.3f}")

    # Train on full training set
    model.fit(X_train, y_train_log)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test_log, y_pred)
    print(f"n_estimators={n}, Test Accuracy={test_acc:.3f}\n")


# Example values to test
n_estimators_list = [50, 100, 200,300]

# Fixed parameters
max_depth_fixed = 10

for n in n_estimators_list:
    model = RandomForestClassifier(n_estimators=n, max_depth=max_depth_fixed, random_state=42)

    # Cross-validation on training set
    cv_scores = cross_val_score(model, X_train_pca, y_train_log, cv=5)
    print(f"n_estimators={n}, CV Accuracy={cv_scores.mean():.3f}")

    # Train on full training set
    model.fit(X_train_pca, y_train_log)

    # Evaluate on test set
    y_pred = model.predict(X_test_pca)
    test_acc = accuracy_score(y_test_log, y_pred)
    print(f"n_estimators={n}, Test Accuracy={test_acc:.3f}\n")

n_estimators_list =  400

# Fixed parameters
max_depth_fixed = [10, 20, 50,100]

for n in max_depth_fixed:
    model = RandomForestClassifier(n_estimators=n_estimators_list, max_depth=n, random_state=42)

    # Cross-validation on training set
    cv_scores = cross_val_score(model, X_train_pca, y_train_log, cv=5)
    print(f"n_estimators={n}, CV Accuracy={cv_scores.mean():.3f}")

    # Train on full training set
    model.fit(X_train_pca, y_train_log)

    # Evaluate on test set
    y_pred = model.predict(X_test_pca)
    test_acc = accuracy_score(y_test_log, y_pred)
    print(f"n_estimators={n}, Test Accuracy={test_acc:.3f}\n")

n_estimators_list =  200

# Fixed parameters
max_depth_fixed = 1
min=[10,20,50]
for n in min:
    model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=n,random_state=42)

    # Cross-validation on training set
    cv_scores = cross_val_score(model, X_train_pca, y_train_log, cv=5)
    print(f"n_estimators={n}, CV Accuracy={cv_scores.mean():.3f}")

    # Train on full training set
    model.fit(X_train_pca, y_train_log)

    # Evaluate on test set
    y_pred = model.predict(X_test_pca)
    test_acc = accuracy_score(y_test_log, y_pred)
    print(f"n_estimators={n}, Test Accuracy={test_acc:.3f}\n")

# Based on your results
final_model = RandomForestClassifier(n_estimators=200,
                                   max_depth=10,
                                   random_state=42)
final_model.fit(X_train_pca, y_train_log)

# Save model
import pickle
with open('best_rf_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

#import xgboost as xgb
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score


# # Define the hyperparameter options
# max_depth_values = [3, 6, 9]  # Choices for max_depth
# learning_rate_values = [0.01, 0.1, 0.3]  # Choices for learning_rate

# # Fixed hyperparameters
# n_estimators = 100
# objective = 'multi:softmax'  # Multi-class classification
# num_class = len(set(y))  # Number of classes in the target variable

# # Loop through different choices of max_depth
# for max_depth in max_depth_values:
#     print(f"Training with max_depth = {max_depth}")

#     # Loop through different choices of learning_rate
#     for learning_rate in learning_rate_values:
#         print(f"  Training with learning_rate = {learning_rate}")

#         # Initialize the XGBoost classifier with the current hyperparameters
#         model = xgb.XGBClassifier(
#             max_depth=max_depth,
#             learning_rate=learning_rate,
#             n_estimators=n_estimators,
#             objective=objective,
#             num_class=num_class,
#             random_state=42
#         )

#         # Train the model
#         model.fit(X_train, y_train_log)

#         # Predict on the test set
#         y_pred = model.predict(X_test)

#         # Calculate the accuracy
#         accuracy = accuracy_score(y_test_log, y_pred)

#         print(f"    Accuracy for max_depth={max_depth}, learning_rate={learning_rate}: {accuracy:.4f}")


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Fixed parameters
n_estimators_list = 200
max_depth_fixed = [3, 6, 10, 20]

# Loop through different values of max_depth
for n in max_depth_fixed:
    model = GradientBoostingClassifier(n_estimators=n_estimators_list, max_depth=n, learning_rate=0.2,random_state=42)

    # Cross-validation on training set
    cv_scores = cross_val_score(model, X_train_pca, y_train_log, cv=5)
    print(f"max_depth={n}, CV Accuracy={cv_scores.mean():.3f}")

    # Train on full training set
    model.fit(X_train_pca, y_train_log)

    # Evaluate on test set
    y_pred = model.predict(X_test_pca)
    test_acc = accuracy_score(y_test_log, y_pred)
    print(f"max_depth={n}, Test Accuracy={test_acc:.3f}\n")

# Fixed parameters
n_estimators_list = [100,50,200]

# Loop through different values of max_depth
for n in n_estimators_list:
    model = GradientBoostingClassifier(n_estimators=n, max_depth=3, learning_rate=0.2,random_state=42)

    # Cross-validation on training set
    cv_scores = cross_val_score(model, X_train_pca, y_train_log, cv=5)
    print(f"max_depth={n}, CV Accuracy={cv_scores.mean():.3f}")

    # Train on full training set
    model.fit(X_train_pca, y_train_log)

    # Evaluate on test set
    y_pred = model.predict(X_test_pca)
    test_acc = accuracy_score(y_test_log, y_pred)
    print(f"max_depth={n}, Test Accuracy={test_acc:.3f}\n")

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Fixed parameters
n_estimators_list = 200
max_depth_fixed = [3, 6, 10, 20]

# Loop through different values of max_depth
for n in max_depth_fixed:
    model = GradientBoostingClassifier(n_estimators=n_estimators_list, max_depth=n, random_state=42)

    # Cross-validation on training set
    cv_scores = cross_val_score(model, X_train, y_train_log, cv=5)
    print(f"max_depth={n}, CV Accuracy={cv_scores.mean():.3f}")

    # Train on full training set
    model.fit(X_train, y_train_log)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test_log, y_pred)
    print(f"max_depth={n}, Test Accuracy={test_acc:.3f}\n")

# Fixed parameters
n_estimators_list = [100,50,200,300]
max_depth_fixed = 6

# Loop through different values of max_depth
for n in n_estimators_list:
    model = GradientBoostingClassifier(n_estimators=n, max_depth=3, random_state=42)

    # Cross-validation on training set
    cv_scores = cross_val_score(model, X_train, y_train_log, cv=5)
    print(f"max_depth={n}, CV Accuracy={cv_scores.mean():.3f}")

    # Train on full training set
    model.fit(X_train, y_train_log)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test_log, y_pred)
    print(f"max_depth={n}, Test Accuracy={test_acc:.3f}\n")

# Based on current best
final_gb_model = GradientBoostingClassifier(n_estimators=100,
                                          max_depth=3,
                                          learning_rate=0.2,
                                          random_state=42)
final_gb_model.fit(X_train_pca, y_train_log)

# # Save model
# import joblib  # More efficient than pickle for sklearn
# joblib.dump(final_gb_model, 'best_gb_model.joblib')


import pickle
with open('best_gb_model.pkl', 'wb') as f:
    pickle.dump(final_gb_model, f)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

n_neighbors_list = [3, 5, 7, 9, 11]

# Loop over different values of n_neighbors
for k in n_neighbors_list:
    model = KNeighborsClassifier(n_neighbors=k)

    # Cross-validation on training set
    cv_scores = cross_val_score(model, X_train_pca, y_train_log, cv=5)
    print(f"n_neighbors={k}, CV Accuracy={cv_scores.mean():.3f}")

    # Train on full training set
    model.fit(X_train_pca, y_train_log)

    # Evaluate on test set
    y_pred = model.predict(X_test_pca)
    test_acc = accuracy_score(y_test_log, y_pred)
    print(f"n_neighbors={k}, Test Accuracy={test_acc:.3f}\n")

n_neighbors_list = [3, 5, 7, 9, 11]

# Loop over different values of n_neighbors
for k in n_neighbors_list:
    model = KNeighborsClassifier(n_neighbors=k)

    # Cross-validation on training set
    cv_scores = cross_val_score(model, X_train, y_train_log, cv=5)
    print(f"n_neighbors={k}, CV Accuracy={cv_scores.mean():.3f}")

    # Train on full training set
    model.fit(X_train, y_train_log)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test_log, y_pred)
    print(f"n_neighbors={k}, Test Accuracy={test_acc:.3f}\n")

# Fixed hyperparameters
n_neighbors_fixed = 11
weights_fixed = 'uniform'

# Algorithms to try
algorithm_list = ['auto', 'ball_tree', 'kd_tree', 'brute']

# Loop over algorithms
for algo in algorithm_list:
    model = KNeighborsClassifier(n_neighbors=n_neighbors_fixed, weights=weights_fixed, algorithm=algo)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_pca, y_train_log, cv=5)
    print(f"algorithm={algo}, CV Accuracy={cv_scores.mean():.3f}")

    # Train and test
    model.fit(X_train_pca, y_train_log)
    y_pred = model.predict(X_test_pca)
    test_acc = accuracy_score(y_test_log, y_pred)
    print(f"algorithm={algo}, Test Accuracy={test_acc:.3f}\n")

best_knn = KNeighborsClassifier(n_neighbors=11,
                              algorithm='brute',
                              weights='uniform',
                              metric='euclidean')  # Change if metric test shows better
best_knn.fit(X_train_pca, y_train_log)

# Save model
# from joblib import dump
# dump(best_knn, 'best_knn_model.joblib')
import pickle
with open('best_knn_model.pkl', 'wb') as f:
    pickle.dump(best_knn, f)

from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

'''from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
#import xgboost as xgb

# Define models with proper sklearn compatibility
models = [
    ('svc', SVC(kernel='sigmoid', C=1, gamma='auto', probability=True, random_state=42)),
    ('xgb', xgb.XGBClassifier(
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        objective='multi:softmax',
        random_state=42,
        use_label_encoder=False,  # Important for sklearn compatibility
        eval_metric='mlogloss'  # Needed for multiclass
    )),
    ('gb', GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.2,
        random_state=42
    )),
    ('knn', KNeighborsClassifier(
        n_neighbors=11,
        algorithm='ball_tree',
        weights='uniform'
    ))
]

# Create voting classifier
voting_clf = VotingClassifier(
    estimators=models,
    voting='soft',
    n_jobs=-1  # Use all available cores
)

# Cross-validation
try:
    cv_scores = cross_val_score(voting_clf, X_train_pca, y_train_log, cv=5)
    print(f"Voting Classifier CV Accuracy: {cv_scores.mean():.3f}")
    
    # Final training
    voting_clf.fit(X_train_pca, y_train_log)
    y_pred = voting_clf.predict(X_test_pca)
    test_accuracy = accuracy_score(y_test_log, y_pred)
    print(f"Voting Classifier Test Accuracy: {test_accuracy:.3f}")
    
except Exception as e:
    print(f"Error: {str(e)}")
    print("Trying alternative approach...")
    
    # Fallback: Train each model separately and manually vote
    for name, model in models:
        model.fit(X_train_pca, y_train_log)
        print(f"{name} accuracy: {model.score(X_test_pca, y_test_log):.3f}")'''

'''from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

# Define base models
svc_model = SVC(kernel='sigmoid', C=1, gamma='auto', probability=True)
gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=11, algorithm='ball_tree', weights='uniform')
xgb_mv = xgb.XGBClassifier(
    max_depth=3,
    learning_rate=0.1,
    n_estimators=100,
    objective='multi:softmax',
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

# Meta-model (final estimator)
meta_model = LogisticRegression(max_iter=700)

# Stacking classifier
stacking_clf = StackingClassifier(
    estimators=[
        ('svc', svc_model),
        ('xgb', xgb_mv),
        ('gb', gb_model),
        ('knn', knn_model),
    ],
    final_estimator=meta_model,
    cv=5,
    passthrough=False,  # Set to True if you want to include original features for meta-model
    n_jobs=-1
)

# Cross-validation on training set
cv_scores = cross_val_score(stacking_clf, X_train, y_train_log, cv=5)
print(f"Stacking Classifier CV Accuracy: {cv_scores.mean():.3f}")

# Train on full training set
stacking_clf.fit(X_train, y_train_log)

# Evaluate on test set
y_pred = stacking_clf.predict(X_test)
test_accuracy = accuracy_score(y_test_log, y_pred)
print(f"Stacking Classifier Test Accuracy: {test_accuracy:.3f}")'''



# Initialize the SVC model
svc_model = SVC(kernel='sigmoid', C=1, gamma='auto')

# Cross-validation on training set
cv_scores_svc = cross_val_score(svc_model, X_train, y_train_log, cv=5)
print(f"SVC Model CV Accuracy: {cv_scores_svc.mean():.3f}")

# Train on full training set
svc_model.fit(X_train, y_train_log)

# Evaluate on test set
y_pred_svc = svc_model.predict(X_test)
test_acc_svc = accuracy_score(y_test_log, y_pred_svc)
print(f"SVC Model Test Accuracy: {test_acc_svc:.3f}\n")

rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=10, random_state=42)

# Cross-validation on training set
cv_scores_rf = cross_val_score(rf_model, X_train_pca, y_train_log, cv=5)
print(f"Random Forest Model CV Accuracy: {cv_scores_rf.mean():.3f}")

# Train on full training set
rf_model.fit(X_train_pca, y_train_log)

# Evaluate on test set
y_pred_rf = rf_model.predict(X_test_pca)
test_acc_rf = accuracy_score(y_test_log, y_pred_rf)
print(f"Random Forest Model Test Accuracy: {test_acc_rf:.3f}\n")

gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.2, random_state=42)

# Cross-validation on training set
cv_scores_gb = cross_val_score(gb_model, X_train_pca, y_train_log, cv=5)
print(f"Gradient Boosting Model CV Accuracy: {cv_scores_gb.mean():.3f}")

# Train on full training set
gb_model.fit(X_train_pca, y_train_log)

# Evaluate on test set
y_pred_gb = gb_model.predict(X_test_pca)
test_acc_gb = accuracy_score(y_test_log, y_pred_gb)
print(f"Gradient Boosting Model Test Accuracy: {test_acc_gb:.3f}\n")


knn_model = KNeighborsClassifier(n_neighbors=11, algorithm='ball_tree', weights='uniform')

# Cross-validation on training set
cv_scores_knn = cross_val_score(knn_model, X_train_pca, y_train_log, cv=5)
print(f"KNeighbors Model CV Accuracy: {cv_scores_knn.mean():.3f}")

# Train on full training set
knn_model.fit(X_train_pca, y_train_log)

# Evaluate on test set
y_pred_knn = knn_model.predict(X_test_pca)
test_acc_knn = accuracy_score(y_test_log, y_pred_knn)
print(f"KNeighbors Model Test Accuracy: {test_acc_knn:.3f}\n")

# Initialize the XGBoost model
'''xgb_mv = xgb.XGBClassifier(
    max_depth=3,
    learning_rate=0.2,
    n_estimators=100,
    objective='multi:softmax',
    random_state=42
)

# Cross-validation on training set
cv_scores_xgb = cross_val_score(xgb_mv, X_train, y_train_log, cv=5)
print(f"XGBoost Model CV Accuracy: {cv_scores_xgb.mean():.3f}")

# Train on full training set
xgb_mv.fit(X_train, y_train_log)

# Evaluate on test set
y_pred_xgb = xgb_mv.predict(X_test)
test_acc_xgb = accuracy_score(y_test_log, y_pred_xgb)
print(f"XGBoost Model Test Accuracy: {test_acc_xgb:.3f}\n")'''

