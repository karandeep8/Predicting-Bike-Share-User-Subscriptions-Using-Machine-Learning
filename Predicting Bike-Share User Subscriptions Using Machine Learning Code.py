# -*- coding: utf-8 -*-
# Importing the Libraries


# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from scipy.stats import ttest_rel
warnings.simplefilter(action='ignore', category=FutureWarning)

"""# Loading and exploration of dataset"""

def load_and_explore_data():
    # Loading the dataset
    df = pd.read_csv('updated_merged_daily_rent_weather_January_2023.csv')

    # EDA
    # Summary of Statistics of Train Dataset
    print("Information of the dataset")
    print(df.info())

    # Checking the first 5 rows of the dataset
    print("\n First 5 rows of the Dataset")
    print(df.head())

    # Checking the distribution of target feature that is 'member_casual'
    print("\n Distribution of Target Variable")
    # counting the values of member_casual
    target_counts = df['member_casual'].value_counts()
    print(target_counts)

    # Class Imbalance Ratio: It is important to know the frequency of each class
    # of target variable if one class is more than other class, model can become
    # bias

    class_ratio = target_counts.min() / target_counts.max()
    print(f"\nClass Imbalance Ratio is: {class_ratio:.3f}")
    # threshold values are chosen arbitary
    print(f"Dataset is {'balanced' if class_ratio > 0.8 else 'moderately imbalanced' if class_ratio > 0.5 else 'highly imbalanced'}")

    # Meta Data: Data about data, answers the question like when, how, what about
    # the data, here for us the meta data will be total number of samples and
    # features
    print("\nMeta Data of Dataset:")
    print(f"Total samples: {len(df)}")
    print(f"Total features: {df.shape[1] - 1}")  # Excluding target

    # Visualization of Target Variable Distribution (using bar chart and pie chart)

    plt.figure(figsize=(18, 10))
    plt.subplot(1, 2, 1)
    target_counts.plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title('Target Variable Distribution', fontsize=20, fontweight='bold')
    plt.xlabel('Target Class', fontsize=15, fontweight='bold')
    plt.ylabel('Count', fontsize=15, fontweight='bold')
    plt.xticks(rotation=0, fontsize=15, fontweight='bold')

    plt.subplot(1, 2, 2)
    plt.pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%',
            colors=['skyblue', 'salmon'], textprops={'fontsize': 15, 'fontweight': 'bold'})
    plt.title('Target Class Proportion', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Target Variable Distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    return df, target_counts

"""# Feature Engineering

## The temporal feature engineering

Is important for machine learning model because it transforms raw timestamp data into meaningful behavioral patterns.

Given date format '04-01-2023': This is meaningless for machine learning model, it can understand the hidden temporal pattern with this format.

We will be extracting weekely patterns from the data, in order to understand the weekely affect on our target variable 'member_casual'. Since we are analysing January data, and first week is new-year's week, many people can take new resolutions of being healthy, so we will take that in the account also.

We need to create categorical variables for weekends and weekdays
Monday, Tuesday, Wednesday, Thursday, and Friday -> 0 to 4
Saturday and Sunday -> 5 and 6
"""

def improved_feature_engineering_corrected(df):
    """
    Feature engineering corrected for January-only date data (no time information)
    """

    print("Starting Feature Engineering...")
    print("Original columns:", df.columns.tolist())

    # Creating a copy to avoid modifying original data
    df_processed = df.copy()

    # Note: Removing those features which can cause data leakage
    features_to_remove = ['ended_at', 'end_station_name', 'duration_of_riding', 'description']
    existing_features_to_remove = [col for col in features_to_remove if col in df_processed.columns]

    if existing_features_to_remove:
        print(f"\nRemoving data leakage features: {existing_features_to_remove}")
        df_processed = df_processed.drop(columns=existing_features_to_remove)

    # Converting started_at to datetime if it's not already
    # not using the format parameter as earlier it was passed and after running
    # the code it was noted format of dates that was passed was wrong
    # then it came to know you don't have to pass the format of dates, pandas
    # can figure it out on its own.

    if 'started_at' in df_processed.columns:
        df_processed['started_at'] = pd.to_datetime(df_processed['started_at'], errors='coerce')

        # Extracting meaningful temporal features from date-only data
        df_processed['day_of_week'] = df_processed['started_at'].dt.dayofweek
        df_processed['day'] = df_processed['started_at'].dt.day
        df_processed['is_weekend'] = (df_processed['day_of_week'] >= 5).astype(int)

        # Create week of month (potentially useful for January patterns)
        df_processed['week_of_month'] = ((df_processed['started_at'].dt.day - 1) // 7) + 1

        # New Year period (first week might have different patterns)
        df_processed['is_new_year_week'] = (df_processed['started_at'].dt.day <= 7).astype(int)

        # Creating weekday categories
        def categorize_weekday(day_of_week):
            if pd.isna(day_of_week):
                return 'unknown'
            elif day_of_week < 5:
                return 'weekday'
            else:
                return 'weekend'

        df_processed['weekday_category'] = df_processed['day_of_week'].apply(categorize_weekday)

        # Drop the original started_at column as we have extracted useful features
        df_processed = df_processed.drop('started_at', axis=1)

    # start_station name contains a lot unique values, because of which it has lot
    # cardinality and thus we need to reduce its cardinality
    # Handling start_station_name in order to reduce cardinality through basic geographic clustering
    if 'start_station_name' in df_processed.columns:
        def create_station_region(station_name):
            if pd.isna(station_name):
                return 'unknown'

            station_name = str(station_name).lower()

            # Geographic clustering based on common DC area patterns
            # look for the terms and assign them new values
            if any(term in station_name for term in ['nw', 'northwest']):
                return 'northwest'
            elif any(term in station_name for term in ['ne', 'northeast']):
                return 'northeast'
            elif any(term in station_name for term in ['sw', 'southwest']):
                return 'southwest'
            elif any(term in station_name for term in ['se', 'southeast']):
                return 'southeast'
            elif any(term in station_name for term in ['metro', 'station']):
                return 'metro_area'
            elif any(term in station_name for term in ['university', 'college', 'school']):
                return 'education'
            elif any(term in station_name for term in ['park', 'rec', 'recreation']):
                return 'recreation'
            elif any(term in station_name for term in ['mall', 'center', 'square']):
                return 'commercial'
            else:
                return 'other'

        df_processed['station_region'] = df_processed['start_station_name'].apply(create_station_region)

        # Removing the original high-cardinality feature
        df_processed = df_processed.drop('start_station_name', axis=1)

    # Weather condition processing in order to simplify complex weather descriptions
    if 'conditions' in df_processed.columns:
        def simplify_weather(condition):
            if pd.isna(condition):
                return 'unknown'

            condition = str(condition).lower()

            if 'rain' in condition or 'drizzle' in condition:
                return 'rainy'
            elif 'clear' in condition:
                return 'clear'
            elif 'cloud' in condition or 'overcast' in condition:
                return 'cloudy'
            elif 'partial' in condition:
                return 'partly_cloudy'
            elif 'snow' in condition:
                return 'snowy'
            else:
                return 'other'

        df_processed['weather_simple'] = df_processed['conditions'].apply(simplify_weather)

    print(f"\nFeature engineering completed!")
    print(f"Original features: {df.shape[1]}")
    print(f"Processed features: {df_processed.shape[1]}")
    print("Final columns:", df_processed.columns.tolist())
    print("\nNote: Time-based features (hour, rush_hour) not available with date-only data")
    print('\nFirst 5 rows of dataset after feature engineering')
    print(df_processed.head())

    # Save the processed dataframe to CSV file
    output_filename = 'bike_sharing_feature_engineered.csv'
    df_processed.to_csv(output_filename, index=False)
    print(f"\nProcessed dataset saved to: {output_filename}")
    print(f"File contains {df_processed.shape[0]} rows and {df_processed.shape[1]} columns")

    return df_processed

"""## Data Leakage

The features or information that will not be able in the real world to make prediction and those features has been used to train the model, this can lead to poor generalization.

Our goal is to predict whether the rider will become the member or not member/casual at the moment they start their ride, not after it's completed.

**ended_at ->** When someone picks up a bike we don't know when they'll return it. So in real world this feature will not be present to the model. Thus it should be removed while training the model.


**duration_of_riding ->** Again this information will not be present until the trip is completed. Thus need to be removed. (Just for an example lets say there is pattern that short trip is usually done by members and long trips are done by casual thus our model will learn that short trips means member and long trip means casual)


**end_station_name ->** In real world, we cannot know where the trip is going to end.

**description ->** can lead to weather data lekage
"""

def check_for_data_leakage(df, target_col='member_casual'):
    # Checking features that can lead to data leakage
    print("\nChecking for potential data leakage...")

    # Features that should be avoided as they can lead to data leakage
    leakage_indicators = [
        'ended_at', 'end_station_name', 'duration_of_riding', 'description'
    ]

    potential_leakage = []
    for col in df.columns:
        if col != target_col:
            col_lower = col.lower()
            if any(indicator in col_lower for indicator in leakage_indicators):
                potential_leakage.append(col)

    if potential_leakage:
        print("Note: Potential data leakage detected in features:")
        for feature in potential_leakage:
            print(f"  - {feature}")
    else:
        print("No obvious fetaure that can lead to data leakage is detected")

    return potential_leakage

"""## Analyzing and Visualizing Missing Values"""

def analyze_missing_values(df):
    # Checking for Missing Data

    missing_values = df.isnull().sum()

    # Percentage of Missing Values as it is helpful in checking the quality of dataset
    missing_percentage = (df.isnull().sum() / len(df)) * 100

    # Summary of Missing Data
    missing_summary = pd.DataFrame({
        'Missing Values': missing_values,
        'Missing Percentage (%)': missing_percentage
    })

    # We will only print those columns which have missing values in them
    missing_summary = missing_summary[missing_summary['Missing Values'] > 0]

    # Sorting the columns which have missing values in them in descending order
    missing_summary = missing_summary.sort_values('Missing Percentage (%)', ascending=False)

    print("\nMissing Value Analysis")
    print(missing_summary)

    # Visualization of Missing Values

    if len(missing_summary) > 0:
        plt.figure(figsize=(18, 10))
        plt.xticks(fontsize=15, fontweight='bold')
        plt.yticks(fontsize=15, fontweight='bold')

        plt.bar(missing_summary.index, missing_summary['Missing Percentage (%)'])
        plt.xticks(rotation=90)
        plt.title('Percentage of Missing Values by Column', fontsize=20, fontweight='bold')
        plt.ylabel('Percentage Missing (%)', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig('Percentage of Missing Values by Column.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("No missing values found in the dataset!")

    return missing_summary

"""## Identifying Feature Types"""

def identify_feature_types(df):
    # Numerical vs. Categorical Column

    # Our bike-sharing dataset have both types of features Numerical as well as Categorical.
    # Before building the model, it is useful to know which columns are which.
    # As later while building the model, we need to perform one hot encoding of categorical features.

    # Identifying numerical columns this mean columns having integer or float datatypes
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # The target column is categorical
    numerical_features = [col for col in numerical_cols if col != 'member_casual']

    # Identifying categorical columns this means columns having object datatypes
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Remove target column from categorical features
    categorical_features = [col for col in categorical_cols if col != 'member_casual']

    print("\nColumn type Summary")
    print(f"Number of Numerical Features: {len(numerical_features)}")
    print(f"Number of Categorical Features: {len(categorical_features)}")
    print(f"Target Column: member_casual")

    print("\nNumerical Features")
    print(numerical_features)

    print("\nCategorical Features")
    print(categorical_features)

    return numerical_features, categorical_features

"""## Analyzing Feature Distribution"""

def analyze_feature_distributions(df, numerical_features, categorical_features):
    # Analyzing the Feature Distribution

    # Analyzing distribution of numerical features
    if len(numerical_features) > 0:
        # Plot only the first 8 numerical features to avoid overcrowding which can impact the visualization
        viz_features = numerical_features[:8] if len(numerical_features) > 8 else numerical_features

        if len(viz_features) > 0:
            plt.figure(figsize=(20, 15))
            plt.xticks(fontsize=15, fontweight='bold')
            plt.yticks(fontsize=15, fontweight='bold')
            for i, feature in enumerate(viz_features, 1):
                plt.subplot(2, 4, i)
                plt.hist(df[feature].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                plt.title(f'Distribution of {feature}', fontsize=20, fontweight='bold')
                plt.xlabel(feature, fontsize=15, fontweight='bold')
                plt.ylabel('Frequency', fontsize=15, fontweight='bold')

            plt.tight_layout()
            plt.savefig(f'Distribution of {feature}.png', dpi=300, bbox_inches='tight')
            plt.show()

    # Analyzing the cardinality of categorical features
    # We should know that how many unique categories are present in each
    # categorical column this is called cardinality

    print("\nCategorical Feature Cardinality")
    categorical_cardinality = {}

    for col in categorical_features:
        num_unique = df[col].nunique()
        categorical_cardinality[col] = num_unique
        print(f"{col}: {num_unique} unique categories")

        # Show top categories for high cardinality features
        if num_unique > 10:
            print(f"  Top 5 categories: {df[col].value_counts().head().to_dict()}")

    # Visualizing cardinality of categorical features
    if len(categorical_features) > 0:
        plt.figure(figsize=(18, 10))
        plt.xticks(fontsize=15, fontweight='bold')
        plt.yticks(fontsize=15, fontweight='bold')
        plt.bar(categorical_cardinality.keys(), categorical_cardinality.values(),
                color='lightcoral', alpha=0.8)
        plt.title('Cardinality of Categorical Columns', fontsize=20, fontweight='bold')
        plt.xlabel('Categorical Features', fontsize=15, fontweight='bold')
        plt.ylabel('Number of Unique Categories', fontsize=15, fontweight='bold')
        plt.xticks(rotation=45, fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig('Cardinality of Categorical Columnsn.png', dpi=300, bbox_inches='tight')
        plt.show()

    return categorical_cardinality

"""## Correlation Analysis"""

def perform_correlation_analysis(df, numerical_features):
    # Correlation Analysis

    # Correlation is a statistical measure which computes the linear dependency between two random variables (features)

    if len(numerical_features) == 0:
        print("No numerical features available for correlation analysis")
        return None, None

    # Encoding target variable for correlation analysis
    le = LabelEncoder()
    df_temp = df.copy()
    df_temp['member_casual_encoded'] = le.fit_transform(df_temp['member_casual'])

    plt.figure(figsize=(18, 10))
    plt.xticks(fontsize=15, fontweight='bold')
    plt.yticks(fontsize=15, fontweight='bold')

    # Calculating correlation matrix including target column
    corr_features = numerical_features + ['member_casual_encoded']
    correlation_matrix = df_temp[corr_features].corr()

    # Creating heatmap to visualize the correlation
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8},
                annot_kws={"fontsize": 15, 'fontweight': 'bold'})

    plt.title('Correlation Matrix of Features', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Correlation Matrix of Features.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Showing features that are most correlated with target column
    target_correlations = correlation_matrix['member_casual_encoded'].abs().sort_values(ascending=False)
    print("\nFeatures most correlated with target column:")
    print(target_correlations.head(10))

    return correlation_matrix, target_correlations

"""# Data Preprocessing

## Filling missing values

There are different techniques to fill the missing values:


1.   For Numerical Features:


      *   Mean Imputation Technique: Filling missing values with the mean of observed values of the given variable (We will be using this).
      *   Random Imputation Technique: Filling missing values with randomly selecting observed values of the given variable.

      *   Deterministic Imputation Technique: Filling missing values by building Linear and Stochastic Models.


2.   For Categorical Features:
        * Mode Imputation Technique: Filling the missing values with the most occurring value of the given variable.
"""

def preprocess_data(df, numerical_features, categorical_features):

    # Filling missing values for numerical columns with mean value
    for col in numerical_features:
        if col in df.columns and df[col].isnull().sum() > 0:
            mean_val = df[col].mean()
            df[col].fillna(mean_val, inplace=True)

    # Filling missing values for categorical columns with mode value
    for col in categorical_features:
        if col in df.columns and df[col].isnull().sum() > 0:
            mode_vals = df[col].mode()
            mode_val = mode_vals[0] if len(mode_vals) > 0 else 'Unknown'
            df[col].fillna(mode_val, inplace=True)

    # Verifying that there are no missing values left
    total_missing = df.isnull().sum().sum()
    print(f"Total missing values after imputation: {total_missing}")

    # Separating features and target columns
    # axis = 1 is column wise
    X = df.drop('member_casual', axis=1)
    y = df['member_casual']

    print(f"Final feature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")

    return X, y

"""## Splitting Dataset into Training and Validation"""

def split_data(X, y):
    ## Splitting the dataset into training and validation sets

    # 80% -> Training Dataset
    # 20% -> Validation Dataset

    # Validation Dataset will be used for hyperparameter tuning of different models.


    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")

    return X_train, X_val, y_train, y_val

"""## Pre Preprocessor"""

def create_preprocessor(numerical_features, categorical_features):
    # Creating Preprocessor Pipeline

    # Preprocessor Pipeline is more robust and allows reuse easily

    # If you have to apply imputation or scaling or encoding then you just have to call the preprocessor


    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_features)
        ])

    # Feature Selection

    # For high-dimensional data we need to select most important features only. Otherwise model can become complex.

    # Use mutual information for feature selection as it works well with mixed data types
    feature_selector = SelectKBest(score_func=mutual_info_classif, k='all')  # Start with all features

    return preprocessor, feature_selector

"""## Evaluating Models"""

def evaluate_model(pipeline, X_val, y_val, model_name):
    # Helper Function

    # This function will be used to evaluate different models and to plot confusion matrix
    # member calss of the target variable is positive class

    y_pred = pipeline.predict(X_val)

    print(f"\n{model_name} Performance on the Validation Set:")
    print(f"Accuracy of the model: {accuracy_score(y_val, y_pred):.4f}")
    print(f"F1 Score of the model: {f1_score(y_val, y_pred, pos_label='member'):.4f}")

    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_val, y_pred))

    # Plotting Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(18, 10))
    plt.xticks(fontsize=15, fontweight='bold')
    plt.yticks(fontsize=15, fontweight='bold')

    # Get unique classes for labels
    classes = sorted(y_val.unique())
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                annot_kws={"fontsize": 20, 'fontweight': 'bold'})

    plt.title(f'Confusion Matrix - {model_name}', fontsize=20, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=15, fontweight='bold')
    plt.ylabel('True Label', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'Confusion Matrix - {model_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

    return y_pred, accuracy_score(y_val, y_pred), f1_score(y_val, y_pred, pos_label='member')

"""# Naive Bayes"""

def train_naive_bayes(X_train, X_val, y_train, y_val, X, y, preprocessor, feature_selector):
    # Naive Bayes

    # Since our dataset contains both numerical and categorical features. We will build Gaussian Naive Bayes Model.

    # Creating the pipeline for the naive bayes model
    nb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', feature_selector),
        ('classifier', GaussianNB())
    ])

    # Training the model on the training set
    nb_pipeline.fit(X_train, y_train)

    # Evaluating the model on the validation dataset
    y_pred_nb, acc_nb, f1_nb = evaluate_model(nb_pipeline, X_val, y_val, "Naive Bayes")

    # Cross-validation for robust evaluation of the model
    # here we have used k fold validation where k is 5
    cv_scores_nb = cross_val_score(nb_pipeline, X, y, cv=5, scoring='f1_macro')
    print(f"\nCross-Validation F1 Scores: {cv_scores_nb}")
    print(f"Mean CV F1 Score: {cv_scores_nb.mean():.4f} (+/- {cv_scores_nb.std() * 2:.4f})")

    return nb_pipeline, y_pred_nb, acc_nb, f1_nb, cv_scores_nb

"""# K-NN"""

def train_knn(X_train, X_val, y_train, y_val, X, y, preprocessor, feature_selector):
    # K-NN

    # Creating the pipeline for K-NN model
    knn_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', feature_selector),
        ('classifier', KNeighborsClassifier())
    ])

    # Training the model on the training set
    # here no hyperparameter tuning of the model is done
    # it is using the default value of k that is 5
    # thus later this model without hyperparameter tuning will be called
    # Default K-NN
    knn_pipeline.fit(X_train, y_train)

    # Evaluating the model
    y_pred_knn, acc_knn, f1_knn = evaluate_model(knn_pipeline, X_val, y_val, "K-NN")

    # Cross-validation for robust evaluation
    cv_scores_knn = cross_val_score(knn_pipeline, X, y, cv=5, scoring='f1_macro')
    print(f"\nCross-Validation F1 Scores for k-NN: {cv_scores_knn}")
    print(f"Mean CV F1 Score: {cv_scores_knn.mean():.4f} (+/- {cv_scores_knn.std() * 2:.4f})")

    # Hyperparameter Tuning of K-NN

    # Accuracy of K-NN is greatly affected by the initialization of the K which
    # represents the number of neighbours have to be considered
    # while classifying the data point.

    # Define the range of K values and other parameters
    param_grid_knn = {
        'classifier__n_neighbors': [5],
        'classifier__metric': ['euclidean']
    }

    # Creating a GridSearchCV object
    # Create a GridSearchCV object, this means create different combinations of
    # values of parameters defined above and check the model accuarcy for each
    # we will use f-1 score as deciding factor
    knn_grid = GridSearchCV(
        knn_pipeline,
        param_grid_knn,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )

    # Fit the grid search to the training data
    # This is the K-NN model after hyperparameter tuning with optimal values of
    # parameters, thus this will be called Best K-NN
    print("Starting K-NN hyperparameter tuning....")
    knn_grid.fit(X_train, y_train)

    # Get the best parameters and best score
    best_params_knn = knn_grid.best_params_
    best_score_knn = knn_grid.best_score_

    print(f"\nBest K-NN parameters: {best_params_knn}")
    print(f"Best cross-validation F1 score: {best_score_knn:.4f}")

    # Evaluating the best model on the validation set
    best_knn = knn_grid.best_estimator_
    y_pred_best_knn, acc_best_knn, f1_best_knn = evaluate_model(best_knn, X_val, y_val, "Best K-NN")

    return knn_pipeline, best_knn, best_params_knn, best_score_knn, y_pred_knn, acc_knn, f1_knn, cv_scores_knn, y_pred_best_knn, acc_best_knn, f1_best_knn, knn_grid

"""# Decision Tree

## Hyperparameters of Decision Tree

1.   **max_depth**: The biggest problem of the decision tree is overfitting. To prevent this. We used hyperparameter max_depth which defines the longest path from the root node to a leaf node in the tree.

2.   **min_samples_split:** To decide whether to split the node futher or not. We use min_samples_split hyperparameter which specifies the minimum number of samples requires to split the node further, if node have samples lower than the minimum number of sample then it becomes the leaf node. Helps to prevent overfitting.


3.   **min_samples_leaf:** specifies the minimum number of samples required to be present in a leaf node. Helps in preventing noise.

4.   **criterion:** specifies the function that will be used to measure the quality of split.

5.   **class_weight:** useful in those datasets where the given data is imbalanced like ours. This allows assigning different weights to different classes
"""

def train_decision_tree(X_train, X_val, y_train, y_val, X, y, preprocessor, feature_selector):
    # Decision Tree

    # Creating the pipeline for Decision Tree
    dt_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', feature_selector),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])

    # Training the model on the training set
    # Training the model on the training set by taking the default values of
    # hyperparameter
    # default values of hyperparameters are:
    # max_depth = None, min_samples_split = 2, min_samples_leaf = 1, criterion = gini,
    # class_weight = None
    dt_pipeline.fit(X_train, y_train)

    # Evaluating the model
    y_pred_dt, acc_dt, f1_dt = evaluate_model(dt_pipeline, X_val, y_val, "Decision Tree")

    # Cross-validation for robust evaluation
    cv_scores_dt = cross_val_score(dt_pipeline, X, y, cv=5, scoring='f1_macro')
    print(f"\nCross-Validation F1 Scores for Decision Tree: {cv_scores_dt}")
    print(f"Mean CV F1 Score: {cv_scores_dt.mean():.4f} (+/- {cv_scores_dt.std() * 2:.4f})")

    # Hyperparameter Tuning of Decision Tree

    param_grid_dt = {
        'classifier__max_depth': [7],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__criterion': ['entropy'],
        'classifier__class_weight': ['balanced']
    }

    # Creating a GridSearchCV object for the Decision Tree
    dt_grid = GridSearchCV(
        dt_pipeline,
        param_grid_dt,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )

    # Fit the grid search to the training data
    print("Starting Decision Tree hyperparameter tuning....")
    dt_grid.fit(X_train, y_train)

    # Get the best parameters and best score
    best_params_dt = dt_grid.best_params_
    best_score_dt = dt_grid.best_score_

    print(f"\nBest parameters for Decision Tree: {best_params_dt}")
    print(f"Best cross-validation F1 score: {best_score_dt:.4f}")

    # Evaluating the best model on the validation set
    best_dt = dt_grid.best_estimator_
    y_pred_best_dt, acc_best_dt, f1_best_dt = evaluate_model(best_dt, X_val, y_val, "Best Decision Tree")

    # Getting those features which are important for the best model
    # Decision Trees are expensive to compute
    # Thus it becomes necessary to determine the important features
    # Getting feature importances from the best model

    # For the best Decision model get feature importance
    # feature_importances_: it provides a measure of how important each and every feature
    # is in making classification with the given trained model.
    if hasattr(best_dt.named_steps['classifier'], 'feature_importances_'):
        feature_importances_dt = best_dt.named_steps['classifier'].feature_importances_
        selected_features = best_dt.named_steps['feature_selection'].get_support()
        all_feature_names = best_dt.named_steps['preprocessor'].get_feature_names_out()
        selected_feature_names = all_feature_names[selected_features]

        # Creating a DataFrame for features that are found to be important
        importance_df_dt = pd.DataFrame({
            'feature': selected_feature_names,
            'importance': feature_importances_dt
        }).sort_values('importance', ascending=False)

        print("\nTop 15 most important features for the Decision Tree Model:")
        print(importance_df_dt.head(15))

        # Visualizing feature importance
        plt.figure(figsize=(18, 10))
        plt.xticks(fontsize=15, fontweight='bold')
        plt.yticks(fontsize=15, fontweight='bold')
        top_features = importance_df_dt.head(15)
        plt.barh(range(len(top_features)), top_features['importance'], color='lightgreen')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.title('Top 15 Important Features for Decision Tree', fontsize=20, fontweight='bold')
        plt.xlabel('Importance', fontsize=15, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('Top 15 Important Features for Decision Tree', dpi=300, bbox_inches='tight')
        plt.show()

    return dt_pipeline, best_dt, best_params_dt, best_score_dt, y_pred_dt, acc_dt, f1_dt, cv_scores_dt, y_pred_best_dt, acc_best_dt, f1_best_dt, dt_grid

"""# Random Forest"""

def train_random_forest(X_train, X_val, y_train, y_val, X, y, preprocessor, feature_selector):
    """# Random Forest"""

    # Creating a pipeline for the Random Forest
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', feature_selector),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    # Training the model on the training set
    rf_pipeline.fit(X_train, y_train)

    # Evaluating the model
    y_pred_rf, acc_rf, f1_rf = evaluate_model(rf_pipeline, X_val, y_val, "Random Forest")

    # Cross-validation for robust evaluation
    cv_scores_rf = cross_val_score(rf_pipeline, X, y, cv=5, scoring='f1_macro')
    print(f"\nCross-Validation F1 Scores for Random Forest: {cv_scores_rf}")
    print(f"Mean CV F1 Score: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std() * 2:.4f})")

    # Hyperparameter Tuning of Random Forest

    # Defining the parameter grid for Random Forest
    param_grid_rf = {
        'classifier__n_estimators': [100],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [4],
        'classifier__max_features': ['sqrt'],
        'classifier__bootstrap': [True],
        'classifier__class_weight': ['balanced']
    }

    # Creating a GridSearchCV object for Random Forest
    rf_grid = GridSearchCV(
        rf_pipeline,
        param_grid_rf,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )

    # Fiting the grid search to the training data
    print("Starting Random Forest hyperparameter tuning...")
    rf_grid.fit(X_train, y_train)

    # Get the best parameters and best score
    best_params_rf = rf_grid.best_params_
    best_score_rf = rf_grid.best_score_

    print(f"\nBest parameters for Random Forest: {best_params_rf}")
    print(f"Best cross-validation F1 score: {best_score_rf:.4f}")

    # Evaluating the best model on the validation set
    best_rf = rf_grid.best_estimator_
    y_pred_best_rf, acc_best_rf, f1_best_rf = evaluate_model(best_rf, X_val, y_val, "Best Random Forest")

    # Get features that are important for the best Random Forest model
    if hasattr(best_rf.named_steps['classifier'], 'feature_importances_'):
        feature_importances_rf = best_rf.named_steps['classifier'].feature_importances_
        selected_features_rf = best_rf.named_steps['feature_selection'].get_support()
        all_feature_names_rf = best_rf.named_steps['preprocessor'].get_feature_names_out()
        selected_feature_names_rf = all_feature_names_rf[selected_features_rf]

        # Creating a DataFrame for features that are important
        importance_df_rf = pd.DataFrame({
            'feature': selected_feature_names_rf,
            'importance': feature_importances_rf
        }).sort_values('importance', ascending=False)

        print("\nTop 15 most important features for Random Forest:")
        print(importance_df_rf.head(15))

        # Visualizing Random Forest feature importance
        plt.figure(figsize=(18, 10))
        plt.xticks(fontsize=15, fontweight='bold')
        plt.yticks(fontsize=15, fontweight='bold')
        top_features_rf = importance_df_rf.head(15)
        plt.barh(range(len(top_features_rf)), top_features_rf['importance'], color='forestgreen')
        plt.yticks(range(len(top_features_rf)), top_features_rf['feature'])
        plt.title('Top 15 Important Features for Random Forest', fontsize=20, fontweight='bold')
        plt.xlabel('Importance', fontsize=15, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('Top 15 Important Features for Random Forest.png', dpi=300, bbox_inches='tight')
        plt.show()

    return rf_pipeline, best_rf, best_params_rf, best_score_rf, y_pred_rf, acc_rf, f1_rf, cv_scores_rf, y_pred_best_rf, acc_best_rf, f1_best_rf, rf_grid

"""# Model Comparison and Statistical Analysis

## Model Comparison

Now we will compare all the models that we have built so far - Naive Bayes, K-NN, Decision Tree and Random Forest
"""

def perform_model_comparison(acc_nb, f1_nb, cv_scores_nb, acc_knn, f1_knn, cv_scores_knn,
                           acc_best_knn, f1_best_knn, best_score_knn, knn_grid,
                           acc_dt, f1_dt, cv_scores_dt, acc_best_dt, f1_best_dt, best_score_dt, dt_grid,
                           acc_rf, f1_rf, cv_scores_rf, acc_best_rf, f1_best_rf, best_score_rf, rf_grid):

    # Storing all model results for comparison
    model_results = {
        'Model': ['Naive Bayes', 'K-NN (default)', 'Best K-NN', 'Decision Tree (default)',
                  'Best Decision Tree', 'Random Forest (default)', 'Best Random Forest'],
        'Validation Accuracy': [acc_nb, acc_knn, acc_best_knn, acc_dt, acc_best_dt, acc_rf, acc_best_rf],
        'Validation F1': [f1_nb, f1_knn, f1_best_knn, f1_dt, f1_best_dt, f1_rf, f1_best_rf],
        'CV F1 Mean': [cv_scores_nb.mean(), cv_scores_knn.mean(), best_score_knn,
                       cv_scores_dt.mean(), best_score_dt, cv_scores_rf.mean(), best_score_rf],
        'CV F1 Std': [cv_scores_nb.std(), cv_scores_knn.std(), knn_grid.cv_results_['std_test_score'][knn_grid.best_index_],
                      cv_scores_dt.std(), dt_grid.cv_results_['std_test_score'][dt_grid.best_index_],
                      cv_scores_rf.std(), rf_grid.cv_results_['std_test_score'][rf_grid.best_index_]]
    }

    # Creating DataFrame for comparison
    comparison_df = pd.DataFrame(model_results)
    print("\nModel Performance Comparison:")
    print(comparison_df.round(4))

    # Visualizing comparison
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))

    # Font settings you can tweak
    title_fs = 20
    label_fs = 15
    tick_fs = 12
    title_fw = 'bold'
    label_fw = 'bold'
    tick_fw = 'bold'

    # F1 Score comparison
    axes[0].bar(comparison_df['Model'], comparison_df['Validation F1'],
                color=['skyblue', 'lightcoral', 'red', 'lightgreen', 'green', 'orange', 'darkorange'],
                alpha=0.8)
    axes[0].set_title('Model F1 Score Comparison', fontsize=title_fs, fontweight=title_fw)
    axes[0].set_ylabel('F1 Score', fontsize=label_fs, fontweight=label_fw)

    # tick sizes
    axes[0].tick_params(axis='x', rotation=45, labelsize=tick_fs)
    axes[0].tick_params(axis='y', labelsize=tick_fs)

    # make tick labels bold (tick_params doesn't support fontweight)
    for tick in axes[0].get_xticklabels():
        tick.set_fontweight(tick_fw)
    for tick in axes[0].get_yticklabels():
        tick.set_fontweight(tick_fw)

    # Accuracy comparison
    axes[1].bar(comparison_df['Model'], comparison_df['Validation Accuracy'],
                color=['skyblue', 'lightcoral', 'red', 'lightgreen', 'green', 'orange', 'darkorange'],
                alpha=0.8)
    axes[1].set_title('Model Accuracy Comparison', fontsize=title_fs, fontweight=title_fw)
    axes[1].set_ylabel('Accuracy', fontsize=label_fs, fontweight=label_fw)

    axes[1].tick_params(axis='x', rotation=45, labelsize=tick_fs)
    axes[1].tick_params(axis='y', labelsize=tick_fs)

    for tick in axes[1].get_xticklabels():
        tick.set_fontweight(tick_fw)
    for tick in axes[1].get_yticklabels():
        tick.set_fontweight(tick_fw)

    plt.tight_layout()
    plt.show()

    return comparison_df

"""## Statistical Significance Testing"""

def perform_statistical_testing(best_knn, best_dt, best_rf, X, y):

    # Performing statistical significance tests between best models
    best_models = {
        'Best k-NN': best_knn,
        'Best Decision Tree': best_dt,
        'Best Random Forest': best_rf
    }

    # Cross-validation scores for statistical testing
    cv_results_for_testing = {}
    for name, model in best_models.items():
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
        cv_results_for_testing[name] = cv_scores

    # Statistical comparison between different best models
    print("\nStatistical Significance Testing (Paired t-test):")
    model_names = list(cv_results_for_testing.keys())

    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            model1, model2 = model_names[i], model_names[j]
            scores1 = cv_results_for_testing[model1]
            scores2 = cv_results_for_testing[model2]

            # Paired t-test
            t_stat, p_value = ttest_rel(scores1, scores2)

            print(f"{model1} vs {model2}:")
            print(f"  Mean F1: {scores1.mean():.4f} vs {scores2.mean():.4f}")
            print(f"  t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
            print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
            print()

    return cv_results_for_testing

"""# Ensemble Models

Ensemble technique refers to combining different models in order to perform the classification task
"""

def create_ensemble_models(X_train, X_val, y_train, y_val, X, y, preprocessor, feature_selector,
                          best_params_knn, best_params_dt, best_params_rf,
                          cv_scores_nb, best_score_knn, best_score_dt, best_score_rf):

    # Voting Classifier with All models

    # Defining individual models with best parameters
    nb_final = GaussianNB()

    knn_final = KNeighborsClassifier(**{k.replace('classifier__', ''): v
                                       for k, v in best_params_knn.items()})

    dt_final = DecisionTreeClassifier(**{k.replace('classifier__', ''): v
                                        for k, v in best_params_dt.items()},
                                      random_state=42)

    rf_final = RandomForestClassifier(**{k.replace('classifier__', ''): v
                                        for k, v in best_params_rf.items()},
                                      random_state=42, n_jobs=-1)

    # Creating the voting classifier with equal weights
    voting_clf_equal = VotingClassifier(
        estimators=[
            ('nb', nb_final),
            ('knn', knn_final),
            ('dt', dt_final),
            ('rf', rf_final)
        ],
        voting='soft'
    )

    voting_equal_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', feature_selector),
        ('classifier', voting_clf_equal)
    ])

    # Training and evaluating the ensemble model
    voting_equal_pipeline.fit(X_train, y_train)
    y_pred_voting_equal, acc_voting_equal, f1_voting_equal = evaluate_model(
        voting_equal_pipeline, X_val, y_val, "Voting Classifier (Equal Weights)")

    cv_scores_voting_equal = cross_val_score(voting_equal_pipeline, X, y, cv=5, scoring='f1_macro')
    print(f"Cross-Validation F1 Scores for Equal Voting: {cv_scores_voting_equal}")
    print(f"Mean CV F1 Score: {cv_scores_voting_equal.mean():.4f} (+/- {cv_scores_voting_equal.std() * 2:.4f})")

    # Weighted Voting Classifier

    # Assign weights to different models based on their CV F1 scores
    weights = np.array([cv_scores_nb.mean(), best_score_knn, best_score_dt, best_score_rf])
    normalized_weights = weights / weights.sum()

    print(f"Performance based weights:")
    print(f"NB: {normalized_weights[0]:.3f}, k-NN: {normalized_weights[1]:.3f}")
    print(f"DT: {normalized_weights[2]:.3f}, RF: {normalized_weights[3]:.3f}")

    # Creating weighted voting classifier
    voting_clf_weighted = VotingClassifier(
        estimators=[
            ('nb', nb_final),
            ('knn', knn_final),
            ('dt', dt_final),
            ('rf', rf_final)
        ],
        voting='soft',
        weights=normalized_weights
    )

    voting_weighted_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', feature_selector),
        ('classifier', voting_clf_weighted)
    ])

    voting_weighted_pipeline.fit(X_train, y_train)
    y_pred_voting_weighted, acc_voting_weighted, f1_voting_weighted = evaluate_model(
        voting_weighted_pipeline, X_val, y_val, "Weighted Voting Classifier")

    cv_scores_voting_weighted = cross_val_score(voting_weighted_pipeline, X, y, cv=5, scoring='f1_macro')
    print(f"Cross-Validation F1 Scores for Weighted Voting: {cv_scores_voting_weighted}")
    print(f"Mean CV F1 Score: {cv_scores_voting_weighted.mean():.4f} (+/- {cv_scores_voting_weighted.std() * 2:.4f})")

    # Best Models only Ensemble Model

    # Weighted Ensemble Model with only Decision Tree and Random Forest
    voting_clf_best = VotingClassifier(
        estimators=[
            ('dt', dt_final),
            ('rf', rf_final)
        ],
        voting='soft',
        weights=[normalized_weights[2], normalized_weights[3]]
    )

    voting_best_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', feature_selector),
        ('classifier', voting_clf_best)
    ])

    voting_best_pipeline.fit(X_train, y_train)
    y_pred_voting_best, acc_voting_best, f1_voting_best = evaluate_model(
        voting_best_pipeline, X_val, y_val, "Best Models Voting (DT + RF)")

    cv_scores_voting_best = cross_val_score(voting_best_pipeline, X, y, cv=5, scoring='f1_macro')
    print(f"Cross-Validation F1 Scores for DT+RF Ensemble: {cv_scores_voting_best}")
    print(f"Mean CV F1 Score: {cv_scores_voting_best.mean():.4f} (+/- {cv_scores_voting_best.std() * 2:.4f})")

    return (voting_equal_pipeline, voting_weighted_pipeline, voting_best_pipeline,
            y_pred_voting_equal, acc_voting_equal, f1_voting_equal, cv_scores_voting_equal,
            y_pred_voting_weighted, acc_voting_weighted, f1_voting_weighted, cv_scores_voting_weighted,
            y_pred_voting_best, acc_voting_best, f1_voting_best, cv_scores_voting_best)

"""# Final Model Selection"""

# --- imports (make sure these are present once in your module) ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate


def final_model_selection_and_testing(
    acc_nb, f1_nb, cv_scores_nb,
    acc_best_knn, f1_best_knn, best_score_knn, knn_grid,
    acc_best_dt, f1_best_dt, best_score_dt, dt_grid,
    acc_best_rf, f1_best_rf, best_score_rf, rf_grid,
    acc_voting_equal, f1_voting_equal, cv_scores_voting_equal,
    acc_voting_weighted, f1_voting_weighted, cv_scores_voting_weighted,
    acc_voting_best, f1_voting_best, cv_scores_voting_best,
    best_knn, best_dt, best_rf, voting_equal_pipeline, voting_weighted_pipeline, voting_best_pipeline,
    X, y, df, missing_summary
):
    """
    Compares individual and ensemble models, selects the best by CV F1,
    retrains on full data, evaluates with 5-fold CV, and (if available) plots feature importances.

    Returns:
        final_model, best_model_name, mean_accuracy, mean_f1
    """

    # -------------------------
    # Model Comparison Table
    # -------------------------
    final_results = {
        'Model': [
            'Naive Bayes',
            'Best k-NN',
            'Best Decision Tree',
            'Best Random Forest',
            'Voting (Equal)',
            'Voting (Weighted)',
            'Voting (Best Models)'
        ],
        'Validation Accuracy': [
            acc_nb, acc_best_knn, acc_best_dt, acc_best_rf,
            acc_voting_equal, acc_voting_weighted, acc_voting_best
        ],
        'Validation F1': [
            f1_nb, f1_best_knn, f1_best_dt, f1_best_rf,
            f1_voting_equal, f1_voting_weighted, f1_voting_best
        ],
        'CV F1 Mean': [
            np.mean(cv_scores_nb), best_score_knn, best_score_dt, best_score_rf,
            np.mean(cv_scores_voting_equal), np.mean(cv_scores_voting_weighted), np.mean(cv_scores_voting_best)
        ],
        'CV F1 Std': [
            np.std(cv_scores_nb),
            knn_grid.cv_results_['std_test_score'][knn_grid.best_index_],
            dt_grid.cv_results_['std_test_score'][dt_grid.best_index_],
            rf_grid.cv_results_['std_test_score'][rf_grid.best_index_],
            np.std(cv_scores_voting_equal), np.std(cv_scores_voting_weighted), np.std(cv_scores_voting_best)
        ]
    }

    final_comparison_df = pd.DataFrame(final_results)
    print("Final Model Performance Comparison:")
    print(final_comparison_df.round(4))

    # -------------------------
    # Best Model by CV F1
    # -------------------------
    best_model_idx = final_comparison_df['CV F1 Mean'].idxmax()
    best_model_name = final_comparison_df.iloc[best_model_idx]['Model']
    best_f1_score = final_comparison_df.iloc[best_model_idx]['CV F1 Mean']

    print(f"\nBest Performing Model: {best_model_name}")
    print(f"Best CV F1 Score: {best_f1_score:.4f}")

    # -------------------------
    # Visualizations
    # -------------------------
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # (A) CV F1 bar chart
    axes[0, 0].bar(
        final_comparison_df['Model'],
        final_comparison_df['CV F1 Mean'],
        color=plt.cm.viridis(np.linspace(0, 1, len(final_comparison_df))),
        alpha=0.8
    )
    axes[0, 0].set_title('Cross-Validation F1 Score Comparison', fontsize=15, fontweight='bold')
    axes[0, 0].set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    axes[0, 0].tick_params(axis='x', labelrotation=45, labelsize=12)
    for lbl in axes[0, 0].get_xticklabels():
        lbl.set_fontweight('bold')
    axes[0, 0].grid(True, alpha=0.3)

    # (B) Validation Accuracy bar chart
    axes[0, 1].bar(
        final_comparison_df['Model'],
        final_comparison_df['Validation Accuracy'],
        color=plt.cm.plasma(np.linspace(0, 1, len(final_comparison_df))),
        alpha=0.8
    )
    axes[0, 1].set_title('Validation Accuracy Comparison', fontsize=15, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].tick_params(axis='x', labelrotation=45, labelsize=12)
    for lbl in axes[0, 1].get_xticklabels():
        lbl.set_fontweight('bold')
    axes[0, 1].grid(True, alpha=0.3)

    # (C) F1 vs Accuracy scatter
    axes[1, 0].scatter(
        final_comparison_df['Validation Accuracy'],
        final_comparison_df['Validation F1'],
        s=200, alpha=0.7, c=range(len(final_comparison_df)), cmap='tab10'
    )
    for i, model in enumerate(final_comparison_df['Model']):
        axes[1, 0].annotate(
            model,
            (final_comparison_df.iloc[i]['Validation Accuracy'], final_comparison_df.iloc[i]['Validation F1']),
            xytext=(5, 5), textcoords='offset points', fontsize=12, fontweight='bold'
        )
    axes[1, 0].set_title('F1 Score vs Accuracy', fontsize=15, fontweight='bold')
    axes[1, 0].set_xlabel('Validation Accuracy', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Validation F1 Score', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # (D) Model complexity vs performance (relative complexity you provided)
    complexity_scores = [1, 3, 5, 7, 4, 6, 8]  # Naive Bayes -> Voting(Best)
    axes[1, 1].scatter(
        complexity_scores,
        final_comparison_df['CV F1 Mean'],
        s=300, alpha=0.7, c=range(len(final_comparison_df)), cmap='tab10'
    )
    for i, model in enumerate(final_comparison_df['Model']):
        axes[1, 1].annotate(
            model,
            (complexity_scores[i], final_comparison_df.iloc[i]['CV F1 Mean']),
            xytext=(5, 5), textcoords='offset points', fontsize=12, fontweight='bold'
        )
    axes[1, 1].set_title('Model Complexity vs Performance', fontsize=15, fontweight='bold')
    axes[1, 1].set_xlabel('Model Complexity (Relative)', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('CV F1 Score', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('Final Model Comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # -------------------------
    # Final Model Selection
    # -------------------------
    if best_model_name == 'Best Random Forest':
        final_model = best_rf
    elif best_model_name == 'Best Decision Tree':
        final_model = best_dt
    elif best_model_name == 'Best k-NN':
        final_model = best_knn
    elif best_model_name == 'Voting (Weighted)':
        final_model = voting_weighted_pipeline
    elif best_model_name == 'Voting (Best Models)':
        final_model = voting_best_pipeline
    else:
        final_model = voting_equal_pipeline

    print(f"\nSelected Final Model: {best_model_name}")

    # -------------------------
    # Retrain on full dataset
    # -------------------------
    print("Training final model on entire dataset...")
    final_model.fit(X, y)

    # -------------------------
    # Final 5-fold CV Evaluation
    # -------------------------
    print(f"\nDataset Shape: {df.shape}")

    final_cv_results = cross_validate(
        final_model, X, y, cv=5,
        scoring=['accuracy', 'f1_macro']
    )

    mean_accuracy = final_cv_results['test_accuracy'].mean()
    mean_f1 = final_cv_results['test_f1_macro'].mean()
    std_accuracy = final_cv_results['test_accuracy'].std()
    std_f1 = final_cv_results['test_f1_macro'].std()

    print("\nFinal Model Performance (5-Fold CV):")
    print(f"Mean Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy * 2:.4f})")
    print(f"Mean F1 Score: {mean_f1:.4f} (+/- {std_f1 * 2:.4f})")

    # -------------------------
    # Feature Importance (if available)
    # -------------------------
    def _get_classifier_from_pipeline(pipeline):
        """Try to locate the classifier step from a sklearn Pipeline."""
        if hasattr(pipeline, 'named_steps'):
            # Prefer a step named 'classifier' if present
            if 'classifier' in pipeline.named_steps:
                return pipeline.named_steps['classifier']
            # Otherwise, fall back to the last step (common pattern)
            # or any step with fit/predict attrs
            for name in reversed(list(pipeline.named_steps.keys())):
                step = pipeline.named_steps[name]
                if hasattr(step, 'fit') and hasattr(step, 'predict'):
                    return step
        return pipeline  # if not a pipeline, return the estimator itself

    clf = _get_classifier_from_pipeline(final_model)

    if hasattr(clf, 'feature_importances_'):
        # Try to fetch selected feature names if pipeline has those steps
        try:
            feature_selector = final_model.named_steps.get('feature_selection', None)
            preprocessor = final_model.named_steps.get('preprocessor', None)

            if preprocessor is not None and feature_selector is not None:
                selected_mask = feature_selector.get_support()
                all_feature_names = preprocessor.get_feature_names_out()
                selected_feature_names = all_feature_names[selected_mask]
            else:
                # Fallback: generic names if we cannot resolve feature names
                selected_feature_names = np.array([f'feature_{i}' for i in range(len(clf.feature_importances_))])

        except Exception:
            selected_feature_names = np.array([f'feature_{i}' for i in range(len(clf.feature_importances_))])

        importance_df_final = pd.DataFrame({
            'feature': selected_feature_names,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\nTop 20 Most Important Features in the Final Model ({best_model_name}):")
        print(importance_df_final.head(20))

        # Plot top features
        plt.figure(figsize=(18, 10))
        plt.xticks(fontsize=15, fontweight='bold')
        plt.yticks(fontsize=15, fontweight='bold')
        top_features_final = importance_df_final.head(20)
        plt.barh(
            range(len(top_features_final)),
            top_features_final['importance'],
            color='purple', alpha=0.7
        )
        plt.yticks(range(len(top_features_final)), top_features_final['feature'])
        plt.title(f'Top 20 Important Features - Final Model ({best_model_name})',
                  fontsize=20, fontweight='bold')
        plt.xlabel('Importance', fontsize=15, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'Top 20 Important Features - Final Model ({best_model_name}).png',
                    dpi=300, bbox_inches='tight')
        plt.show()

    elif hasattr(clf, 'estimators_'):
        print(f"\nFinal model ({best_model_name}) is an ensemble - individual feature importances shown above/by base learners (if exposed).")

    # -------------------------
    # Results Summary
    # -------------------------
    print("\nFinal Results Summary")
    print(f"Dataset: {df.shape[0]} samples, {df.shape[1]-1} original features")
    try:
        class_dist = dict(df['member_casual'].value_counts())
        print(f"Class distribution: {class_dist}")
    except Exception:
        pass
    try:
        handled_cols = missing_summary.shape[0] if len(missing_summary) > 0 else 0
        print(f"Missing values handled: {handled_cols} columns had missing data")
    except Exception:
        pass

    print(f"\nBest Model: {best_model_name}")
    print("Final Cross-Validation Results:")
    print(f"  - Accuracy: {mean_accuracy:.4f} ± {std_accuracy*2:.4f}")
    print(f"  - F1 Score: {mean_f1:.4f} ± {std_f1*2:.4f}")

    # NOTE: Return 4 items to match your current caller unpack
    return final_model, best_model_name, mean_accuracy, mean_f1

"""# Main Function"""

def main():
    # Main function to execute the entire bike sharing classification pipeline

    print("\n Bike Sharing Member vs Casual Classification")

    # Step 1: Data Loading and Exploration
    print("\nStep 1: Data Loading and Exploration")
    df, target_counts = load_and_explore_data()

    # Step 2: Feature Engineering
    print("\nStep 2: Feature Engineering")
    df = improved_feature_engineering_corrected(df)

    # Step 3: Data Leakage Check
    print("\nStep 3: Data Leakage Validation")
    check_for_data_leakage(df)

    # Step 4: Missing Value Analysis
    print("\nStep 4: Missing Value Analysis")
    missing_summary = analyze_missing_values(df)

    # Step 5: Feature Type Identification
    print("\nStep 5: Feature Type Identification")
    numerical_features, categorical_features = identify_feature_types(df)

    # Step 6: Feature Distribution Analysis
    print("\nStep 6: Feature Distribution Analysis")
    categorical_cardinality = analyze_feature_distributions(df, numerical_features, categorical_features)

    # Step 7: Correlation Analysis
    print("\nStep 7: Correlation Analysis")
    correlation_matrix, target_correlations = perform_correlation_analysis(df, numerical_features)

    # Step 8: Data Preprocessing
    print("\nStep 8: Data Preprocessing")
    X, y = preprocess_data(df, numerical_features, categorical_features)

    # Step 9: Data Splitting
    print("\nStep 9: Data Splitting")
    X_train, X_val, y_train, y_val = split_data(X, y)

    # Step 10: Pipeline Creation
    print("\nStep 10: Pipeline Creation")
    print("="*50)
    preprocessor, feature_selector = create_preprocessor(numerical_features, categorical_features)

    # Step 11: Naive Bayes Model
    print("\nStep 11: Naive Bayes Model")
    nb_pipeline, y_pred_nb, acc_nb, f1_nb, cv_scores_nb = train_naive_bayes(
        X_train, X_val, y_train, y_val, X, y, preprocessor, feature_selector)

    # Step 12: K-NN Model
    print("\nStep 12: K-NN Model")
    (knn_pipeline, best_knn, best_params_knn, best_score_knn, y_pred_knn, acc_knn, f1_knn,
     cv_scores_knn, y_pred_best_knn, acc_best_knn, f1_best_knn, knn_grid) = train_knn(
        X_train, X_val, y_train, y_val, X, y, preprocessor, feature_selector)

    # Step 13: Decision Tree Model
    print("\nStep 13: Decision Tree Model")
    (dt_pipeline, best_dt, best_params_dt, best_score_dt, y_pred_dt, acc_dt, f1_dt,
     cv_scores_dt, y_pred_best_dt, acc_best_dt, f1_best_dt, dt_grid) = train_decision_tree(
        X_train, X_val, y_train, y_val, X, y, preprocessor, feature_selector)

    # Step 14: Random Forest Model
    print("\nStep 14: Random Forest Model")
    (rf_pipeline, best_rf, best_params_rf, best_score_rf, y_pred_rf, acc_rf, f1_rf,
     cv_scores_rf, y_pred_best_rf, acc_best_rf, f1_best_rf, rf_grid) = train_random_forest(
        X_train, X_val, y_train, y_val, X, y, preprocessor, feature_selector)

    # Step 15: Model Comparison
    print("\nStep 15: Model Comparison")
    comparison_df = perform_model_comparison(
        acc_nb, f1_nb, cv_scores_nb, acc_knn, f1_knn, cv_scores_knn,
        acc_best_knn, f1_best_knn, best_score_knn, knn_grid,
        acc_dt, f1_dt, cv_scores_dt, acc_best_dt, f1_best_dt, best_score_dt, dt_grid,
        acc_rf, f1_rf, cv_scores_rf, acc_best_rf, f1_best_rf, best_score_rf, rf_grid)

    # Step 16: Statistical Significance Testing
    print("\nStep 16: Statistical Significance Testing")
    cv_results_for_testing = perform_statistical_testing(best_knn, best_dt, best_rf, X, y)

    # Step 17: Ensemble Methods
    print("\nStep 17: Ensemble Methods")
    (voting_equal_pipeline, voting_weighted_pipeline, voting_best_pipeline,
     y_pred_voting_equal, acc_voting_equal, f1_voting_equal, cv_scores_voting_equal,
     y_pred_voting_weighted, acc_voting_weighted, f1_voting_weighted, cv_scores_voting_weighted,
     y_pred_voting_best, acc_voting_best, f1_voting_best, cv_scores_voting_best) = create_ensemble_models(
        X_train, X_val, y_train, y_val, X, y, preprocessor, feature_selector,
        best_params_knn, best_params_dt, best_params_rf,
        cv_scores_nb, best_score_knn, best_score_dt, best_score_rf)

    # Step 18: Final Model Selection and Testing
    print("\nStep 18: Final Model Selection and Testing")
    final_model, best_model_name, mean_accuracy, mean_f1, final_comparison_df = final_model_selection_and_testing(
        acc_nb, f1_nb, cv_scores_nb, acc_best_knn, f1_best_knn, best_score_knn, knn_grid,
        acc_best_dt, f1_best_dt, best_score_dt, dt_grid, acc_best_rf, f1_best_rf, best_score_rf, rf_grid,
        acc_voting_equal, f1_voting_equal, cv_scores_voting_equal,
        acc_voting_weighted, f1_voting_weighted, cv_scores_voting_weighted,
        acc_voting_best, f1_voting_best, cv_scores_voting_best,
        best_knn, best_dt, best_rf, voting_equal_pipeline, voting_weighted_pipeline, voting_best_pipeline,
        X, y, df, missing_summary)


    print("\n Execution Completed Successfully")
    print(f"Final selected model: {best_model_name}")
    print(f"Expected performance: F1 = {mean_f1:.4f}, Accuracy = {mean_accuracy:.4f}")

    return final_model, best_model_name, mean_accuracy, mean_f1

# Main execution
if __name__ == "__main__":
    # Run the complete pipeline
    final_model, best_model_name, mean_accuracy, mean_f1 = main()
    print("\n")
    print(f"- Final model: {best_model_name}")
    print(f"- Performance: {mean_f1:.1%} F1 score, {mean_accuracy:.1%} accuracy")
