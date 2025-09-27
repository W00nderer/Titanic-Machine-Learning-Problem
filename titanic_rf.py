from sklearn.ensemble import RandomForestClassifier # Using Random Forest model
from scipy.stats import randint # For randomized numbers
from sklearn.preprocessing import OneHotEncoder # One Hot Encoding of cateegorical data
from sklearn.model_selection import RandomizedSearchCV, cross_val_score # Randomized Search of the best hyper parameters, cross validation scores to check mean, std
import pandas as pd # Dataframe
import time # Recording hyper parameter tuning time


train_df = pd.read_csv("train.csv") # Import the training data

# Separate the data from the answer (who survived)
X_train = train_df.drop('Survived', axis='columns') 
y_train = train_df['Survived']

X_train['Age'] = X_train['Age'].fillna(X_train['Age'].median()) # Filling in missing data with mean imputation
bin_edges = [0,20,40,60,80] # Bins to categorize the ages
X_train['Age'] = pd.cut(X_train['Age'], bin_edges,labels=[0,1,2,3]) # Categorize the ages into 4 labels

X_train['Embarked'] = X_train['Embarked'].fillna(X_train['Embarked'].mode()[0]) # Filling in missing data with mode imputation(the most frequent varient)

#One hot encoding of sex of the passangers(1.0 - male, 0.0 - female)
sex_encoder = OneHotEncoder(sparse_output=False, drop='first')
sex_encoded = sex_encoder.fit_transform(X_train[['Sex']])
sex_df = pd.DataFrame(sex_encoded, columns=sex_encoder.get_feature_names_out(['Sex']))

# One hot encoding for 'Embarked' column
emb_encoder = OneHotEncoder(sparse_output=False, drop='first')
emb_encoded = emb_encoder.fit_transform(X_train[['Embarked']])
emb_df = pd.DataFrame(emb_encoded, columns=emb_encoder.get_feature_names_out(['Embarked']))

# Dropping unneccessary columns, add the new 'Sex' and 'Embarked' columns
X_train = X_train.drop(['Sex', 'Embarked', 'Name', 'Ticket', 'Fare','Cabin'], axis=1)
X_train = pd.concat([X_train, sex_df, emb_df], axis=1)

# Save the processed data
X_train.to_csv("processed_data.csv")

# The same for the test data
X_test = pd.read_csv("test.csv")
test_result = pd.read_csv("gender_submission.csv")
y_test = test_result['Survived']

X_test['Age'] = X_test['Age'].fillna(X_test['Age'].median())
bin_edges = [0,20,40,60,80]
X_test['Age'] = pd.cut(X_test['Age'], bin_edges,labels=[0,1,2,3])

X_test['Embarked'] = X_test['Embarked'].fillna(X_test['Embarked'].mode()[0])

sex_encoder = OneHotEncoder(sparse_output=False, drop='first')
sex_encoded = sex_encoder.fit_transform(X_test[['Sex']])
sex_df = pd.DataFrame(sex_encoded, columns=sex_encoder.get_feature_names_out(['Sex']))

emb_encoder = OneHotEncoder(sparse_output=False, drop='first')
emb_encoded = emb_encoder.fit_transform(X_test[['Embarked']])
emb_df = pd.DataFrame(emb_encoded, columns=emb_encoder.get_feature_names_out(['Embarked']))

X_test = X_test.drop(['Sex', 'Embarked', 'Name', 'Ticket', 'Fare','Cabin'], axis=1)
X_test = pd.concat([X_test, sex_df, emb_df], axis=1)

#Making sure Test has all the same columns as in Train
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)


tune_time = time.time() # Starting the timer for hyperparameter tuning

# Initiating Random Forest model with bootstrap and 'gini' criterion
model = RandomForestClassifier(random_state=0, 
                               bootstrap=True, 
                               oob_score=True,
                               criterion='gini',
                               n_jobs = -1)

# Tuning parameters with the most impact
param_dist = {'n_estimators': [50, 100, 150, 200, 250],
              'max_depth':[2, 4, 5, 6, 7],
              'min_samples_leaf': randint(1, 10),
              'min_samples_split': randint(2, 20),
              'max_leaf_nodes': [None, 5, 10, 20]}

# RancomizedSearchCV with 30 iterations and 5 CV
rand_search = RandomizedSearchCV(
    random_state=0,
    estimator=model,
    param_distributions=param_dist,
    n_iter=30,
    cv=5
)
rand_search.fit(X_train, y_train)

# Printing out the best parameters
print('Best parameters: ')
for param, val in rand_search.best_params_.items():
    print(f"{param}: {val}")

best_model = rand_search.best_estimator_ # using the best model found by randomized search

# The mean, std, oob
scores = cross_val_score(best_model, X_train, y_train, cv = 5)
print(f"Mean: {scores.mean():.2f} Std: {scores.std():.3f} OOB: {best_model.oob_score_:.2f}")

tune_time_end = time.time() # timer stop

# Training and test accuracy
print(f"Training accuracy: {best_model.score(X_train, y_train):.2f}")
print(f"Test accuracy: {best_model.score(X_test, y_test):.2f}")

# Total hyperparameter tuning time
timer = tune_time_end - tune_time 
print(f"Hyperparameter tuning time: {timer:.2f} s")