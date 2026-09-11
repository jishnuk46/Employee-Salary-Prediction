import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =====================================================
# 1. LOAD DATASET
# =====================================================

# Change the filename if your Kaggle file has a different name
df = pd.read_csv("employee_income.csv")

print("================================")
print("EMPLOYEE SALARY PREDICTION")
print("================================")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())


# =====================================================
# 2. REMOVE DUPLICATES
# =====================================================

df.drop_duplicates(inplace=True)

print("\nShape after removing duplicates:")
print(df.shape)


# =====================================================
# 3. IDENTIFY TARGET COLUMN
# =====================================================

# Common salary/income column names
possible_targets = [
    "salary",
    "Salary",
    "income",
    "Income",
    "salary_in_usd",
    "Salary_In_USD",
    "annual_salary",
    "Annual_Salary"
]

target_column = None

for column in possible_targets:
    if column in df.columns:
        target_column = column
        break


if target_column is None:

    print("\nERROR: Salary/Income column not found.")

    print("Available columns are:")
    print(df.columns.tolist())

    raise ValueError(
        "Please change 'target_column' to the salary column in your dataset."
    )


print("\nTarget Column:")
print(target_column)


# =====================================================
# 4. HANDLE MISSING VALUES
# =====================================================

print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Fill numerical missing values with median
numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

for column in numeric_columns:

    if column != target_column:

        df[column] = df[column].fillna(
            df[column].median()
        )


# Fill categorical missing values with mode
categorical_columns = df.select_dtypes(
    include=["object"]
).columns

for column in categorical_columns:

    if df[column].isnull().sum() > 0:

        df[column] = df[column].fillna(
            df[column].mode()[0]
        )


# Remove rows where target is missing
df = df.dropna(
    subset=[target_column]
)


print("\nMissing values after cleaning:")
print(df.isnull().sum())


# =====================================================
# 5. CONVERT TARGET TO NUMERIC
# =====================================================

# If salary contains commas or currency symbols,
# remove them before converting to numeric.

if df[target_column].dtype == "object":

    df[target_column] = (
        df[target_column]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.replace("€", "", regex=False)
    )

    df[target_column] = pd.to_numeric(
        df[target_column],
        errors="coerce"
    )


# Remove rows where salary could not be converted
df = df.dropna(
    subset=[target_column]
)


# =====================================================
# 6. SALARY DISTRIBUTION
# =====================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    df[target_column],
    kde=True
)

plt.title("Employee Salary Distribution")

plt.xlabel("Salary")

plt.ylabel("Number of Employees")

plt.tight_layout()

plt.savefig(
    "salary_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =====================================================
# 7. ENCODE CATEGORICAL COLUMNS
# =====================================================

label_encoders = {}

for column in df.select_dtypes(
    include=["object"]
).columns:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(
        df[column].astype(str)
    )

    label_encoders[column] = encoder


print("\nData after encoding:")
print(df.head())


# =====================================================
# 8. CORRELATION HEATMAP
# =====================================================

plt.figure(figsize=(12, 8))

sns.heatmap(
    df.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Employee Dataset Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =====================================================
# 9. FEATURES AND TARGET
# =====================================================

X = df.drop(
    target_column,
    axis=1
)

y = df[target_column]


print("\n================================")
print("FEATURES AND TARGET")
print("================================")

print("\nInput Features:")
print(X.columns.tolist())

print("\nTarget:")
print(target_column)


# =====================================================
# 10. TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:")
print(X_train.shape)

print("Testing Data:")
print(X_test.shape)


# =====================================================
# 11. LINEAR REGRESSION
# =====================================================

print("\n================================")
print("LINEAR REGRESSION")
print("================================")

linear_model = LinearRegression()

print("Training Linear Regression...")

linear_model.fit(
    X_train,
    y_train
)

print("Linear Regression trained successfully!")

linear_pred = linear_model.predict(
    X_test
)


linear_mae = mean_absolute_error(
    y_test,
    linear_pred
)

linear_mse = mean_squared_error(
    y_test,
    linear_pred
)

linear_rmse = np.sqrt(
    linear_mse
)

linear_r2 = r2_score(
    y_test,
    linear_pred
)


print("\nLinear Regression Results:")

print("MAE:", linear_mae)

print("MSE:", linear_mse)

print("RMSE:", linear_rmse)

print("R2 Score:", linear_r2)


# =====================================================
# 12. DECISION TREE REGRESSION
# =====================================================

print("\n================================")
print("DECISION TREE")
print("================================")

decision_tree = DecisionTreeRegressor(
    random_state=42
)

print("Training Decision Tree...")

decision_tree.fit(
    X_train,
    y_train
)

print("Decision Tree trained successfully!")


decision_pred = decision_tree.predict(
    X_test
)


decision_mae = mean_absolute_error(
    y_test,
    decision_pred
)

decision_mse = mean_squared_error(
    y_test,
    decision_pred
)

decision_rmse = np.sqrt(
    decision_mse
)

decision_r2 = r2_score(
    y_test,
    decision_pred
)


print("\nDecision Tree Results:")

print("MAE:", decision_mae)

print("MSE:", decision_mse)

print("RMSE:", decision_rmse)

print("R2 Score:", decision_r2)


# =====================================================
# 13. RANDOM FOREST REGRESSION
# =====================================================

print("\n================================")
print("RANDOM FOREST")
print("================================")

random_forest = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

print("Training Random Forest...")

random_forest.fit(
    X_train,
    y_train
)

print("Random Forest trained successfully!")


forest_pred = random_forest.predict(
    X_test
)


forest_mae = mean_absolute_error(
    y_test,
    forest_pred
)

forest_mse = mean_squared_error(
    y_test,
    forest_pred
)

forest_rmse = np.sqrt(
    forest_mse
)

forest_r2 = r2_score(
    y_test,
    forest_pred
)


print("\nRandom Forest Results:")

print("MAE:", forest_mae)

print("MSE:", forest_mse)

print("RMSE:", forest_rmse)

print("R2 Score:", forest_r2)


# =====================================================
# 14. MODEL COMPARISON
# =====================================================

results = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "MAE": [
        linear_mae,
        decision_mae,
        forest_mae
    ],

    "MSE": [
        linear_mse,
        decision_mse,
        forest_mse
    ],

    "RMSE": [
        linear_rmse,
        decision_rmse,
        forest_rmse
    ],

    "R2 Score": [
        linear_r2,
        decision_r2,
        forest_r2
    ]
})


print("\n================================")
print("MODEL COMPARISON")
print("================================")

print(
    results.to_string(index=False)
)


# =====================================================
# 15. R2 SCORE COMPARISON
# =====================================================

plt.figure(figsize=(9, 5))

sns.barplot(
    data=results,
    x="Model",
    y="R2 Score"
)

plt.title(
    "Employee Salary Prediction - Model Comparison"
)

plt.xlabel(
    "Machine Learning Model"
)

plt.ylabel(
    "R2 Score"
)

plt.xticks(
    rotation=15
)

plt.tight_layout()


plt.savefig(
    "model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =====================================================
# 16. DECISION TREE VISUALIZATION
# =====================================================

print("\n================================")
print("DECISION TREE VISUALIZATION")
print("================================")


tree_visual = DecisionTreeRegressor(
    random_state=42,
    max_depth=3
)


tree_visual.fit(
    X_train,
    y_train
)


plt.figure(
    figsize=(25, 12)
)


plot_tree(
    tree_visual,
    feature_names=X.columns,
    filled=True,
    rounded=True,
    fontsize=9
)


plt.title(
    "Decision Tree - Employee Salary Prediction",
    fontsize=18
)


plt.tight_layout()


plt.savefig(
    "decision_tree.png",
    dpi=300,
    bbox_inches="tight"
)


print(
    "Decision Tree saved as decision_tree.png"
)

plt.show()


# =====================================================
# 17. FEATURE IMPORTANCE
# =====================================================

print("\n================================")
print("DECISION TREE FEATURE IMPORTANCE")
print("================================")


feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": decision_tree.feature_importances_

})


feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


print(
    feature_importance
)


plt.figure(
    figsize=(10, 6)
)


sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature"
)


plt.title(
    "Employee Salary - Feature Importance"
)


plt.xlabel(
    "Importance"
)


plt.ylabel(
    "Feature"
)


plt.tight_layout()


plt.savefig(
    "feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# =====================================================
# 18. ACTUAL VS PREDICTED SALARY
# =====================================================

plt.figure(
    figsize=(8, 6)
)


sns.scatterplot(
    x=y_test,
    y=forest_pred
)


plt.xlabel(
    "Actual Salary"
)


plt.ylabel(
    "Predicted Salary"
)


plt.title(
    "Actual vs Predicted Salary - Random Forest"
)


# Perfect prediction line
min_value = min(
    y_test.min(),
    forest_pred.min()
)

max_value = max(
    y_test.max(),
    forest_pred.max()
)


plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    color="red",
    linestyle="--"
)


plt.tight_layout()


plt.savefig(
    "actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# =====================================================
# 19. BEST MODEL
# =====================================================

best_model = results.loc[
    results["R2 Score"].idxmax()
]


print("\n================================")
print("BEST MODEL")
print("================================")


print(
    "Best Performing Algorithm:",
    best_model["Model"]
)


print(
    "Best R2 Score:",
    best_model["R2 Score"]
)


print(
    "\nThe model with the highest R2 Score "
    "is the best-performing model."
)


# =====================================================
# 20. FINAL OUTPUT
# =====================================================

print("\n================================")
print("PROJECT COMPLETED SUCCESSFULLY")
print("================================")


print("\nGenerated files:")

print("1. salary_distribution.png")

print("2. correlation_heatmap.png")

print("3. model_comparison.png")

print("4. decision_tree.png")

print("5. feature_importance.png")

print("6. actual_vs_predicted.png")
