💼 Employee Salary Prediction Using Machine Learning
📌 Project Overview
This project predicts employee salary/income using machine learning regression algorithms.

The project uses an employee income dataset obtained from Kaggle. The data is preprocessed, analyzed, visualized, and used to train multiple machine learning models.

Three regression algorithms are compared:

Linear Regression
Decision Tree Regression
Random Forest Regression
The models are evaluated using MAE, MSE, RMSE, and R² Score. The model with the highest R² Score is selected as the best-performing model.

🎯 Objectives
The main objectives of this project are:

Analyze employee salary data.
Perform data preprocessing.
Handle missing values and duplicate records.
Convert categorical data into numerical values.
Perform exploratory data analysis.
Identify relationships between employee attributes and salary.
Train multiple regression models.
Compare model performance.
Visualize model results.
Identify important salary-related features.
Select the best-performing machine learning model.
📊 Dataset
The project uses the Employee Income / Employee Salary dataset obtained from Kaggle.

The dataset contains employee-related information that can be used to estimate income or salary.

The target variable is the employee's:

Salary / Income

The exact columns depend on the version of the Kaggle dataset being used.

🛠️ Technologies Used
Programming Language
Python
Libraries
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Machine Learning Algorithms
Linear Regression
Decision Tree Regression
Random Forest Regression
Development Tools
Visual Studio Code
Git
GitHub
📂 Project Structure
Employee-Salary-Prediction/
│
├── employee_income.csv
├── employee_salary_prediction.py
├── README.md
├── requirements.txt
│
├── salary_distribution.png
├── correlation_heatmap.png
├── model_comparison.png
├── decision_tree.png
├── feature_importance.png
└── actual_vs_predicted.png

🔄 Machine Learning Workflow
Dataset
   ↓
Data Loading
   ↓
Data Inspection
   ↓
Remove Duplicates
   ↓
Handle Missing Values
   ↓
Encode Categorical Features
   ↓
Exploratory Data Analysis
   ↓
Feature & Target Selection
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Feature Importance
   ↓
Best Model Selection

🧹 Data Preprocessing
The following preprocessing steps are performed:

1. Remove Duplicate Records
Duplicate rows are removed using:

df.drop_duplicates(inplace=True)

2. Handle Missing Values
Numerical missing values are replaced using the median, while categorical missing values are handled using the mode.

3. Convert Salary to Numeric
Salary values containing symbols such as $, ₹, £, or € and commas are cleaned before converting them into numerical values.

4. Encode Categorical Features
Categorical columns are converted into numerical values using LabelEncoder.

📈 Exploratory Data Analysis
Several visualizations are created to understand the dataset.

Salary Distribution
A histogram is created to understand the distribution of employee salaries.

Output:

salary_distribution.png

Correlation Heatmap
A correlation heatmap is generated to understand relationships between the numerical features.

Output:

correlation_heatmap.png

🤖 Machine Learning Models
1. Linear Regression
Linear Regression is used as a baseline regression model.

It attempts to establish a linear relationship between employee characteristics and salary.

linear_model = LinearRegression()

2. Decision Tree Regression
Decision Tree Regression is used to capture non-linear relationships between employee features and salary.

decision_tree = DecisionTreeRegressor(
    random_state=42
)

A separate Decision Tree with limited depth is also created for visualization.

Output:

decision_tree.png

3. Random Forest Regression
Random Forest Regression uses multiple decision trees to improve prediction performance.

random_forest = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

📏 Model Evaluation
The models are evaluated using four regression metrics.

MAE — Mean Absolute Error
MAE measures the average absolute difference between actual and predicted salary.

Lower MAE is better.

MSE — Mean Squared Error
MSE measures the average squared difference between actual and predicted salary.

Lower MSE is better.

RMSE — Root Mean Squared Error
RMSE is the square root of MSE.

It represents prediction error in the same units as salary.

Lower RMSE is better.

R² Score
R² Score measures how well the model explains the variation in employee salary.

Higher R² Score is better.

The model with the highest R² Score is selected as the best-performing model.

📊 Model Comparison
The project compares:

Model	MAE	MSE	RMSE	R² Score
Linear Regression	Calculated during execution	Calculated during execution	Calculated during execution	Calculated during execution
Decision Tree	Calculated during execution	Calculated during execution	Calculated during execution	Calculated during execution
Random Forest	Calculated during execution	Calculated during execution	Calculated during execution	Calculated during execution

The exact results depend on the dataset and train-test split.

The comparison is visualized in:

model_comparison.png

⭐ Feature Importance
The Decision Tree model is used to identify which features have the greatest influence on salary prediction.

The feature importance graph is saved as:

feature_importance.png

This visualization helps identify the most influential employee attributes.

📉 Actual vs Predicted Salary
The project also creates a scatter plot comparing actual salary values with predicted salary values from the Random Forest model.

Output:

actual_vs_predicted.png

A prediction line is included to make it easier to understand how close the predictions are to the actual values.

🏆 Best Model
The program automatically identifies the best model using the highest R² Score.

Example:

================================
BEST MODEL
================================

Best Performing Algorithm: Random Forest
Best R2 Score: 0.XX

The actual result is determined when the program is executed.

🚀 Installation
Step 1: Clone the Repository
git clone https://github.com/jishnuk46/Employee-Salary-Prediction.git

Step 2: Open the Project
cd Employee-Salary-Prediction

Step 3: Install Dependencies
pip install -r requirements.txt

Or install the libraries directly:

pip install pandas numpy matplotlib seaborn scikit-learn

▶️ How to Run
Run the Python program:

python employee_salary_prediction.py

The program will:

Load the dataset.
Inspect the data.
Remove duplicate records.
Handle missing values.
Encode categorical features.
Perform exploratory data analysis.
Split the dataset into training and testing data.
Train three regression models.
Calculate evaluation metrics.
Compare the models.
Generate visualization files.
Identify the best-performing model.
📁 Generated Files
After running the program, the following files are generated:

salary_distribution.png
correlation_heatmap.png
model_comparison.png
decision_tree.png
feature_importance.png
actual_vs_predicted.png

🔐 Data Privacy
This project should use publicly available or sample data.

Do not upload confidential employee information to GitHub, including:

Personal phone numbers
Home addresses
Government ID numbers
Bank information
Passwords
Private employee records
Use only data that you are legally allowed to publish.

⚠️ Limitations
The current project has some limitations:

LabelEncoder is used for categorical features.
Hyperparameter tuning is not performed.
Only one train-test split is used.
Cross-validation is not implemented.
The model's performance depends on the quality of the dataset.
Salary prediction can be affected by factors that are not included in the dataset.
🔮 Future Improvements
Future versions of the project can include:

One-Hot Encoding for categorical features.
Cross-validation.
Hyperparameter tuning.
GridSearchCV.
RandomizedSearchCV.
Gradient Boosting.
XGBoost.
Extra Trees Regression.
Feature engineering.
Outlier detection.
A web interface using Streamlit.
Saving the trained model using Joblib.
An interactive salary prediction system.
📚 Learning Outcomes
This project provides practical experience in:

Python programming
Data preprocessing
Exploratory Data Analysis
Data visualization
Regression
Machine learning
Model evaluation
Feature importance
Git
GitHub
Project documentation
👨‍💻 Author
Jishnu K

Student Machine Learning Project

📄 License
This project is created for educational purposes.

You are free to use and modify the project for learning and academic purposes.
