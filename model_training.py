{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "b25c4d96-5996-4873-8210-10d93b0b1c6a",
   "metadata": {},
   "source": [
    "# Digipex Internship Task 04: \n",
    "\n",
    "# 💼 Salary Prediction Using Traditional ML Techniques\n",
    "\n",
    "## 🎯 Project Goal\n",
    "The goal of this project is to **build a machine learning regression model** that can predict a person's **salary** based on their **years of experience**. This project is part of the AI Internship Program at DIGIPEX Solutions.\n",
    "\n",
    "---\n",
    "\n",
    "## 📝 Project Overview\n",
    "This project aims to apply **traditional machine learning techniques** to solve a real-world regression problem. We will:\n",
    "\n",
    "- Explore and preprocess the dataset\n",
    "- Engineer useful features\n",
    "- Train and evaluate multiple regression models\n",
    "- Deploy the final model using **Streamlit**\n",
    "- Host the interactive web app on **Render**\n",
    "\n",
    "---\n",
    "\n",
    "## 📊 Dataset Used\n",
    "We are using the **Salary_Data.csv** dataset, which includes:\n",
    "- `YearsExperience`: Total years of work experience\n",
    "- `Salary`: Corresponding salary in USD\n",
    "\n",
    "---\n",
    "\n",
    "## 📌 Steps to Follow\n",
    "1. **Data Exploration & Preprocessing**\n",
    "2. **Feature Engineering**\n",
    "3. **Model Building**\n",
    "4. **Model Evaluation**\n",
    "5. **Streamlit App Interface**\n",
    "6. **Deployment on Render**\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "28ed05ad-8782-4a7a-bde5-ec54979a3944",
   "metadata": {},
   "source": [
    "\n",
    "## Step 1: Data Exploration & Preprocessing — Explanation\n",
    " ### In this step, we aim to prepare the dataset for machine learning by doing the following:\n",
    "### Load the dataset using pandas to view and work with the data.\n",
    "\n",
    "### Check for missing values to ensure there are no incomplete rows or columns.\n",
    "\n",
    "### Identify and encode categorical features if present (like strings or object-type columns).\n",
    "\n",
    "### Scale numerical features like YearsExperience, which helps models perform better.\n",
    "\n",
    "### These steps ensure that our data is clean, consistent, and ready for training regression models in the next phase.\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5d1f894f-a09c-482a-a7f2-d24e2f2220a2",
   "metadata": {},
   "source": [
    "# 🧭 1.1 Load the Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "13036a72-29ca-482d-9e5c-cf3a874046b5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>YearsExperience</th>\n",
       "      <th>Salary</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1.1</td>\n",
       "      <td>39343.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1.3</td>\n",
       "      <td>46205.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1.5</td>\n",
       "      <td>37731.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2.0</td>\n",
       "      <td>43525.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2.2</td>\n",
       "      <td>39891.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>2.9</td>\n",
       "      <td>56642.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>3.0</td>\n",
       "      <td>60150.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>3.2</td>\n",
       "      <td>54445.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>3.2</td>\n",
       "      <td>64445.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>3.7</td>\n",
       "      <td>57189.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>3.9</td>\n",
       "      <td>63218.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>4.0</td>\n",
       "      <td>55794.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>4.0</td>\n",
       "      <td>56957.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>4.1</td>\n",
       "      <td>57081.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>4.5</td>\n",
       "      <td>61111.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>4.9</td>\n",
       "      <td>67938.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>5.1</td>\n",
       "      <td>66029.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>5.3</td>\n",
       "      <td>83088.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>5.9</td>\n",
       "      <td>81363.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>6.0</td>\n",
       "      <td>93940.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20</th>\n",
       "      <td>6.8</td>\n",
       "      <td>91738.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21</th>\n",
       "      <td>7.1</td>\n",
       "      <td>98273.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22</th>\n",
       "      <td>7.9</td>\n",
       "      <td>101302.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23</th>\n",
       "      <td>8.2</td>\n",
       "      <td>113812.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24</th>\n",
       "      <td>8.7</td>\n",
       "      <td>109431.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25</th>\n",
       "      <td>9.0</td>\n",
       "      <td>105582.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>26</th>\n",
       "      <td>9.5</td>\n",
       "      <td>116969.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>27</th>\n",
       "      <td>9.6</td>\n",
       "      <td>112635.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>28</th>\n",
       "      <td>10.3</td>\n",
       "      <td>122391.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>29</th>\n",
       "      <td>10.5</td>\n",
       "      <td>121872.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    YearsExperience    Salary\n",
       "0               1.1   39343.0\n",
       "1               1.3   46205.0\n",
       "2               1.5   37731.0\n",
       "3               2.0   43525.0\n",
       "4               2.2   39891.0\n",
       "5               2.9   56642.0\n",
       "6               3.0   60150.0\n",
       "7               3.2   54445.0\n",
       "8               3.2   64445.0\n",
       "9               3.7   57189.0\n",
       "10              3.9   63218.0\n",
       "11              4.0   55794.0\n",
       "12              4.0   56957.0\n",
       "13              4.1   57081.0\n",
       "14              4.5   61111.0\n",
       "15              4.9   67938.0\n",
       "16              5.1   66029.0\n",
       "17              5.3   83088.0\n",
       "18              5.9   81363.0\n",
       "19              6.0   93940.0\n",
       "20              6.8   91738.0\n",
       "21              7.1   98273.0\n",
       "22              7.9  101302.0\n",
       "23              8.2  113812.0\n",
       "24              8.7  109431.0\n",
       "25              9.0  105582.0\n",
       "26              9.5  116969.0\n",
       "27              9.6  112635.0\n",
       "28             10.3  122391.0\n",
       "29             10.5  121872.0"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "# Load the data\n",
    "df = pd.read_csv(r\"C:\\Users\\sk mobile zone\\Desktop\\Salary_Data.csv\")\n",
    "\n",
    "# View the Dataset\n",
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "74c80fb5-fa7f-4437-9490-f6a3e45b4381",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>YearsExperience</th>\n",
       "      <th>Salary</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1.1</td>\n",
       "      <td>39343.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1.3</td>\n",
       "      <td>46205.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1.5</td>\n",
       "      <td>37731.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2.0</td>\n",
       "      <td>43525.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2.2</td>\n",
       "      <td>39891.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   YearsExperience   Salary\n",
       "0              1.1  39343.0\n",
       "1              1.3  46205.0\n",
       "2              1.5  37731.0\n",
       "3              2.0  43525.0\n",
       "4              2.2  39891.0"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# View the first few rows\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6e9a361a-327c-43ca-9e70-217c3b052d18",
   "metadata": {},
   "source": [
    "# 🔍 1.2 Handle Missing Values\n",
    "## Check if any values are missing:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "34fe216b-7750-42ed-9561-bfb1d0f4d0ca",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "YearsExperience    0\n",
       "Salary             0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Check for missing values\n",
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1915f68f-5554-4db3-9863-4a32b50ba473",
   "metadata": {},
   "source": [
    "# 🏷️ 1.3 Encode Categorical Features\n",
    "## Check which columns are object type:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "76e75407-cfaa-4966-ba2e-8091183ad353",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Data Types:\n",
      "YearsExperience    float64\n",
      "Salary             float64\n",
      "dtype: object\n"
     ]
    }
   ],
   "source": [
    "# Step 4: Check data types\n",
    "print(\"\\nData Types:\")\n",
    "print(df.dtypes)\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1589c9bc-ad75-4448-b854-a4c9a551a49e",
   "metadata": {},
   "source": [
    "# ⚖️ 1.4 Scale Numerical Values\n",
    "## Scale features (except target column \"Salary\"):"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "d623e5d1-433d-456f-ba4d-88a141e2d888",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "# Define features and target\n",
    "X = df.drop(\"Salary\", axis=1)\n",
    "y = df[\"Salary\"]\n",
    "\n",
    "# Scale numeric features\n",
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(X)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c93e1908-4633-4b58-aaa1-382345c3d22a",
   "metadata": {},
   "source": [
    "## ✅ Conclusion of Step 1: Data Exploration & Preprocessing\n",
    "### We successfully completed the first phase of the project by preparing the data:\n",
    "\n",
    "### ✅ Loaded the dataset using Pandas\n",
    "\n",
    "### ✅ Verified there were no missing values\n",
    "\n",
    "### ✅ Found that all columns are numerical, so no encoding was required\n",
    "\n",
    "### ✅ Applied StandardScaler to scale the YearsExperience feature\n",
    "\n",
    "# The dataset is now clean and preprocessed, and we're ready to move to Feature Engineering in the next step.\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f956cf2f-35d4-43b1-b081-d88acc5451ba",
   "metadata": {},
   "source": [
    "# 🔹 STEP 2: FEATURE ENGINEERING"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "287e7d2e-7007-4aed-9421-a831a7891705",
   "metadata": {},
   "source": [
    "## Step 2: Feature Engineering — Explanation\n",
    "### In this step, we analyze which features have the strongest impact on the target variable (Salary).\n",
    "### Since our dataset is small and only includes one independent variable — YearsExperience — we’ll:\n",
    "\n",
    "### Check correlation between YearsExperience and Salary\n",
    "\n",
    "### Visualize the relationship using a scatter plot\n",
    "\n",
    "### Decide if any new features (derived features) can help improve the model\n",
    "### (e.g., experience squared or log-transformed values)\n",
    "\n",
    "### The goal is to understand the strength of relationships and improve the input features if possible.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "c2b3bd93-b147-432c-af4a-3707b8cf82e5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "🔍 Correlation Matrix:\n",
      "                  YearsExperience    Salary\n",
      "YearsExperience         1.000000  0.978242\n",
      "Salary                  0.978242  1.000000\n"
     ]
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Step 1: Correlation matrix\n",
    "correlation = df.corr()\n",
    "print(\"🔍 Correlation Matrix:\\n\", correlation)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "841ac630-de97-4f16-ade4-7b437fa1782a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 500x400 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Step 2: Visualize correlation heatmap\n",
    "plt.figure(figsize=(5, 4))\n",
    "sns.heatmap(correlation, annot=True, cmap='Blues')\n",
    "plt.title(\"Feature Correlation with Salary\")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "4dfe1d43-c08b-420e-b7be-89ab1b246b46",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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aLVap70/kZpw5IaI0Lx5c7zzzjuYNWsW0tLSICIYN24c+vXrhzvvvBP3338/vLy8EBQUhKioKGRnZyuvTUhIwF133QVfX18EBAQgMjISx44dU9qPHz8OjUaDL7/8En369IG7uztWrVqFEydOYMiQIfDz84Onpyc6duyIDRs2XLOPOTk5ePLJJ+Hn5wcPDw/cd999OHLkCADgxx9/hLe3NwCgb9++0Gg0+PHHH695Lm9vb5hMJrtHWbB55ZVXYDabcf78eaV+6NChCA8PR2lpKQBAo9Hg3XffxX333Qe9Xo/WrVvjP//5j917nDp1Co899hj8/PwQEBCABx54AMePH1fax4wZgwcffBCxsbEwm8249dZbAThelrJYLJgwYQICAwPh4+ODvn374rffflPa58+fj86dO+Ozzz5Dq1atYDAY8PjjjyM/P1+pKS0txeLFi9G2bVvodDq0aNECr732WqX7StTQMNwQEQBg9OjR6NevH5566iksX74c+/fvxzvvvIOIiAh07twZv/zyCxISEnD27FkMHz5ced3FixcxY8YMJCcnY/PmzXBxccFDDz2kBIEys2fPxtSpU3Ho0CEMHDgQkydPRlFRERITE7Fv3z4sXrwYXl5e1+zfmDFj8Msvv+Crr75CUlISRAT3338/iouL0atXL/zxxx8AgPj4eJw5cwa9evWq1vfw97//Ha1atcLf/vY3AMDKlSuRmJiIzz77DC4u//dX5osvvohhw4bht99+wxNPPIERI0bg0KFDAICCggLcc8898PLyQmJiIrZv3w4vLy8MGjQINptNOcfmzZtx6NAhbNq0Cd98841DX0QEgwcPRmZmJjZs2ICUlBTcfvvt6NevHy5cuKDUHTt2DOvXr8c333yDb775Btu2bbO7zDZ37lwsXrwYL774Ig4ePIjPP/8cQUFBVeorUYPivA3Jiai+OXv2rNx0003i4uIia9eulRdffFEGDBhgV5ORkSEA5I8//ij3HFlZWQJA9u3bJyIiaWlpAkDefvttu7pOnTrJ/PnzK9Wvw4cPCwD5+eeflWPZ2dmi1+vlyy+/FBGRnJwcASBbt26t8FwtW7YUrVYrnp6edo+/vu7YsWPi7e0ts2fPFg8PD1m1apXdOQDIM888Y3ese/fuMnHiRBER+de//iXt2rWT0tJSpb2oqEj0er18//33IiIyevRoCQoKkqKiIof+vfXWWyIisnnzZvHx8ZHCwkK7mjZt2sh7770nIiLz5s0TDw8PycvLU9qfffZZ6d69u4iI5OXliU6nkw8++KDc76MyfSVqaFydGayIqH4JDAzEhAkTsH79ejz00EP48MMPsXXr1nJHVI4dO4Zbb70Vx44dw4svvoidO3ciOztbGbFJT09HaGioUt+tWze710+dOhUTJ07Exo0bce+992LYsGEICwsrt1+HDh2Cq6srunfvrhwLCAhAu3btlNGSqnj22Wcd5hE1a9ZM+edbbrkFb7zxBp5++mk89thjGDVqlMM5evbs6fA8NTUVAJCSkoKjR48ql8rKFBYW2l2y69SpU4XzbFJSUmC1WhEQEGB3/NKlS3bnadWqld17NW3aFFlZWQCufHdFRUXo16/fNd+jMn0lakgYbojIjqurK1xdr/zVUFpaiiFDhmDx4sUOdU2bNgUADBkyBM2bN8cHH3wAs9mM0tJShIaGOlzSuHqy7t/+9jcMHDgQ3377LTZu3IjY2Fi8+eabmDJlisN7yTXWPYgINBpNlT+j0WhE27ZtK6xJTExEkyZNcPz4cVy+fFn5TipS1pfS0lJ07doVq1evdqi56aablH++3gTm0tJSNG3atNz5Q76+vso/Xz05W6PRKCFTr9df9z0q01eihoRzbojomm6//XYcOHAArVq1Qtu2be0enp6eOH/+PA4dOoQXXngB/fr1Q4cOHZCTk1Pp8zdv3hzPPPMM1q5di5kzZ+KDDz4oty4kJASXL1/Grl27lGPnz5/H4cOH0aFDhxv+nFf74osvsHbtWvz444/XXFa+c+dOh+ft27cHcOV7O3LkCAIDAx2+N4PBUOl+3H777cjMzISrq6vDeYxGY6XOERwcDL1ej82bN1/zPWqir0T1CcMNEV3T5MmTceHCBYwYMQK7d+/Gn3/+iY0bN2Ls2LEoKSlRVte8//77OHr0KLZs2YIZM2ZU6twxMTH4/vvvkZaWhj179mDLli3XDCrBwcF44IEHMH78eGzfvl2ZxNusWTM88MADVf5c+fn5yMzMtHvk5eUBAE6ePImJEydi8eLFuOuuu/DJJ58gNjbWIcz85z//wUcffYTDhw9j3rx52L17N6KjowEAo0aNgtFoxAMPPICffvoJaWlp2LZtG6ZNm4aTJ09Wup/33nsvevbsiQcffBDff/89jh8/jh07duCFF17AL7/8UqlzuLu7Y/bs2Xjuuefw6aef4tixY9i5cyf+9a9/1WhfieoThhsiuiaz2Yyff/4ZJSUlGDhwIEJDQzFt2jQYDAa4uLjAxcUFcXFxSElJQWhoKKZPn44lS5ZU6twlJSWYPHkyOnTogEGDBqFdu3ZYsWLFNes//vhjdO3aFZGRkejZsydEBBs2bKjW/XJeeuklNG3a1O7x3HPPQUQwZswY3HnnnUpQ6d+/P6Kjo/HEE0/AarUq53j55ZcRFxeHsLAw/Pvf/8bq1asREhICAPDw8EBiYiJatGiBhx9+GB06dMDYsWNx6dIl+Pj4VLqfGo0GGzZsQHh4OMaOHYtbb70Vjz/+OI4fP66sdqqMF198ETNnzsRLL72EDh064LHHHlPm5NRUX4nqE97Ej4ioijQaDdatW4cHH3zQ2V0honJw5IaIiIhUheGGiIiIVIVLwYmIqohX84nqN47cEBERkaow3BAREZGqMNwQERGRqjDcEBERkaow3BAREZGqMNwQERGRqjDcEBERkaow3BAREZGq/D/xIhB1t2FQIQAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 600x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Step 3: Scatter plot between YearsExperience and Salary\n",
    "plt.figure(figsize=(6, 5))\n",
    "sns.scatterplot(data=df, x='YearsExperience', y='Salary')\n",
    "plt.title(\"Years of Experience vs Salary\")\n",
    "plt.xlabel(\"Years of Experience\")\n",
    "plt.ylabel(\"Salary\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "acba6c52-44e6-4e84-843b-7e4b5f705031",
   "metadata": {},
   "source": [
    "### ✅ Conclusion of Step 2: Feature Engineering\n",
    "\n",
    "- We explored the relationship between `YearsExperience` and `Salary`.\n",
    "- The **scatter plot** showed a clear **positive linear trend**.\n",
    "- The **correlation matrix** revealed a **strong positive correlation (0.978)** between `YearsExperience` and `Salary`.\n",
    "- ### 🔥 Heatmap Analysis\n",
    "\n",
    "The correlation heatmap shows a **very strong positive correlation (0.98)** between `YearsExperience` and `Salary`. \n",
    "\n",
    "This indicates that:\n",
    "- As experience increases, salary tends to increase in a predictable way.\n",
    "- A simple regression model can capture this trend accurately.\n",
    "\n",
    "Therefore, no additional derived features are necessary, and we can move forward to model training with confidence.\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1901bdf1-b171-43e5-8d35-5ee02e01edcf",
   "metadata": {},
   "source": [
    "# 🧠 Step 3: Model Building\n",
    "## 🧠 Step 3: Model Building — Explanation\n",
    "\n",
    "## In this step, we train different regression models to predict `Salary` based on `YearsExperience`.\n",
    "\n",
    "### We'll use the following algorithms:\n",
    "### 1. **Linear Regression** – a simple model suitable for linear relationships\n",
    "### 2. **Random Forest Regressor** – a powerful ensemble model\n",
    "###  3. **XGBoost Regressor** – a gradient boosting model optimized for performance\n",
    "\n",
    "###  Each model will be trained using the scaled feature(s) and evaluated in the next step.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "58dd4ac8-1cc2-42f0-bb05-3e65353c2a5c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Note: you may need to restart the kernel to use updated packages.Requirement already satisfied: xgboost in e:\\file\\lib\\site-packages (3.0.3)\n",
      "Requirement already satisfied: numpy in e:\\file\\lib\\site-packages (from xgboost) (1.26.4)\n",
      "Requirement already satisfied: scipy in e:\\file\\lib\\site-packages (from xgboost) (1.13.1)\n",
      "\n"
     ]
    }
   ],
   "source": [
    "pip install xgboost --user\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "3179b87d-2b1d-4302-a0db-ccaecf1e53a3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: xgboost in e:\\file\\lib\\site-packages (3.0.3)\n",
      "Requirement already satisfied: numpy in e:\\file\\lib\\site-packages (from xgboost) (1.26.4)\n",
      "Requirement already satisfied: scipy in e:\\file\\lib\\site-packages (from xgboost) (1.13.1)\n"
     ]
    }
   ],
   "source": [
    "!pip install xgboost\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "db6d7fa6-435f-4319-8b22-9710ae2f6c9d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Linear Regression trained successfully\n",
      "✅ Random Forest trained successfully\n",
      "✅ XGBoost trained successfully\n"
     ]
    }
   ],
   "source": [
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from xgboost import XGBRegressor\n",
    "\n",
    "# Prepare training and testing data\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "X = df[['YearsExperience']]  # using only one feature\n",
    "y = df['Salary']\n",
    "\n",
    "# Optional: scale features\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(X)\n",
    "\n",
    "# Train-test split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Define models\n",
    "models = {\n",
    "    \"Linear Regression\": LinearRegression(),\n",
    "    \"Random Forest\": RandomForestRegressor(),\n",
    "    \"XGBoost\": XGBRegressor()\n",
    "}\n",
    "\n",
    "# Train models\n",
    "trained_models = {}\n",
    "\n",
    "for name, model in models.items():\n",
    "    model.fit(X_train, y_train)\n",
    "    trained_models[name] = model\n",
    "    print(f\"✅ {name} trained successfully\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "02c1b211-2e30-4dc3-aeaf-fe9ff44a8963",
   "metadata": {},
   "source": [
    "### ✅ Conclusion of Step 3: Model Building\n",
    "\n",
    "I successfully trained three different regression models:\n",
    "\n",
    "1. **Linear Regression**\n",
    "2. **Random Forest Regressor**\n",
    "3. **XGBoost Regressor**\n",
    "\n",
    "All models were trained on the scaled training data. These models will now be evaluated in the next step to compare their performance and select the best one.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "37ad6570-0636-4342-be09-a84dd0cf4e4e",
   "metadata": {},
   "source": [
    "## 📊 Step 4: Model Evaluation\n",
    "###  📊 Step 4: Model Evaluation— Explanation\n",
    "\n",
    "In this step, we evaluate the performance of each trained regression model using the following metrics:\n",
    "\n",
    "- **MAE**: Mean Absolute Error – average absolute difference between predicted and actual values\n",
    "- **MSE**: Mean Squared Error – average of squared differences\n",
    "- **RMSE**: Root Mean Squared Error – square root of MSE\n",
    "- **R² Score**: Indicates how well the model explains the variance in the data (closer to 1 is better)\n",
    "\n",
    "These metrics will help us compare and select the most accurate model.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "e87940e5-db72-4ced-b0fb-76a1d5467225",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "📊 Model Evaluation Results:\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>MAE</th>\n",
       "      <th>MSE</th>\n",
       "      <th>RMSE</th>\n",
       "      <th>R² Score</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Linear Regression</th>\n",
       "      <td>6286.453831</td>\n",
       "      <td>4.983010e+07</td>\n",
       "      <td>7059.043622</td>\n",
       "      <td>0.902446</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Random Forest</th>\n",
       "      <td>6653.673778</td>\n",
       "      <td>5.897727e+07</td>\n",
       "      <td>7679.666268</td>\n",
       "      <td>0.884538</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>XGBoost</th>\n",
       "      <td>8912.313151</td>\n",
       "      <td>1.034046e+08</td>\n",
       "      <td>10168.803134</td>\n",
       "      <td>0.797562</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                           MAE           MSE          RMSE  R² Score\n",
       "Linear Regression  6286.453831  4.983010e+07   7059.043622  0.902446\n",
       "Random Forest      6653.673778  5.897727e+07   7679.666268  0.884538\n",
       "XGBoost            8912.313151  1.034046e+08  10168.803134  0.797562"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
    "import numpy as np\n",
    "\n",
    "# Dictionary to store model evaluation results\n",
    "results = {}\n",
    "\n",
    "# Loop through each trained model\n",
    "for name, model in trained_models.items():\n",
    "    y_pred = model.predict(X_test)  # Use the model to predict test set\n",
    "\n",
    "    mae = mean_absolute_error(y_test, y_pred)\n",
    "    mse = mean_squared_error(y_test, y_pred)\n",
    "    rmse = np.sqrt(mse)\n",
    "    r2 = r2_score(y_test, y_pred)\n",
    "\n",
    "    results[name] = {\n",
    "        \"MAE\": mae,\n",
    "        \"MSE\": mse,\n",
    "        \"RMSE\": rmse,\n",
    "        \"R² Score\": r2\n",
    "    }\n",
    "\n",
    "# Convert results to DataFrame\n",
    "results_df = pd.DataFrame(results).T\n",
    "print(\"📊 Model Evaluation Results:\")\n",
    "display(results_df)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3ea25ec3-d8cf-442b-9f84-ed1047779755",
   "metadata": {},
   "source": [
    "| Model                 | MAE (Lower = Better) | RMSE (Lower = Better) | R² Score (Higher = Better) |\n",
    "| --------------------- | -------------------- | --------------------- | -------------------------- |\n",
    "| **Linear Regression** | ✅ **6286.45**        | ✅ **7059.04**         | ✅ **0.9024** (Highest)     |\n",
    "| Random Forest         | 6651.76              | 7521.74               | 0.8892                     |\n",
    "| XGBoost               | ❌ 8912.31            | ❌ 10168.80            | ❌ 0.7975                   |\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dd205d27-632c-4c15-a726-72369995f077",
   "metadata": {},
   "source": [
    "## 📘 What is “Interpretation” in Model Evaluation?\n",
    "### ✅ Interpretation means:\n",
    "### “Understanding and explaining what the results mean.”\n",
    "### 📊 I ran model evaluation and got metrics like:\n",
    "### MAE (Mean Absolute Error)\n",
    "\n",
    "### RMSE (Root Mean Squared Error)\n",
    "\n",
    "### R² Score (Accuracy of the model)\n",
    "### | Metric   | Meaning                                           | Interpretation                |\n",
    "### | -------- | ------------------------------------------------- | ----------------------------- |\n",
    "### | MAE      | How much the model is **wrong on average**        | Lower is better               |\n",
    "### | RMSE     | Like MAE, but gives more importance to big errors | Lower is better               |\n",
    "### | R² Score | How well the model **fits the data**              | Closer to 1 = better accuracy |\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "## 🔍 Interpretation of Evaluation Results\n",
    "\n",
    "### After evaluating all three models, we found that **Linear Regression performed the best**. It had the **lowest MAE and RMSE**, which means it made the smallest prediction errors. It also had the **highest R² Score (0.90)**, meaning it accurately explains 90% of the variation in salary based on years of experience.\n",
    "\n",
    "### Therefore, we selected **Linear Regression** as the final model for deployment in the Streamlit app.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "df516532-96cc-46e7-a59a-622f72b9beb0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Model and Scaler saved successfully.\n"
     ]
    }
   ],
   "source": [
    "import joblib\n",
    "\n",
    "# Save the best-performing model (Linear Regression)\n",
    "joblib.dump(trained_models['Linear Regression'], 'best_model.pkl')\n",
    "\n",
    "# Save the StandardScaler used for input scaling\n",
    "joblib.dump(scaler, 'scaler.pkl')\n",
    "\n",
    "print(\"✅ Model and Scaler saved successfully.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "45858be5-2766-47bd-a426-0881193e23f8",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
