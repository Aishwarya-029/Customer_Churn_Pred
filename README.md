# 🔮 Customer Churn Prediction with Machine Learning

Customer retention is one of the biggest challenges for businesses today. This project focuses on **predicting customer churn** using machine learning so companies can take proactive actions to keep their customers.

---

## 🌟 Why This Project?
- Acquiring new customers is **5x more expensive** than retaining existing ones.  
- By predicting churn, companies can offer **discounts, personalized offers, or better services** to customers most likely to leave.  
- This project demonstrates how data-driven insights can directly support **business strategy**.

---

## 📊 Dataset
- **Source:** Kaggle – Telco Customer Churn  
- **Rows:** ~7,000 customer records  
- **Features:**  
  - Demographics → `gender`, `SeniorCitizen`, `Partner`, `Dependents`  
  - Services → `PhoneService`, `InternetService`, `TechSupport`, `StreamingTV`, etc.  
  - Account Info → `Contract`, `PaperlessBilling`, `PaymentMethod`  
  - Charges → `MonthlyCharges`, `TotalCharges`, `tenure`  
- **Target Variable:** `Churn` (Yes/No)

---

## 🛠️ Project Workflow
1. **EDA & Visualization** → Discovered key churn drivers like contract type and payment method.  
2. **Data Preprocessing** → Cleaned missing values, encoded categorical variables, scaled features.  
3. **Model Training** → Tested Logistic Regression, Random Forest, Gradient Boosting.  
   - 🏆 Random Forest gave the best accuracy (~85%).  
4. **Deployment** → Built an interactive **Streamlit app** (`main.py`) for real-time predictions.

---

## 🚀 How to Use
Clone the repo and run the Streamlit app:

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
streamlit run main.py
