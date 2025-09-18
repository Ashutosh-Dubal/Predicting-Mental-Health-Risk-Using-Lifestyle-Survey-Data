
# 🧠 Predicting Mental Health In Tech Using Survy

> This project is still being worked on

This project explores the 2016 OSMI Mental Health in Tech Survey, a dataset that captures how employees in the technology sector perceive and respond to mental health challenges in the workplace. The goal of this project is twofold:  

1. Exploratory Data Analysis (EDA):  
	• Understand demographic and workplace factors that influence mental health.  
	• Visualize patterns across age, gender, company size, geography, and workplace policies.  
  
2. Predictive Modeling:  
	Build machine learning models to predict whether an individual is likely to:       
		• Seek help if they face mental health challenges.  
		• Seek treatment for mental health conditions.  
	Compare the influence of workplace support, personal history, and company culture on awareness versus action.  
  
By combining EDA with predictive modeling, this project aims to provide insights into how workplace policies and cultural attitudes impact mental health outcomes.  

## 📚 Table of Contents  
1. [Dataset Description]()  
2. [Challenges & Learnings]()  
3. [How to Install and Run the Project]()  
4. [How to Use the Project]()  
5. [Sample Output]()  
6. [Key Insights & Analysis]()  
7. [Tech Stack]()  
8. [Project Structure]()  
9. [Author]()  
10. [License]()  

---

## 📦 Dataset Description

The dataset comes from the OSMI Mental Health in Tech Survey (2016), which collected responses from employees in the technology sector about their experiences with mental health.

• Rows (responses): 1,251  
  
• Columns (features): 27 survey questions + derived cleaned features  

• Key features include:  

        • Demographics: Age, Gender, Country, State (US only)            
        
        • Workplace Factors: company size, remote work, leave policies, benefits, wellness programs, anonymity  
        
        • Personal/Social Factors: family history of mental illness, supervisor and coworker support  
        
		 • Targets:  
  
            • treatment: whether the respondent sought treatment for mental health  
            
            • seek_help: whether the respondent knows about options/resources for help  

The raw dataset is publicly available on [Kaggle](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey).

---

## 🧠 Challenges & Learnings  

---

## 🛠️ How to Install and Run the Project

---

## 📦 How to Use This Project

---

## 📊 Sample Output 

---

## 👁 Key Insights & Analysis

---

## 🔧 Tech Stack

This project is built with: Python 3.9

Data Processing & Analysis: pandas, numpy

Visualization: matplotlib, seaborn

Modeling (planned): scikit-learn (Random Forest, Logistic Regression, Gradient Boosting, etc.)

Project Organization: git & GitHub for version control

---
## 📁 Project Structure

```
toronto-bike-share-analysis/  
├── data/  
│   ├── raw/                  # Original raw CSVs 
│      └── survy.csv  
│   └── clean/                # Cleaned & engineered datasets   
│      └── survy.csv  
│  
├── visuals/                      # Generated plots and charts  
│   └── EDA/  
│  
├── src/                          # Main Python scripts  
│   ├── fetch_data.py  
│   ├── clean_data.py  
│   └── EDA.py  
│  
├── requirements.txt              # Required Python packages  
├── run_pipeline.sh               # Optional shell script to run the full analysis  
├── README.md                     # Project documentation (you are here!)  
└── .gitignore                    # Files and folders to ignore in version control  
```

---

## 👨‍💻 Author

Ashutosh Dubal  
🔗 [GitHub Profile](https://github.com/Ashutosh-Dubal)

---

## 📜 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
