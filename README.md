
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

- Rows (responses): 1,251  
- Columns (features): 27 survey questions + derived cleaned features  
- Key features include:
        
  - Demographics: Age, Gender, Country, State (US only)
  - Workplace Factors: company size, remote work, leave policies, benefits, wellness programs, anonymity
  - Personal/Social Factors: family history of mental illness, supervisor and coworker support
  - Targets:
    - treatment: whether the respondent sought treatment for mental health  
    - seek_help: whether the respondent knows about options/resources for help  

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

The survey reveals how workplace environment, personal history, and demographics interact to shape whether individuals seek help or pursue treatment for mental health concerns.
	
1. Family history leaves a strong imprint
    
   - People with a family history of mental illness are far more likely to pursue treatment, suggesting that prior exposure raises awareness and reduces stigma.
   - Interestingly, when it comes to simply seeking help, the relationship is weaker — individuals may only act when symptoms become unavoidable.
	
2. Workplace culture matters
    
    - Supervisor support emerges as a major factor: employees who feel their supervisors are supportive are more likely to seek help and follow through with treatment.
    - Coworker openness also plays a role, encouraging individuals to reach out before problems escalate. These findings highlight that mental health policies are only effective when paired with a supportive culture.
	
3. Self-employment shows hidden risks
    
    - Self-employed individuals are consistently less likely to seek help or access treatment. Without HR, structured healthcare plans, or coworker networks, they may face additional barriers to prioritizing mental health.

4. Company size creates opportunity gaps
    
    - Larger organizations tend to see higher rates of both help-seeking and treatment. This may reflect access to stronger healthcare benefits, formal leave policies, and peer support systems compared to smaller companies or startups.

5. Gender and cultural differences remain significant

    - Gender analysis suggests disparities in help-seeking and treatment, with women often reporting higher engagement compared to men.
    - Country-level data reveals wide variation, likely tied to differences in healthcare systems, cultural norms, and stigma around discussing mental health.

8. Correlation patterns confirm the story
    
    - The top correlated features for treatment center on family history and company size. 
    - The top correlated features for seeking help emphasize supervisor support, coworker relationships, and organizational context.

🆚 Seek Help vs Treatment

One of the central goals of this project is to understand the difference between seeking help for mental health and actually receiving treatment. While these two concepts are related, the data shows they are driven by different factors:

🔍 Seek Help
	
- Represents the intent or willingness to reach out for support (talking to coworkers, considering counseling, disclosing to a supervisor).
- Strongly influenced by workplace culture and social support:
  - Employees with supportive supervisors or open coworkers are much more likely to seek help.
  - Company size also plays a role, with larger organizations normalizing help-seeking more than smaller companies or startups.  

💊 Treatment
	
- Represents the actual act of receiving professional care (therapy, counseling, medical treatment).
- More strongly tied to structural and access-related factors:
  - Family history of mental illness increases awareness and likelihood of pursuing treatment.
  - Company size and employment status matter — larger companies often provide better healthcare coverage, while self-employed individuals face significant barriers.

⚖️ Key Takeaway
	
- Seek Help is shaped by social and cultural dynamics — whether employees feel safe and supported enough to ask.
- Treatment is shaped by resources and access — whether they can actually get professional care once they decide to.

| Aspect                  | **Seek Help** 🗣️ | **Treatment** 💊 |
|--------------------------|------------------|------------------|
| **Definition**          | Willingness or intent to reach out for support (coworkers, supervisors, considering counseling). | Actual act of receiving professional care (therapy, medication, medical treatment). |
| **Main Drivers**        | Workplace culture, supervisor/coworker support, openness to discuss mental health. | Access to resources, healthcare coverage, family history of mental illness. |
| **Influenced by**       | Social and cultural factors (stigma, peer support, company openness). | Structural and financial factors (insurance, company size, employment status). |
| **Key Predictors**      | Supportive supervisor 👩‍💼, supportive coworkers 👥, company size 🏢. | Family history 🧬, company healthcare policies 🏥, self-employment status 💼. |
| **Typical Barriers**    | Stigma, fear of disclosure, lack of supportive culture. | Cost, lack of insurance, limited access to professional care. |
| **Survey Pattern**      | Higher in large companies and supportive environments. | Higher among those with family history and stable healthcare access. |

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
