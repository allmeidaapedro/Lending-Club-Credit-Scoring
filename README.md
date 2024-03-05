# Lending Club Credit Scoring

<img src="reports/logoLC.jpeg" height="40%">

# 1. Project Description
- In this project, I will build three **machine learning** models to predict the three components of expected loss in the context of **credit risk modeling** at the **Lending Club** (a peer-to-peer credit company): **Probability of Default (PD), Exposure at Default (EAD) and Loss Given Default (LGD)**. The expected loss will be the product of these elements: **Expected Loss (EL) = PD * EAD * LGD**. These models will be used to stablish a **credit policy**, deciding wheter to grant a loan or not for new applicants **(application model)** based on their **credit scores** and **expected losses** on loans. By estimating the Expected Loss (EL) from each loan, the Lending Club can also assess the required capital to hold to protect itself against defaults.
- The PD modelling encompasses a binary classification problem with target being 1 in case of non-default and 0 in case of default. A Logistic Regression model will be built. 
- The LGD and EAD modelling encompasses a beta regression problem, that is, a regression task in which the dependent variables are beta distributed, the recovery rate and credit conversion factor, respectively.

# 2. Business Problem and Objectives
- **What is the Lending Club?:**
    - LendingClub is a **peer-to-peer lending platform** that facilitates the borrowing and lending of money directly between individuals, without the need for traditional financial institutions such as banks. The platform operates as an online marketplace, connecting borrowers seeking personal loans with investors willing to fund those loans.
- **What is the business problem?**
    - LendingClub faces a significant business challenge related to **managing default risks effectively** while **optimizing returns** for its investors. The platform facilitates peer-to-peer lending, connecting borrowers with investors, and relies on **accurate risk assessments to maintain a sustainable and profitable lending ecosystem.** Thus, the CEO wants us to provide insights about which factors are associated with credit risk in Lending Club's operations, and to construct models capable of predicting the probability of default for new applicants and possible losses on its loans in order to establish a credit policy, deciding when to grant a loan or not for an applicant. An important observation is that the CEO wants these models to be easy to understand. Since our company works on the internet, making customers happy and being clear is really important. So, we need to be able to explain why we decide to approve or deny a loan.
- **Which are the project objectives and benefits?**
    1. Identify the factors associated with **credit risk** in the form of business **insights.**
    2. Develop an accurate **Probability of Default (PD) Model**, constructing a scorecard. This will allow Lending Club to decide wheter to grant a loan or not to a new applicant (**application model**), based on **credit scores.**
    3. Develop **Exposure at Default (EAD) and Loss Given Default (LGD) Models**, to estimate the **Expected Loss** in loans. This will allow Lending Club to **hold** sufficient **capital** to protect itself against default in each loan.
    4. Improve **risk management** and optimize **returns** by establishing a **credit policy**, trying to balance risk and **ROI** of Lending Club's assets.
    5. Apply **model monitoring** and maintenance techniques to safeguard our results from population instability, characterized by significant changes in loan applicants' characteristics. This will allow us to understand whether the built model is still useful in the future or whether the loan applicants characteristics changed significantly, such that we will need to redevelop it.
- **Which are the important concepts to know in the context of credit risk?**
    - **Financial institutions**, like LendingClub and online lending platforms, **make money by lending to people and businesses.** When they lend money, they **charge interest**, which is a significant source of their **profits**. **Managing credit risk well is crucial** for these institutions. This means ensuring that borrowers pay back their loans on time to avoid losses.
    - **Credit risk** is the possibility that a borrower might not fulfill their financial obligations, leading to a loss for the lender. If a borrower fails to meet the agreed-upon terms, it's called a "default," and it can result in financial losses for the lender. The **default** definition is associated with a time horizon. For example, if a borrower hasn't paid their debt within 90 days of the due date, they are considered in default.
    - In the credit market, important **rules** help keep things honest and clear. **Basel III** is one such set of rules, making sure banks have **enough money (capital requirements)** and follow **guidelines for assessing loan risks**. The **Internal Rating-Based Approach (IRB-A)** lets banks figure out credit risks using concepts like Probability of Default (PD), Exposure at Default (EAD), and Loss Given Default (LGD). Another rule, **International Financial Reporting Standard 9 (IFRS 9)**, gives standards for measuring financial assets. It's special because it looks at the chance of a loan not being paid back over its entire life, unlike Basel, which checks it for one year. These rules help banks have enough money, handle risks well, and keep the credit market steady and trustworthy.
    - The **"expected loss (EL)"** is the average estimated loss that a lender can expect from loans that default. It involves three factors: the **probability of default (likelihood of a borrower defaulting)**, **loss given default (portion of the amount the bank is exposed to that can't be recovered in case of default)**, and **exposure at default (potential loss at the time of default, considering the outstanding loan amount and other factors)**.
    - **LendingClub**, operating as a peer-to-peer lending platform, uses a **"PD Model/Credit Scoring Model" to assess borrowers' creditworthiness using credit scores**. This helps determine the **likelihood of loan repayment**, guiding the decision to **approve or deny the loan.** The **required capital to guard against default** for each loan is calculated using **EAD and LGD Models** to estimate the **Expected Loss (EL)**, contributing to minimizing risk in credit operations.
    - When creating a Credit Scoring Model, which assesses creditworthiness for loan approval, using data available at the time of the application is considered an **"application model."** It is distinct from a **"behavior model."** This is the model I will build here.
    - A **"credit policy"** is a set of guidelines that financial institutions follow to evaluate and manage lending risk. Factors such as the expected ROI for each loan application, credit scores, risk classes, expected losses, and so on, are included.
    - **"Return on Investment (ROI)"** is a key measure of loan profitability. Balancing ROI with risk is vital for effective credit policy management. While higher-risk loans may offer more significant potential returns, they also come with a higher chance of default.

# 3. Solution Pipeline
- The **solution pipeline** is based on the **crisp-dm** framework:
    1. Business understanding.
    2. Data understanding.
    3. Data preparation.
    4. Modelling.
    5. Validation.
    6. Deployment.

# 4. Technologies and Toools Used
- Python (Pandas, Numpy, Matplotlib, Seaborn, Sciki-Learn, Statsmodels, Virtual Envs).
- Machine learning classification and regression algorithms.
- Statistics.
- Data cleaning, manipulation, visualization and exploration.

# 5. Project Structure

# 6. Main Business Credit Risk Insights
- Lending Club's current investment portfolio presents the following characteristics:
    - **Personal Indicators:**
        - Approximately 12% are defaulters/bad borrowers.
        - Nearly three out of four loans have a 36-month term.
        - More than 75% have at least 2 years of professional experience, with over 30% having ten years or more.
        - Over 90% own a house through a mortgage or pay rent, while only 8.5% own their houses outright.
        - Nearly 90% have grades ranging from A to D, while grades F and G make up less than 4% of the borrowers.
        - The reason for taking out 80% of the loans is to either consolidate debt or use them for credit card payments.
        - Over 15% live in California.
        - Everything pointed out above suggests a conservative profile among applicants: older individuals with financial and professional stability.


    <img src="reports/personal_indicators.png">


   -  **Financial Indicators:**
        - The maximum funded amount is $35,000, with 50% falling in the range of $8,000 to $20,000. The average is around $14,000.
        - Half of the interest rates range between 11% and 16.8%, with a maximum charge of 26% and a minimum of 5.42%.
        - The average annual income is $72,970, but this value can vary significantly, including individuals with extremely high incomes. It is extremely right-skewed.
        - Half have a debt-to-income ratio up to 16.6%.
        - Half have a credit limit ranging from $13,500 to $37,300. However, similar to annual income, this value can vary significantly, including individuals with extremely high credit limits.
        - Everything pointed out above suggests a conservative investment portfolio, with no high funded amounts or interest rates charged. 


    <img src="reports/financial_indicators.png">


    - **Credit Risk Indicators:**
        - There is a monotonic decrease in default rate as the applicant's grade improves (from G to A). Higher grades correspond to lower credit risk, with the bad rate for G-grade being 6.4 times higher than that for A-grade.
        - The bad rate consistently increases as the interest rate rises, indicating that higher interest rates are associated with higher credit risk. Loans with more than 20% interest rate have a bad rate approximately 8 times higher than those with 5% to 7% interest rates.
        - The bad rate consistently decreases as annual income increases, reflecting that lower annual incomes are associated with higher credit risk. For instance, individuals with annual incomes from 1,748 dollars to 24,111 dollars have a bad rate about two times higher than those with annual incomes of 120,000 dollars or higher. The same pattern holds for the debt-to-income ratio.
    

    <img src="reports/credit_risk_grade.png">


    - There is an observed increasing trend in the number of loans granted over time.
    - Although Lending Club has a conservative portfolio, the default rate is very high, and motivates our project. It needs to manage risks effectively to maximize profit and maintain healthy business. 
    

    <img src="reports/increasing_trend.png">


