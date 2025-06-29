Credit Scoring Business Understanding

1. How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Capital Accord requires financial institutions to measure and manage credit risk rigorously and transparently. A core principle is that banks must demonstrate that their credit risk models are sound, stable, and validated, especially if they want to use them for regulatory capital calculation (Internal Ratings-Based Approach).

This regulatory emphasis means that models must be interpretable: stakeholders, auditors, and regulators need to understand how risk scores are produced. A “black-box” model could create compliance challenges and expose the bank to regulatory penalties or reputational damage if it cannot explain decisions to customers or regulators. Well-documented assumptions, feature choices, and transformation steps (like Weight of Evidence coding) provide the audit trail needed to ensure the model aligns with Basel II’s pillars of minimum capital requirements, supervisory review, and market discipline.

A clear, interpretable, and well-documented model builds trust, enables validation, and supports regulatory reporting.

\*_References:_

- Basel II Capital Accord Summary (Risk-Officer) (https://www.risk-officer.com/Credit_Risk.htm)
- Corporate Finance Institute — Credit Risk (https://corporatefinanceinstitute.com/resources/commercial-lending/credit-risk/)


2.  Since we lack a direct “default” label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

In our case, the provided eCommerce dataset does not directly record whether a customer defaulted on repayments (e.g., failed to repay a loan). To train a supervised model, we need a target variable. Therefore, we must create a proxy variable that approximates default risk using observable behavioral data — for example, high RFM (Recency, Frequency, Monetary) risk profiles, missed payments, or suspicious spending patterns.

However, proxy variables are inherently imperfect representations of true default behavior. They introduce modeling uncertainty:

- If the proxy is poorly designed, the model may classify reliable customers as high-risk (false positives) or miss actual defaulters (false negatives).
- Business risks include lost revenue opportunities, unfairly rejecting good customers, or increased losses if risky customers are misclassified.
- Regulators may scrutinize such proxies for bias, requiring evidence that the model is fair and does not lead to discriminatory outcomes.

When no direct default label exists, a thoughtfully designed proxy is crucial — but it must be continuously validated and improved as more actual loan performance data becomes available. \*_References:_
-World Bank — Credit Scoring Approaches (https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf)

- HKMA — Alternative Credit Scoring (https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf)

3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

Simple, interpretable models (e.g., Logistic Regression with Weight of Evidence (WoE)): 
\*_Pros_  
-High transparency: easy to explain feature impacts and scoring rules  
-Well-established in banking and credit scoring; easier to justify to regulators  
-Faster to implement and validate  
-Lower risk of hidden bias

\*_Cons_

- May underperform on complex, non-linear relationships in the data

Complex, high-performance models (e.g., Gradient Boosting, Random Forests):

\*_Pros_

-Often yield higher predictive accuracy and capture subtle patterns  
-Can improve profit margins by better distinguishing good vs. bad borrowers

\*_Cons_

-Harder to interpret and explain to non-technical stakeholders  
-Risk of overfitting if not carefully tuned and monitored  
-May require additional model governance, local interpretable techniques (e.g., SHAP or LIME), and clear documentation to meet Basel II requirements

In regulated environments like banking, the trade-off must balance predictive performance with compliance and explainability. Many institutions use simple models as baseline production models and explore complex models as challenger models, combining the strengths of both approaches.

Accuracy matters, but interpretability and regulatory compliance often take precedence when lives and livelihoods depend on fair lending decisions.

\*_References:_

- Statistica Sinica — Statistical Methods for Credit Scoring (https://www3.stat.sinica.edu.tw/statistica/oldpdf/A28n535.pdf)  
  -Towards Data Science — How to Develop a Credit Risk Model (https://towardsdatascience.com/how-to-develop-a-credit-risk-model-and-scorecard-91335fc01f03)
