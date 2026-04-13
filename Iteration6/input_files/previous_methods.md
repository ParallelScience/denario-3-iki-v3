1. **Data Cleaning and Harmonization**
   - Load the dataset, treating '.' as NaN.
   - Recode binary variables (QHD, QGM, QF, QED_1, QKC, QGH) to 0/1.
   - Map ordinal Likert scales (QEA, QEB, QGP, QGU, QKB_1, QGO) to numeric integers (-2 to +2).
   - Create three distinct "Task Transformation Indices" (Repetitive, Creative, Complex) by averaging the respective change scores (QGP_1–3 for past; QGU_1–3 for future).
   - Standardize continuous predictors (age, experience, AI frequency) and create country-specific ordinal income ranks.

2. **Handling "Not Sure" and Missing Data**
   - Perform a logistic regression to predict the probability of being "Not sure" regarding job security (QEA_2/QEB_2) based on demographics and AI usage.
   - Compare the mean "Organizational Support Index" scores between the "Not sure" group and the rest of the sample to determine if uncertainty is a proxy for poor organizational communication.
   - Conduct a sensitivity analysis by comparing regression coefficients of the main model against a model where "Not sure" is treated as a separate dummy variable to assess if it represents a distinct "Uninformed" segment.

3. **Affective Disposition Dimensionality Reduction**
   - Conduct Exploratory Factor Analysis (EFA) on the 16 binary `QHD` items using a tetrachoric correlation matrix.
   - Extract two latent factors: "Positive Affect" (e.g., excited, empowered) and "Negative Affect" (e.g., worried, threatened).
   - Calculate McDonald’s omega to verify internal consistency and retain factor scores as individual-level covariates.

4. **Structural Configuration of Organizational Enablers**
   - Perform a Principal Component Analysis (PCA) on the 11 `QKB_1` (importance) and `QKB_2` (actual practice) items to reduce them into a single "Organizational Support Index."
   - Retain the index as the primary predictor, while preparing to test the interaction between `QKB_1_4` (Regular Training) and `QKB_1_11` (Employee Involvement) as a secondary, granular analysis.

5. **Task-Level Mediation Analysis**
   - Construct a mediation model (SEM) where the Organizational Support Index influences Job Security (QEA_2/QEB_2) through the three Task Transformation Indices.
   - Frame the mediation path to test whether organizational support facilitates a positive perception of task augmentation (e.g., increases in creative/complex work) which subsequently predicts higher job security.

6. **Agentic AI Moderation**
   - Use `QKD` (deployment of agentic AI) as a categorical moderator in the regression models.
   - Include the main effects of both the Organizational Support Index and the Task Transformation Indices alongside the `QKD` interaction term to ensure unbiased estimates.

7. **Regression Modeling**
   - Perform Ordinal Logistic Regression to predict job security (QEA_2 and QEB_2) using the Organizational Support Index, Task Transformation Indices, Affective Disposition scores, and the `QKD` interaction term.
   - Include a quadratic term (Task Transformation Index squared) to test for non-linear (U-shaped) relationships between task change and job security.
   - Include `HIDDG` (job profile) and `QDB` (sector) as control variables, keeping the model parsimonious to ensure interpretability.

8. **Sensitivity and Robustness Checks**
   - Calculate Variance Inflation Factors (VIF) to monitor multicollinearity.
   - Report standardized coefficients, 95% confidence intervals, and FDR-adjusted p-values.
   - Finalize the robustness check by comparing the mediation model results with and without the "Uninformed" (Not sure) group to ensure findings are not driven by sample exclusion.