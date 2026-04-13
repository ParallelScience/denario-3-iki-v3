1. **Data Preprocessing and Feature Engineering**:
   - Load the dataset, treating '.' as NaN.
   - Map binary columns (QHD, QGM, QF, QED_1, QKC, QGH) to 0/1.
   - Map ordinal Likert scales (QEA, QEB, QGP, QGU, QKB_1, QGO) to integer scales (-2 to +2). Map frequency/duration variables (QC, QGI, QGS) to integer ranks.
   - Construct "Income Rank" by normalizing income bands within each country (QDA).
   - Create composite indices for organizational enablers (QKB) and company culture (QGO) by averaging relevant Likert items.
   - Retain "Not sure" responses as a distinct category or indicator variable for sensitivity analysis rather than immediate exclusion.

2. **Dimensionality Reduction for Affective Disposition**:
   - Assess suitability for dimensionality reduction using the Kaiser-Meyer-Olkin (KMO) measure and Bartlett’s test of sphericity.
   - Perform Exploratory Factor Analysis (EFA) on the 16 binary `QHD` items. If multi-dimensional, extract two factors (e.g., "Positive Affect" and "Negative Affect") instead of one.
   - Validate the internal consistency of extracted factors using Cronbach’s alpha or McDonald’s omega.

3. **Latent Class Analysis (LCA) for Job Security Trajectories**:
   - Perform LCA using `QEA_2` and `QEB_2` as ordinal indicators.
   - Estimate models with 2–5 classes. Select the optimal model based on BIC, AIC, and Entropy (aiming for > 0.8).
   - Assign each respondent to their most likely latent class (e.g., "Resiliently Optimistic," "Anxiously Declining," "Stagnant Neutral").

4. **Multicollinearity and Control Setup**:
   - Calculate Variance Inflation Factors (VIF) for all predictors, specifically checking for high correlation between composite indices (QKB, QGO) and their constituent items (e.g., `QKB_1_11`, `QKB_1_4`).
   - If multicollinearity is high, use residuals of items after regressing out the index, or use items directly.
   - Prepare the model to include `QDA` (Country) as a fixed effect to account for nested cultural/economic variance.

5. **Multinomial Logistic Regression**:
   - Regress latent class membership on organizational enablers, culture index, "Affective Disposition" factor(s), job profile (`HIDDG`), sector (`QDB`), and country fixed effects.
   - Use "Stagnant Neutral" as the reference category to interpret odds ratios for other classes.

6. **Interaction Analysis for Organizational Catalysts**:
   - Introduce an interaction term between "employee involvement in development" (`QKB_1_11`) and "regular training" (`QKB_1_4`).
   - Calculate and visualize marginal effects to determine how the probability of belonging to the "Resiliently Optimistic" class shifts based on these variables.

7. **Robustness and Sensitivity Analysis**:
   - Perform 5-fold cross-validation to assess predictive stability.
   - Conduct sensitivity analysis by comparing models that exclude "Not sure" responses against models that treat "Not sure" as a distinct category or use Multiple Imputation (MI) to ensure results are not biased by missing data patterns.

8. **Statistical Significance and Reporting**:
   - Apply FDR correction to all p-values.
   - Report standardized coefficients, odds ratios, 95% confidence intervals, and model fit statistics (Entropy, BIC/AIC).
   - Generate probability plots for interaction effects.