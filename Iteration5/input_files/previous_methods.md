1. **Data Preprocessing and Feature Engineering**
   - Clean the dataset by converting '.' to NaN. Map binary variables (QHD, QGM, QF, QED_1, QKC, QGH) to 0/1.
   - Map ordinal Likert scales (QEA, QEB, QGP, QGU, QKB_1, QGO) to numeric integers (-2 to +2).
   - Generate a country-specific ordinal income rank (QDF). Standardize continuous predictors (age, experience, AI frequency).
   - Define "High-Complexity" sectors (IT, Life Sciences, Finance) vs. "Low-Complexity" sectors for subsequent subgroup analysis.

2. **Affective Disposition Modeling**
   - Conduct Exploratory Factor Analysis (EFA) on the 16 binary `QHD` items using a tetrachoric correlation matrix.
   - Extract two factors: "Positive Affect" and "Negative Affect." Calculate McDonald’s omega to ensure internal consistency.
   - Retain these factor scores as control variables (covariates) in the multinomial regression to account for baseline psychological temperament.

3. **Latent Class Analysis (LCA) Estimation**
   - Estimate a 3-class LCA model using `QEA_2` and `QEB_2` as nominal categorical indicators to identify "Resiliently Optimistic," "Anxiously Declining," and "Stagnant Neutral" segments.
   - Treat "Not sure" responses as a distinct nominal category within the LCA to prevent distortion of the latent distance between other classes.
   - Validate the model using BIC, AIC, and Entropy metrics.

4. **Multinomial Logistic Regression: Class Membership Analysis**
   - Perform a multinomial logistic regression to predict class membership (using "Stagnant Neutral" as the reference category).
   - Include "Positive Affect," "Negative Affect," `HIDDG` (job profile), and `QDB` (sector) as control covariates.
   - Incorporate `QKB_1_4` (Regular Training) and `QKB_1_11` (Employee Involvement) as primary predictors.
   - Check Variance Inflation Factor (VIF) for `QKB_1_4` and `QKB_1_11` to ensure multicollinearity does not destabilize the model.

5. **Testing the "Dual-Pillar" Interaction**
   - Include an interaction term (`QKB_1_4 * QKB_1_11`) in the multinomial model.
   - Calculate the "policy lift" by computing predicted probabilities of class membership at specific combinations (Low/Low, High/Low, Low/High, High/High) to quantify the impact of organizational support.

6. **Sectoral Complexity and Demographic Diagnostics**
   - Conduct a multi-group analysis or include interaction terms between sector complexity and organizational enablers to determine if the "Dual-Pillar" effect varies by industry.
   - Perform a diagnostic check to see if the "Not sure" class is disproportionately composed of specific demographics (e.g., job level, department) to identify organizational disconnects.

7. **Sensitivity and Mediation Analysis**
   - Test whether "Affective Disposition" factors mediate the relationship between AI usage frequency (`QC`) and job security.
   - Conduct a "Full Information" sensitivity check by comparing the coefficients of the main model (including "Not sure" as a category) against a model where "Not sure" respondents are excluded to ensure robustness.

8. **Marginal Effects and Visualization**
   - Calculate and plot the Average Marginal Effects (AMEs) for the "Dual-Pillar" predictors on the probability of belonging to the "Resiliently Optimistic" class.
   - Report odds ratios with 95% confidence intervals for all predictors, applying FDR correction to all p-values to ensure statistical rigor.