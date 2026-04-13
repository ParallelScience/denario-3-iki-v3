1. **Data Refinement and Variable Selection**
   - Process the dataset, treating '.' as NaN. Map all binary variables (QHD, QGM, QF, QED_1, QKC, QGH) to 0/1.
   - Map ordinal variables (QC, QGI, QGS, QGR) to numeric scales. For income (QDF), generate a country-specific ordinal rank.
   - Identify the single item from the `QGO` block with the strongest bivariate correlation with the "Anxiously Declining" class to serve as a proxy for organizational culture.
   - Create a binary "Usage State" variable using `QA_1_1` (Non-users vs. Active users).

2. **Affective Disposition Modeling**
   - Perform EFA on the 16 binary `QHD` items using a tetrachoric correlation matrix.
   - Extract two factors: "Positive Affect" and "Negative Affect." Validate using McDonald’s omega.
   - Z-standardize the resulting factor scores (mean=0, SD=1) to ensure comparability of coefficients in subsequent regression models.

3. **Latent Class Analysis (LCA) Estimation**
   - Estimate the LCA using `QEA_2` and `QEB_2` as categorical indicators.
   - Treat "Not sure" responses as a distinct category in the LCA to capture "AI-Ambivalence." If the model fails to converge, treat "Not sure" as missing data (MAR).
   - Restrict to a 3-class solution ("Resiliently Optimistic," "Anxiously Declining," "Stagnant Neutral") and validate using BIC, AIC, and Entropy.

4. **Binary Logistic Regression: The "Resilience" Contrast**
   - Filter the dataset to include only "Resiliently Optimistic" and "Anxiously Declining" classes.
   - Perform binary logistic regression (1 = Resilient, 0 = Anxious) using z-standardized "Positive Affect," "Negative Affect," the `QGO` proxy, `HIDDG` (job profile), and `QDB` (sector) as predictors.
   - Include a random intercept for `QDA` (Country). Report the confusion matrix to validate classification performance.

5. **Testing the "Dual-Pillar" Hypothesis**
   - Include `QKB_1_4` (Regular Training) and `QKB_1_11` (Employee Involvement) as predictors in the logistic regression.
   - Perform a model comparison: compare the main-effects model against a model including the interaction term (`QKB_1_4 * QKB_1_11`) to test if the "Dual-Pillar" effect is additive or multiplicative.

6. **Moderation Analysis: The "Informed Skeptic" Segment**
   - Create an interaction term between `QKB_1_11` (Employee Involvement) and the `QGM` (AI Task Comfort) composite score.
   - If significant, perform simple slopes analysis and calculate the Johnson-Neyman interval to identify the specific range of `QGM` where involvement becomes counter-productive.

7. **Sensitivity and Robustness Checks**
   - Perform a Chi-square test of independence between the 3-class membership and the "Usage State" (`QA_1_1`) to determine if the "Stagnant Neutral" class is statistically over-represented by non-users.
   - Re-run the logistic regression excluding "Not sure" respondents to ensure findings are not driven by uncertainty.

8. **Marginal Effects and Visualization**
   - Calculate and plot predicted probabilities of "Resiliently Optimistic" membership across levels of "Regular Training" and "Employee Involvement."
   - Report odds ratios with 95% confidence intervals for all predictors, applying FDR correction to all p-values.