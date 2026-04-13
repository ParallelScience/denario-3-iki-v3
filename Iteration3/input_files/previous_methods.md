1. **Data Preprocessing and Feature Engineering**
    - Load the dataset, mapping '.' to NaN. Create a binary indicator for "Not sure" responses for `QEA_2` and `QEB_2`.
    - Construct "Income Rank" by normalizing income bands within each country (`QDA`).
    - Map ordinal variables (`QC`, `QGI`, `QGS`, `QGR`) to numeric scales and binary variables (`QHD`, `QGM`, `QF`, `QED_1`, `QKC`, `QGH`) to 0/1.
    - Create a composite "Company Culture" index (`QGO_Index`) using Likert items. Create an "Organizational Enablers" index (`QKB_Index`) by averaging items, ensuring internal consistency via Cronbach’s alpha. To avoid multicollinearity in later steps, create a secondary `QKB_Index_Reduced` that excludes the specific "Regular Training" (`QKB_1_4`) and "Employee Involvement" (`QKB_1_11`) items.

2. **Affective Disposition Modeling**
    - Perform EFA on the 16 binary `QHD` items using a tetrachoric correlation matrix and WLSMV estimation.
    - Extract two factors: "Positive Affect" and "Negative Affect" using oblique rotation. Validate using McDonald’s omega.
    - Check the correlation between factors; if highly negatively correlated, treat them as a single "AI Sentiment" spectrum. Drop items with low communality or cross-loadings to ensure distinct factors.

3. **Latent Class Analysis (LCA) of Job Security Trajectories**
    - Perform LCA using `QEA_2` and `QEB_2` as categorical indicators. Include "Not sure" as a distinct category.
    - Estimate models with 2–5 classes. Select the optimal model based on BIC, AIC, and Entropy (> 0.8). Compare these fit indices against a model treating "Not sure" as missing data to determine if "Uninformed" is a distinct psychological state.
    - Assign each respondent to their most likely latent class.

4. **Post-hoc Characterization of Classes**
    - Compare classes using ANOVA (continuous) and Chi-square (categorical) on `QC` (AI usage), `QGN` (AI posture), and `QED_1` (Company AI goals).
    - Calculate effect sizes (Cohen’s d or Eta-squared) to determine if the "Stagnant Neutral" class is "unexposed" (low usage/no goals) or simply "inert."

5. **Multinomial Logistic Regression with Random Intercepts**
    - Regress latent class membership on `QKB_Index_Reduced`, `QGO_Index`, "Positive Affect," "Negative Affect," `HIDDG`, and `QDB`.
    - Include a random intercept for `QDA` (Country) to account for cross-national variations.
    - Check VIFs to ensure no multicollinearity. Report McFadden’s Pseudo-R-squared.

6. **Mediation and Moderation Analysis**
    - Use a Generalized Structural Equation Model (GSEM) with a multinomial logit link to test if `QKB_Index_Reduced` mediates the effect of "Regular Training" (`QKB_1_4`) on class membership.
    - Include a path from `QKB_Index_Reduced` to "Negative Affect" to test if enablers act as a buffer against anxiety.
    - Control for "Prior AI Exposure" (`QC` or `QGN`) to ensure the indirect effect is not confounded by company maturity.

7. **Sensitivity Analysis**
    - Re-run the LCA excluding "Not sure" respondents; compare class proportions and item-response probabilities against the original model to assess stability.
    - Within the "Anxiously Declining" class, regress "Negative Affect" on "Employee Involvement" (`QKB_1_11`) to test for buffering effects.

8. **Marginal Effects and Visualization**
    - Calculate and plot predicted probabilities of class membership across ranges of `QKB_Index` and `QGO_Index`.
    - Visualize the interaction between "Employee Involvement" and "Regular Training" regarding the probability of "Resiliently Optimistic" class membership.
    - Apply FDR correction to all p-values and report odds ratios with 95% confidence intervals. Discuss country-level dominance in specific classes as a proxy for national labor market context.