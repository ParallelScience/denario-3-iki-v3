1. **Data Preprocessing and Handling "Not Sure" Responses**
    - Load the dataset, mapping '.' to NaN.
    - For ordinal Likert scales (QEA, QEB, QGP, QGU, QKB_1, QGO), map substantive responses to an integer scale (-2 to +2).
    - Treat "Not sure" as a distinct categorical level (e.g., coded as 99) to preserve the "uninformed" state as a unique category for subsequent analysis.
    - Construct "Income Rank" by normalizing income bands within each country (QDA) to create a relative ordinal rank.

2. **Refined Affective Disposition Modeling**
    - Perform Exploratory Factor Analysis (EFA) on the 16 binary `QHD` items using a tetrachoric correlation matrix and Weighted Least Squares (WLSMV) estimation to account for the binary nature of the data.
    - Apply an oblique rotation (e.g., Promax or Oblimin) to extract two correlated factors: "Positive Affect" and "Negative Affect."
    - Validate internal consistency using McDonald’s omega and calculate factor scores for each respondent.

3. **Latent Class Analysis (LCA) for Job Security Trajectories**
    - Perform LCA using `QEA_2` and `QEB_2` as categorical indicators, explicitly including "Not sure" as a valid category level within the LCA model to ensure latent classes reflect the full response distribution.
    - Estimate models with 2–5 classes. Select the optimal model based on BIC, AIC, and Entropy (aiming for > 0.8).
    - Assign each respondent to their most likely latent class (e.g., "Resiliently Optimistic," "Anxiously Declining," "Stagnant Neutral").

4. **Multinomial Logistic Regression with Moderation**
    - Regress latent class membership on organizational enablers (QKB index), culture index (QGO), "Positive Affect," "Negative Affect," job profile (`HIDDG`), sector (`QDB`), and country fixed effects.
    - Include an interaction term between the "Company Culture" index and "Job Level" (`QDG`) to test if cultural impact is moderated by hierarchical position.
    - Use "Stagnant Neutral" as the reference category and report McFadden’s Pseudo-R-squared to assess model goodness-of-fit.

5. **Targeted Moderation Analysis for HR Interventions**
    - Conduct a formal moderation analysis: `Class ~ Training * (Baseline_Uncertainty)`, where `Baseline_Uncertainty` is a dummy variable identifying respondents who selected "Not sure" in the job security items.
    - This step tests whether training interventions are more effective in shifting employees toward the "Resiliently Optimistic" class for those who were initially uncertain.

6. **Robustness and Placebo Testing**
    - Perform 5-fold cross-validation to assess the predictive stability of the multinomial model.
    - Conduct a "placebo test" by regressing class membership on organizational scale variables (`Global Employee Size` and `Market Capitalization`). If these variables show significant effects, include them as control variables to adjust for organizational scale bias.

7. **Marginal Effects Visualization**
    - Calculate and plot the predicted probabilities of class membership across the range of key predictors (e.g., training, involvement, culture).
    - Use these plots to illustrate how specific organizational configurations shift the probability of an employee occupying a positive stability class.

8. **Statistical Significance and Reporting**
    - Apply FDR correction to all p-values.
    - Report odds ratios, 95% confidence intervals, and model fit statistics (Entropy, BIC/AIC, Pseudo-R-squared).
    - Synthesize findings to highlight the interaction between organizational policy and employee psychological state.