1. **Data Preprocessing and Feature Engineering**
   - Load the dataset, treating '.' as NaN. Map binary variables (QHD, QGM, QF, QED_1, QKC, QGH) to 0/1.
   - Map ordinal Likert scales (QEA, QEB, QGP, QGU, QKB_1, QGO) to numeric integers (-2 to +2).
   - Retain individual task transformation items (Repetitive, Creative, Complex) as separate variables rather than a single index to capture distinct displacement vs. augmentation dynamics.
   - Standardize continuous predictors (age, experience, AI frequency) and create country-specific ordinal income ranks.

2. **Latent Class Trajectory Modeling**
   - Perform Latent Class Analysis (LCA) on the joint distribution of current (`QEA_2`) and 3-year expected (`QEB_2`) job security.
   - Treat "Not sure" as a distinct nominal category within the LCA to capture the "Information Gap" as a latent state.
   - Determine the optimal number of classes using BIC and entropy scores. Validate class composition by performing Chi-square tests against `HIDDG` (Job Profile) and `QDB` (Industry) to ensure classes represent psychological trajectories rather than structural artifacts.

3. **Affective Disposition Dimensionality Reduction**
   - Conduct Exploratory Factor Analysis (EFA) on the 16 binary `QHD` items using a tetrachoric correlation matrix.
   - Extract two latent factors: "Positive Affect" and "Negative Affect."
   - Calculate McDonald’s omega for internal consistency and retain factor scores as continuous covariates. Check for high correlations between "Negative Affect" and the `QF` (Deterrents) block to prevent multicollinearity in later models.

4. **Granular Policy Identification via Elastic Net**
   - Use Elastic Net regression to identify sparse, robust drivers of job security from the 11 `QKB` items.
   - Run models separately by job level (`HIDDG`) and sector (`QDB`) to determine if policy levers are universal or context-dependent.
   - Use cross-validation to select optimal penalty parameters, ensuring the model identifies the most impactful policy levers while controlling for baseline demographics.

5. **Moderation Analysis: The Fragility of Support**
   - Incorporate the `QF` (Deterrents) block as binary moderators.
   - Test for interaction effects between the top organizational enablers (from Step 4) and high-impact deterrents (e.g., `QF_4` accountability, `QF_7` privacy).
   - Determine if the positive impact of organizational support is significantly dampened by the presence of these specific deterrents.

6. **Multinomial Logistic Regression**
   - Regress latent class membership on organizational enablers, Affective Disposition scores, deterrents, and the individual task transformation items (Repetitive, Creative, Complex).
   - Set "Anxiously Declining" as the reference category to make the interpretation of odds ratios for "Resiliently Optimistic" and "Information Gap" states intuitive.
   - Control for job level (`QDG`) and industry (`QDB`) to ensure observed effects are not artifacts of seniority or sector-specific AI adoption.

7. **Policy Simulation and ROI Visualization**
   - Using the coefficients from the multinomial model, calculate the predicted probability of an employee falling into the "Resiliently Optimistic" class under four scenarios: (1) No training/No involvement, (2) Training only, (3) Involvement only, (4) Both.
   - Generate probability plots to visualize the "ROI of Policy," demonstrating the marginal utility of combining training with employee involvement.

8. **Robustness and Sensitivity Checks**
   - Calculate Variance Inflation Factors (VIF) to ensure the inclusion of interaction terms does not introduce excessive multicollinearity.
   - Perform a sensitivity analysis by comparing the multinomial model results against a model excluding the "Not sure" group to verify findings are not driven solely by the "Passive Disengagement" segment.
   - Report FDR-adjusted p-values and 95% confidence intervals for all primary predictors.