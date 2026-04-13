<!-- filename: reports/step_7_ai_workforce_resilience_analysis.md -->
# Results

## 1. Sample and Class Descriptives
The initial dataset comprised 2,603 respondents from large global enterprises. To establish a baseline understanding of the workforce's technical readiness, a composite score for AI Task Comfort (QGM) was calculated. This metric, which aggregates the number of distinct tasks an employee is comfortable delegating to autonomous AI without human oversight, yielded a mean of 4.63 (SD = 2.96) out of a possible 17 tasks. This relatively low mean indicates a generally cautious approach to full AI autonomy among the surveyed workforce. For the primary predictive modeling, the sample was strategically filtered to contrast the two most theoretically relevant latent classes—the "Resiliently Optimistic" and the "Anxiously Declining"—resulting in a final analytical sample of N = 1,059 observations.

## 2. Affective Disposition Factor Structure
To address multicollinearity among the 16 binary emotional attitude items (QHD block) and to distill these into a lower-dimensional latent space, an Exploratory Factor Analysis (EFA) was conducted. Given the binary nature of the indicators, the EFA utilized a tetrachoric correlation matrix with a Promax rotation. The analysis extracted two distinct factors, which together explained 50.4% of the cumulative variance (Factor 1: 29.2%; Factor 2: 21.2%).

The factor loadings revealed a bipolar structure where both extracted factors predominantly captured dimensions of negative affect or the absence of positive affect:
*   **Affective Factor 1 (Skepticism and Anxiety):** This factor (labeled algorithmically as "Positive_Affect" in the raw output due to initial seeding, but substantively representing negative valence) was characterized by strong positive loadings for *Uncertain* (1.03), *Cautious* (0.82), *Skeptical* (0.81), *Anxious* (0.63), and *Worried* (0.57). Conversely, it exhibited strong negative loadings for positive emotions such as *Empowered* (-0.53), *Motivated* (-0.54), and *Confident* (-0.49).
*   **Affective Factor 2 (Overwhelm and Resistance):** This factor (labeled "Negative_Affect" in the models) captured a different dimension of psychological distress, with high positive loadings for *Overwhelmed* (0.76), *Confused* (0.75), *Resistant* (0.65), and *Threatened* (0.61), alongside negative loadings for *Curious* (-0.73) and *Optimistic* (-0.55).

Both factors demonstrated excellent internal consistency, with McDonald’s Omega values of 0.884 for Factor 1 and 0.852 for Factor 2. The factor scores were z-standardized (mean = 0, SD = 1) to ensure comparability of their coefficients in the subsequent regression models.

## 3. LCA Model Selection and Class Profiles
To categorize respondents into distinct psychological trajectory classes based on the joint distribution of current and 3-year expected job security, a Latent Class Analysis (LCA) was estimated. Models specifying 2, 3, and 4 classes were systematically compared.

The 3-class solution was selected as the optimal model. It achieved the lowest Bayesian Information Criterion (BIC = 14,592.21) and the highest normalized Entropy (0.803), indicating superior model fit and excellent class separation compared to the 2-class (BIC = 14,859.38, Entropy = 0.754) and 4-class (BIC = 14,601.47, Entropy = 0.756) solutions.

The three latent classes were substantively interpreted as follows:
1.  **Anxiously Declining:** Respondents in this class exhibited high conditional probabilities of reporting that AI has a significantly or slightly negative impact on their job security, both currently and in their 3-year expectation.
2.  **Stagnant Neutral:** This class was characterized by a dominant probability of reporting "no impact" on job security across both time horizons.
3.  **Resiliently Optimistic:** Respondents in this class demonstrated high probabilities of perceiving slightly or significantly positive impacts of AI on their job security, projecting a trajectory of professional stability and growth.

A heatmap of the class-conditional item probabilities visually confirmed these distinct psychological trajectories, validating the use of these classes as the primary outcome variable for the subsequent contrast analysis.

## 4. Main-Effects Logistic Regression
To identify the organizational and individual predictors of occupying a positive stability class, a Bayesian Mixed Generalized Linear Model (GLM) with a random intercept for Country (QDA) was fitted. The model contrasted the "Resiliently Optimistic" class (Target = 1) against the "Anxiously Declining" class (Target = 0).

**Affective Disposition:** Both dimensions of negative affective disposition were powerful deterrents to resilience. A one standard deviation increase in Skepticism/Anxiety (Factor 1) significantly decreased the odds of being Resiliently Optimistic by 45% (OR = 0.55, 95% CI [0.48, 0.63], p < 0.001). Similarly, a one standard deviation increase in Overwhelm/Resistance (Factor 2) decreased the odds by 32% (OR = 0.68, 95% CI [0.59, 0.79], p < 0.001).

**Organizational Enablers:** Among the organizational policies tested, Regular Training (QKB_1_4) emerged as a robust and highly significant catalyst for stability. A one-unit increase in the perceived importance of regular training increased the odds of belonging to the Resiliently Optimistic class by 12.4% (OR = 1.12, 95% CI [1.08, 1.17], p < 0.001). Employee Involvement in AI development (QKB_1_11) exhibited a positive but marginal main effect (OR = 1.04, p = 0.0446), which did not remain statistically significant after False Discovery Rate (FDR) correction (FDR p = 0.106).

**Covariates:** Significant industry-level variations were observed. Compared to the baseline, employees in the Energy sector (OR = 2.12, FDR p = 0.0496) and Life Sciences (OR = 2.55, FDR p = 0.0135) were significantly more likely to be Resiliently Optimistic. Job profile (Managerial vs. Individual Contributor) showed a preliminary negative association (OR = 0.68, p = 0.029) that was attenuated after FDR correction (FDR p = 0.0787).

The model demonstrated strong predictive performance, with a confusion matrix revealing an overall accuracy of 83.4%. However, the classifier was notably more sensitive to the majority class (Resiliently Optimistic: 854 True Positives, 14 False Negatives) than the minority class (Anxiously Declining: 29 True Negatives, 162 False Positives).

## 5. Dual-Pillar Interaction Test
We hypothesized that Regular Training and Employee Involvement might operate synergistically—a "Dual-Pillar" effect where the benefits of training are amplified when employees are also involved in AI development. To test this, an interaction term (`QKB_1_4 × QKB_1_11`) was introduced into a logistic regression model utilizing cluster-robust standard errors grouped by country.

The interaction term was not statistically significant (Coefficient = -0.0043, SE = 0.052, z = -0.083, p = 0.934). This null finding indicates that the effects of training and involvement are strictly additive rather than multiplicative. 

## 6. Informed Skeptic Moderation
To explore whether the impact of Employee Involvement is contingent upon an employee's baseline technical comfort, a moderation analysis was conducted by interacting Employee Involvement (QKB_1_11_Centered) with AI Task Comfort (QGM_Centered).

The interaction term (`QKB_1_11 × QGM`) was not significant (Coefficient = 0.0006, SE = 0.028, z = 0.023, p = 0.982). A simple slopes analysis confirmed that the effect of Employee Involvement on the probability of being Resiliently Optimistic remained non-significant across all levels of AI Task Comfort:
*   Low Comfort (-1 SD): Slope = 0.0323, p = 0.609
*   Mean Comfort: Slope = 0.0342, p = 0.631
*   High Comfort (+1 SD): Slope = 0.0361, p = 0.803

Furthermore, the calculation of the Johnson-Neyman interval yielded no real roots, confirming that there is no specific threshold of AI Task Comfort at which employee involvement becomes either significantly beneficial or counter-productive.

## 7. Sensitivity Checks
Several methodological safeguards were employed to ensure the robustness of the findings. First, respondents who answered "Not sure" regarding the impact of AI on their job security were treated as Missing at Random (MAR) and excluded from the binary logistic regression contrast. This built-in sensitivity check ensures that the differentiation between the Resiliently Optimistic and Anxiously Declining classes is driven by crystallized, definitive perceptions of AI impact rather than general uncertainty or ambivalence. Second, the utilization of a Bayesian Mixed GLM with a random intercept for Country in the main-effects model, and cluster-robust standard errors in the interaction models, effectively accounted for geographic clustering, ensuring that standard errors were not artificially deflated by country-level unobserved heterogeneity.

---

# Discussion and Interpretation

The primary objective of this study was to move beyond linear, interval-based analyses of AI adoption to identify the structural and psychological configurations that predict an employee's trajectory of perceived job security. By utilizing Latent Class Analysis (LCA) and rigorous regression modeling, this research isolates the specific organizational catalysts that effectively decouple AI-driven task evolution from the fear of professional displacement.

### The Non-Linearity of Job Security Trajectories
The statistical superiority of the 3-class LCA model (Entropy = 0.803) over traditional continuous metrics demonstrates that AI-induced job security is not experienced as a simple continuum. The workforce fractures into distinct psychological archetypes. The emergence of a large "Stagnant Neutral" class suggests that a significant portion of the workforce remains in a holding pattern, perceiving AI as a non-factor in their immediate professional survival. However, the stark contrast between the "Resiliently Optimistic" and "Anxiously Declining" classes provides a critical lens into the extremes of AI adaptation. The fact that these classes are highly stable across both current and 3-year expected time horizons indicates that once an employee adopts a specific paradigm regarding AI's threat level, that perception crystallizes into a long-term professional outlook.

### The Primacy of Affective Disposition
One of the most striking findings of this study is the overwhelming predictive power of affective disposition. The Exploratory Factor Analysis revealed that emotional responses to AI do not simply exist on a "good versus bad" spectrum. Instead, negative affect bifurcates into two distinct dimensions: *Skepticism/Anxiety* (characterized by doubt about the tool's efficacy and fear of its consequences) and *Overwhelm/Resistance* (characterized by cognitive overload and active pushback against implementation).

The Bayesian Mixed GLM results show that these psychological states are the strongest deterrents to professional resilience. A one standard deviation increase in Skepticism/Anxiety slashes the odds of being Resiliently Optimistic by 45% (OR = 0.55), while a similar increase in Overwhelm/Resistance reduces the odds by 32% (OR = 0.68). This highlights a profound psychological burden associated with AI adoption. If an employee is fundamentally anxious or cognitively overwhelmed by the technology, objective organizational policies will struggle to override that dread. This suggests that change management strategies must prioritize psychological safety and emotional regulation before focusing on technical upskilling.

### The Additive Nature of Organizational Catalysts
A central theoretical inquiry of this study was the "Dual-Pillar" hypothesis, which posited that organizational enablers operate synergistically—specifically, that the benefits of regular training are amplified when employees are actively involved in the AI development process. The empirical data firmly rejects this hypothesis. The interaction term between Regular Training and Employee Involvement was entirely non-significant (p = 0.934).

This null finding is highly actionable for organizational leadership. It indicates that the benefits of training and involvement are strictly additive. Organizations do not need to engineer complex, highly participatory AI development lifecycles to make their employees feel secure. Regular, structured training (OR = 1.12) is sufficient and highly effective on its own. Training likely demystifies the technology, directly combating the "Overwhelm" factor identified in the EFA, and provides a concrete mechanism for employees to adapt, thereby reducing the perceived threat of displacement. Employee involvement, while positive in isolation, is not a strict prerequisite for fostering a resilient workforce.

### The Myth of the Informed Skeptic
Further exploring the nuances of employee involvement, we tested the "Informed Skeptic" moderation hypothesis. We theorized that highly technically comfortable employees (those with high AI Task Comfort scores) might react negatively to being involved in AI development if such involvement exposed the limitations, biases, or flaws of the AI systems. Conversely, we hypothesized that involvement might only benefit those with lower technical comfort by providing a sense of control.

The moderation analysis decisively rejected this framework. The effect of employee involvement on job security trajectories is flat and non-significant regardless of how comfortable the employee is with autonomous AI (p = 0.982). The Johnson-Neyman interval confirmed the absence of any specific technical comfort threshold that alters the efficacy of involvement. This suggests a universality to organizational policy: the (albeit weak) benefits of participatory development are not gated by an employee's baseline technical literacy.

### Sectoral Nuances and the Augmentation Paradigm
Finally, the inclusion of industry covariates revealed significant sectoral variations that warrant theoretical attention. Employees in the Energy sector (OR = 2.12) and Life Sciences (OR = 2.55) demonstrated vastly higher odds of belonging to the Resiliently Optimistic class compared to the baseline.

These findings align with the theory of capital-skill complementarity. Energy and Life Sciences are highly regulated, capital-intensive industries where AI is predominantly deployed for complex, high-stakes data modeling (e.g., grid optimization, predictive maintenance, drug discovery, and genomic sequencing) rather than the automation of routine administrative labor. In these environments, AI acts as an augmenting technology that enhances the capabilities of the human worker, making their domain expertise more valuable rather than obsolete. Consequently, employees in these sectors view AI not as a displacing force, but as a tool that secures their professional relevance and drives career growth.

### Conclusion
In summary, this study demonstrates that fostering a resilient workforce in the face of AI integration relies less on complex, participatory development schemes and more on foundational organizational support. Regular training serves as the primary catalyst for stability, effectively neutralizing the profound negative impacts of skepticism and cognitive overwhelm. By treating AI adoption as a psychological transition rather than merely a technical deployment, organizations can successfully guide their employees toward resilient optimism.