

Iteration 0:
# Research Summary: Latent Class Trajectories of AI-Induced Job Security

### 1. Project Overview & Methodology
*   **Objective:** Identify organizational catalysts for professional stability by modeling job security as heterogeneous psychological trajectories rather than a global mean.
*   **Data:** 2,603 employees at $1B+ revenue firms across 9 countries.
*   **Method:** 
    *   **EFA:** Reduced 16 `QHD` emotional items into two factors: *Affective Valence* (bipolar: threat vs. empowerment) and *Cautious Curiosity*.
    *   **LCA:** Segmented respondents into 4 classes based on current (`QEA_2`) and 3-year expected (`QEB_2`) job security: *Resiliently Optimistic* (31.8%), *Stagnant Neutral* (28.7%), *Moderately Optimistic* (26.5%), and *Anxiously Declining* (13.0%).
    *   **Multinomial Logistic Regression:** Predicted class membership using organizational enablers (`QKB`), culture (`QGO`), and affective factors.

### 2. Key Findings
*   **Catalysts for Optimism:** *Employee Involvement* (`QKB_1_11`) and *Regular Training* (`QKB_1_4`) are independent, additive drivers of the "Resiliently Optimistic" class. Simultaneous implementation doubles the probability of positive security perception (from ~20% to ~40%).
*   **Protections against Decline:** *Company Culture* (experimentation, psychological safety) is the primary buffer against the "Anxiously Declining" class. Specific AI policies (training/involvement) failed to significantly differentiate this group from the neutral baseline.
*   **Sensitivity Insight:** Training is most effective for "uncertain" employees; *Involvement* is the superior driver for those with crystallized views.

### 3. Constraints & Limitations
*   **Model Performance:** 5-fold CV accuracy of 38.34% is robust for 4-class psychological prediction but indicates significant unexplained variance.
*   **Missing Data:** "Not sure" responses are non-random; sensitivity analysis confirms that excluding them attenuates the impact of training but strengthens the observed effects of culture and involvement.
*   **Interaction:** No significant multiplicative interaction between training and involvement; effects are additive.

### 4. Recommendations for Future Research
*   **Targeted Interventions:** Future experiments should test if "Involvement" interventions (e.g., co-design workshops) specifically shift employees from "Anxiously Declining" to "Resiliently Optimistic" compared to passive training.
*   **Longitudinal Tracking:** The current LCA is cross-sectional. Future work must validate if these latent classes are stable or if individuals transition between them as AI adoption matures.
*   **Refine Culture Index:** Given the protective power of culture, decompose the `QGO` index to identify which specific cultural element (e.g., "learning from mistakes" vs. "speaking candidly") is the strongest moderator of job threat.
        