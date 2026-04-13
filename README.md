# denario-3-iki-v3

**Scientist:** denario-3
**Date:** 2026-04-13

# IKI-Data-Raw: Workplace AI Adoption Survey — Job Security & Nature of Work Analysis

## Research Focus

This project investigates which individual, organisational, and AI-adoption variables predict (a) perceived **job security** and (b) **changing nature of work** among employees at large companies ($1B+ revenue) across 9 countries. The aim is to go beyond bivariate correlations and identify robust predictors using regression, random forests, and mediation analysis.

## File

- **Path:** `/home/node/work/projects/iki_v2/IKI-Data-Raw.csv`
- **Format:** Tab-separated values (load with `sep='\t'`, `low_memory=False`)
- **Shape:** 2,603 rows (respondents) × 234 columns (variables)
- **Row identifier:** `Respid` (integer, unique per respondent)
- **Missing values:** Coded as `.` (period). Treat `.` as NaN.

---

## Primary Outcome Variables

### Job Security
- `QEA_2` — "Security of your current role: What impact does AI currently have?" Scale: Significantly negative / Slightly negative / No impact / Slightly positive / Significantly positive / Not sure. Distribution: No impact 31%, Slightly positive 27%, Significantly positive 26%, Slightly negative 12%, Significantly negative 3%.
- `QEB_2` — Same question, 3-year expected impact. Distribution: Slightly positive 28%, Significantly positive 27%, No impact 26%, Slightly negative 13%, Significantly negative 4%.
- `QGR` — "How confident are you that your company will protect jobs as AI becomes more capable?" Scale: Not at all confident / Slightly confident / Somewhat confident / Very confident / Not sure. Distribution: ~46% Somewhat confident, ~29% Very confident.
- `QF_3` — Binary: "I am worried about losing my job" (among deterrents to AI use). 1 = Yes.

### Changing Nature of Work (past 3 years)
- `QGP_1` — AI's impact on *repetitive work* volume (Large decrease → Large increase, 5-level).
- `QGP_2` — AI's impact on *creative work* volume.
- `QGP_3` — AI's impact on *complex work* volume.
- `QGH1_1` — Binary: no changes to team roles to date.
- `QGH1_2` — Binary: roles on team have changed to date.

### Changing Nature of Work (3-year expectation)
- `QGU_1` — Expected AI impact on *repetitive work* volume.
- `QGU_2` — Expected AI impact on *creative work* volume.
- `QGU_3` — Expected AI impact on *complex work* volume.
- `QGH2_1` — Binary: no changes to team roles expected.
- `QGH2_2` — Binary: team roles will change in next 3 years.

---

## Predictor / Feature Variables

### Demographics
- `QDA` — country of work (9 countries)
- `QDB` — industry (13 sectors)
- `QDC` — age (integer, 18–69)
- `HidQDC` — generational cohort: Gen Z, Millennial, GenX, Boomer
- `QDD` — education level (ordinal, 7 levels)
- `QDE_Year` — years of professional experience (integer)
- `QDH` — department (IT, HR, Finance, Sales/Marketing, etc.)
- `QDG` — job level (11 levels, ordinal from Intern to C-suite)
- `HIDDG` — job profile (Managerial / Senior / Individual contributor)
- `QDF` — income band (country-specific local currency; use as ordinal rank within country)

### Company characteristics
- `Global Employee Size` — ordinal band (6 levels)
- `Global Annual Revenue` — ordinal band
- `Market Capitalization` — ordinal band

### AI adoption and usage
- `QC` — AI use frequency (7-level ordinal: Once a month or fewer → Many times a day)
- `QGI` — hours saved per week by AI (ordinal: increases time, None, <1h, 1–3h, 3–5h, >5h)
- `QGS` — AI output error frequency (5-level ordinal: Never → Constantly)
- `QGG` — AI disclosure behaviour (5 categories: accurately promote, accurately w/ drawbacks, overstate, downplay, prefer not to say)
- `QKD` — company has deployed agentic AI: Yes/No/Not sure
- `QGN` — company AI posture (5 levels: doesn't use → transforming most processes)
- `QA_1_1` — binary: "I do not currently use AI tools at work"

### AI emotional attitudes (QHD block, 16 binary Yes/No)
Optimistic, Empowered, Motivated, Excited, Curious, Confident, Indifferent, Uncertain, Cautious, Skeptical, Worried, Resistant, Confused, Overwhelmed, Threatened, Anxious.
Column format: `QHD_<N>: <Emotion> - Which of the following describes how you feel about using AI at work?`

### AI task comfort — autonomous AI (QGM block, 17 binary)
Whether respondent is comfortable with AI completing tasks autonomously (Yes/No): Summarize, Search, Content creation, Customer service, Internal comms, Research, Predictive analysis, Other analysis, Optimizing processes, QA, Software development, Product development, Automating routine tasks, Schedules/emails, Managing workflows, Monitoring performance.

### Perceived AI impact — other dimensions (QEA/QEB — excluding the job security item already listed)
Current impact ratings (QEA_1, QEA_3–QEA_15) and 3-year expected ratings (QEB_1, QEB_3–QEB_15) for: Career growth, Time on tasks, AI output quality, Own work quality, Output volume, Cost of output, Effort required, High-value work time, Accuracy, Accountability, Job satisfaction, Personal agency, Communication, New ideas.

### Company enablers (QKB block)
11 items × 2 dimensions: importance rating (QKB_1_1–11, 5-level) and whether company does each (QKB_2_1–11, 5-level).
Items: clear AI strategy, guidelines on tool use, encourage experimentation, regular training, tool access, safeguards, leadership support, incentives, freedom to choose tools, freedom on when to use AI, employee involvement in development.

### Deterrents to AI use (QF block, 13 binary)
- `QF_1` — No access to AI tools
- `QF_2` — Don't know how to use AI tools
- `QF_3` — Worried about losing job
- `QF_4` — Fear of accountability
- `QF_5` — Creates more work (needs checking)
- `QF_6` — Distrust of AI accuracy
- `QF_7` — Privacy/data safety concerns
- `QF_8` — Doesn't fit work processes
- `QF_9` — Don't see benefit
- `QF_10` — Moral reservations
- `QF_11` — Colleague resistance
- `QF_13` — Nothing deters me

### AI incentives (QKC block, 8 binary)
Financial reward, Peer recognition, Learning benefits, Career progression, No incentives (not required), No incentives (required), Discourages AI.

### Company culture (QGO block, 5 Likert items)
Encourages experimentation, learning from mistakes, speaking candidly, asking for help, implementing AI ethically. Scale: Strongly disagree → Strongly agree.

### Company AI goals (QED block)
`QED_1_1`–`QED_1_7` binary: company has goal of completing tasks faster, completing more tasks, reducing costs, reducing errors, freeing up time for high-value work, no goals set, don't know.

---

## Encoding Notes

- **Binary columns** (QHD, QGM, QA_1, QGL, QF, QED_1, QKC, QGH): Values are `Yes` / `No` / `Not Selected`. Map Yes → 1, else → 0.
- **Ordinal Likert columns** (QEA, QEB, QGP, QGU, QKB_1, QGO): stored as strings. Define explicit ordered mappings to integers (e.g. Significantly negative = −2 → Significantly positive = +2; Not sure = NaN).
- **QGP/QGU**: Values are "Large decrease", "Moderate decrease", "No major change", "Moderate increase", "Large increase" (map −2 to +2).
- **QC** (frequency): Map to 1–7 integers (Once a month or fewer → Many times a day).
- **QGI** (hours saved): Map ordinal including the "AI increases time" category (= −1 or 0), None = 0, <1h = 1, etc.
- **QGS** (error frequency): Map Never = 0 → Constantly = 4.
- **QGR** (job protection confidence): Map Not at all = 1 → Very confident = 4, Not sure = NaN.
- **Income (QDF_*)**: Only one column populated per respondent. Build a single ordinal income rank within each country's band (lowest band = 1, highest = max).

---

## Suggested Analysis Plan

1. **Construct composite outcome indices:**
   - *Job Security Index (current)*: encode QEA_2 as −2 to +2.
   - *Job Security Index (future)*: encode QEB_2 as −2 to +2.
   - *Nature of Work Change Index*: average of QGP_1, QGP_2, QGP_3 (encoded −2 to +2).
   - *Future Nature of Work Change Index*: average of QGU_1, QGU_2, QGU_3.

2. **Bivariate correlations:** Spearman rank correlations between each predictor and each outcome. Report the top 20 predictors per outcome with effect sizes and FDR-adjusted p-values.

3. **OLS / Ordinal regression:** Regress each job security outcome on demographics + AI usage + attitudes + organisational factors. Report standardised beta coefficients.

4. **Random Forest feature importance:** Train RF regressor (or classifier) on the full predictor set. Report permutation importance top 20. Cross-validate with 5-fold CV.

5. **Subgroup comparisons:** Compare job security perceptions by generation (Gen Z / Millennial / GenX / Boomer), job level (individual contributor / manager / senior), industry, and country.

6. **Mediation sketch:** Test whether AI emotional attitudes (anxiety, threat) mediate the relationship between AI usage frequency and job security.
