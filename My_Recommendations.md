# My Genuine ML Project Recommendations

*My own suggestions, without a structured framework — just honest advice.*

---

## The Honest Starting Point

Most master's ML projects fall into one of two traps: they either pick something too safe (house price prediction, Titanic survival) and produce a tutorial rehash, or they overreach into something that needs institutional data access and stall before they even begin.

The sweet spot is a project where the **dataset is genuinely interesting on its own**, the **research question isn't already answered by a thousand Medium articles**, and the **ML methods you learn are transferable** to real work.

Here are four projects I would genuinely suggest. I have a clear favourite at the end.

---

## Option 1 — Food Product Quality Analysis with Open Food Facts

**What it is:**
Open Food Facts is a crowdsourced, open database of food products from around the world — nutritional information, ingredients, additives, packaging, labels, and the official Nutri-Score (a standardised nutrition grade from A to E). It has over 3 million products contributed by volunteers globally.

**Why it's interesting:**
Almost nobody uses this dataset for a master's project, which immediately differentiates your work. Ultra-processed food consumption is one of the most significant and under-addressed public health issues of the last two decades. A project that predicts Nutri-Score from raw nutritional values, clusters food products into processing archetypes, or identifies which ingredient combinations are systematically mislabelled — that is a real research contribution.

**What you could actually do:**
- Predict Nutri-Score grade (A–E) from nutritional values using classification — a meaningful multi-class problem with real consequences if wrong
- Cluster products by processing level (NOVA classification: unprocessed vs ultra-processed) using unsupervised methods
- Identify inconsistencies: products labelled "healthy" that cluster with ultra-processed items
- NLP extension: parse ingredient lists to extract additives and examine their relationship to Nutri-Score

**Dataset:**
- URL: https://world.openfoodfacts.org/data
- Format: CSV, MongoDB dump, or Parquet (your choice)
- Size: 3M+ products; you would realistically work with a filtered subset (e.g. UK or Canadian products) of ~100K–500K rows
- Access: Free, no registration, open database license (ODbL)
- Known issues: User-contributed data is noisy — missing values are common, nutritional fields are inconsistent. This is a feature, not a bug: data cleaning at scale is a real skill.

**The research angle that would stand out:**
Rather than just predicting Nutri-Score, ask whether the Nutri-Score algorithm itself is consistent — are there food product clusters where human-assigned Nutri-Score grades disagree with what a model trained on nutritional composition would predict? That gap is scientifically interesting.

---

## Option 2 — Scientific Paper Retraction Prediction

**What it is:**
Retraction Watch maintains a database of scientific papers that were retracted — withdrawn after publication due to fraud, error, or misconduct. Combined with metadata from arXiv or PubMed, you can study whether early signals in a paper's writing, citation network, or journal context predict eventual retraction.

**Why it's interesting:**
Research integrity is a growing crisis in science. This project sits at the intersection of NLP, network analysis, and responsible AI — and almost no student projects touch it. The question "can we identify likely retractions before they happen?" has direct real-world value for peer reviewers, institutions, and funding bodies.

**What you could actually do:**
- Binary classification: retracted vs non-retracted paper, using features from title/abstract text, author metadata, journal impact factor, and citation count
- NLP: compare linguistic patterns in retracted vs non-retracted papers (hedging language, claim certainty, unusual statistical phrasing)
- Temporal analysis: how quickly are papers cited after publication, and does velocity predict retraction?

**Dataset:**
- Retraction Watch database: https://retractionwatch.com/retraction-watch-database-user-guide/ — downloadable CSV, free for non-commercial research (requires email registration)
- Supplementary: arXiv dataset (https://www.kaggle.com/datasets/Cornell-University/arxiv) — 1.7M papers, free with Kaggle account, rich metadata
- OR: PubMed API (https://www.ncbi.nlm.nih.gov/home/develop/api/) — free, no registration required
- Known limitation: Retracted papers are a small minority — class imbalance problem similar to fraud detection

**The research angle that would stand out:**
Focus on a specific field (e.g. psychology or medicine where retraction rates are relatively higher) and ask whether linguistic uncertainty markers in the abstract correlate with retraction reason. This gives you an NLP + classification + domain-specific angle in one project.

---

## Option 3 — Stack Overflow Developer Survey: Burnout and Job Satisfaction

**What it is:**
Stack Overflow publishes its annual developer survey publicly every year. The 2023 survey had over 90,000 respondents across 185 countries. Most people use this data for salary prediction — that angle is exhausted. But the survey contains detailed questions about mental health, work-life balance, job satisfaction, remote work preferences, and career trajectories that are almost never the subject of a serious ML project.

**Why it's interesting:**
Developer burnout and retention are billion-dollar problems for the tech industry. A project that asks "what combination of work environment, compensation, and role factors best predicts developer job satisfaction — and does this differ by country, experience level, or gender?" is both technically interesting and genuinely useful.

**What you could actually do:**
- Classification or regression: predict job satisfaction score from role, environment, and demographic features
- Clustering: identify distinct developer profiles (high-satisfaction, burnout-risk, career-transitioning) and characterise each cluster
- NLP on free-text responses: what language do satisfied vs dissatisfied developers use when describing their work?
- Fairness analysis: do the predictors of job satisfaction differ systematically by gender or country?

**Dataset:**
- URL: https://insights.stackoverflow.com/survey
- Format: CSV, free download, no registration
- Size: ~90,000 rows (2023); ~70,000+ in prior years
- License: Open Database License (ODbL)
- Known issues: Self-selected sample (mostly English-speaking Stack Overflow users); under-represents non-Western developers; some survey questions change year to year

**What makes this original:**
Every student who touches this dataset predicts salary. Nobody asks about the structural predictors of burnout and whether they interact with geography, experience, or employment type. If you frame it that way and include a fairness lens, it is a genuinely novel piece of work.

---

## Option 4 — Transit Delay Prediction Using City Open Data

**What it is:**
Most major cities publish real-time and historical public transit data under open licenses. New York, Chicago, Toronto, and London all have years of bus and subway delay records, route data, weather correlations, and ridership counts available for free download.

**Why it's interesting:**
This is a real time-series prediction problem with direct civic value. Accurate delay prediction improves trip planning, reduces missed connections, and informs infrastructure investment decisions. The challenge is interesting because delays are caused by a mix of predictable factors (weather, peak hours, event days) and unpredictable ones (mechanical failures, accidents).

**What you could actually do:**
- Regression: predict delay in minutes for a given route, time of day, and weather condition
- Classification: predict whether a service will experience significant disruption (binary)
- Feature engineering: combine GTFS transit data with weather, events calendar, and historical delay records
- Comparison: SARIMA vs gradient-boosted trees vs LSTM for the time-series component

**Dataset options:**
- NYC MTA: https://new.mta.info/developers — free, GTFS format, extensive historical records
- Chicago Transit Authority: https://www.transitchicago.com/data/ — free, CSV and GTFS
- Transport for London: https://tfl.gov.uk/info-for/open-data-users/ — free API
- City of Toronto Transit: https://open.toronto.ca — free open data portal (correct URL — data.toronto.ca resolves incorrectly)

**Honest caveat:**
GTFS data requires more data engineering upfront than a clean CSV. Budget an extra week on data preparation. The payoff is that you learn to work with a real-world data format used across the entire public transit industry globally.

---

## My Actual Recommendation

If I were advising you directly, I would suggest the **Open Food Facts project** as your primary option, with **Stack Overflow Burnout** as the alternative if you prefer a cleaner dataset with less data engineering.

Here is why I favour Open Food Facts:

1. **The dataset is immediately downloadable today** with no registration, no application, and no rate limits
2. **The problem is genuinely understudied** at the student project level — you will not be the hundredth person to write "I trained XGBoost on this dataset"
3. **The data quality challenges are instructive** — dealing with a messy, real-world, user-contributed database teaches you more practical data science than a pre-cleaned Kaggle competition file
4. **Multiple ML paradigms apply naturally** — you can do supervised classification (Nutri-Score prediction), unsupervised clustering (food product archetypes), and NLP (ingredient list parsing) all within the same project scope, which demonstrates breadth
5. **The research question has a genuine critical angle** — asking whether the Nutri-Score grading system is internally consistent is more interesting than simply predicting it

---

## A Note on Scope

Whichever project you choose, I would resist the temptation to do everything. A master's project that does one thing well — with a clear research question, honest evaluation, and thoughtful interpretation of results — is far stronger than one that runs ten models and reports accuracy on all of them without asking why the differences exist.

Pick one primary research question. Run a simple baseline first. Understand why it works or fails before adding complexity. The insight matters more than the algorithm.

---

*These are my genuine suggestions — no scoring matrices, no completeness constraints. Just what I would actually tell a student.*
