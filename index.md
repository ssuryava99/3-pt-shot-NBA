---
layout: default
title: "3 Point Reliance and NBA Team Success"
---

# 3 Point Reliance and NBA Team Success

### Predicting Winning Records from 10 Years of NBA Shot Profiles

**Saarang Suryavanshi** (ssuryava@umich.edu)

---

## Introduction

This project is for NBA fans who, like me, glaze Steph Curry but hate the Celtics for living and dying by the 3. For this project I scraped **10 seasons of team level NBA stats (2014‑15 through 2023‑24)** off Basketball‑Reference and merged the per game, advanced, and standings tables together. After tossing out the “League Average” rows and a few junk columns, I was left with **≈ 300 rows × 38 columns**: every team season in the modern 3 chucking era.

My main question is very simple:

> **Does leaning on the 3 pointer actually help a team finish the regular season with a winning record?**
> (A winning record = Win Pct > 0.500)

Joe Mazzulla says “shoot more threes,” but I want to see if the numbers back that up across an entire decade. Figuring this out is useful for all NBA fans on Reddit who constantly argue that “live by the three, die by the three” is how basketball should be played.

Key columns I’ll reference:

| Column              | What it is                                      | Why I care                       |
| ------------------- | ----------------------------------------------- | -------------------------------- |
| **3PA**             | Raw 3 point attempts per game                   | Volume indicator              |
| **3P%**             | Team 3 point percentage                         | Efficiency check                 |
| **ThreePointFreq**  | 3PA / FGA                                       | *How* 3 heavy a team is      |
| **ThreePA\_per100** | Pace adjusted 3PA                               | Controls for fast vs. slow teams |
| **TS%**             | True shooting percentage                        | Overall scoring efficiency       |
| **Pace**            | Possessions per 48 min                          | Needed for per‑100 scaling       |
| **WinPct**          | Season win percentage                           | Builds the target variable       |
| **WinningRecord**   | 1 = WinPct > 0.500                              | What I’m predicting              |
| **Conference**      | East / West                                     | Categorical feature              |
| **Coach**           | Head coach name                                 | Possible strategic signal        |
| **High3P\_Flag**    | 1 = above median ThreePointFreq for that season | Quick binary “3 heavy?” tag  |

---

## Data Cleaning and Exploratory Data Analysis

### Cleaning moves I made and why they matter

* **Dropped “League Average” rows**: "League Average" isn't it's own team, it's a culmination of all the teams, so including them would double count league totals.
* **Stripped off the asterisks** off playoff teams (e.g. *DEN*) so merge keys line up. I needed all of the team names to be consistent.
* **Converted percent strings → floats** (`"35.4%"` → 0.354) so that any math actually works.
* **Filled missing `Pace`** by stealing the same value from the advanced table when Basketball-Reference returned NaN. Then, with the leftover NaNs, I mean imputed them in the modeling pipeline.
* **Engineered features** (`ThreePointFreq`, `ThreePA_per100`, `TS%`, `High3P_Flag`, `WinningRecord`) right in the notebook for exploration. I also recreate them in sklearn transformers later.

The first five rows of the cleaned frame (below) displays all of the data after I cleaned it (hence its name). There's no asterisks in the team names, number/percent columns are numeric, and my new columns show up on the right.

<iframe
    src="{{ '/assets/cleaned_head.html' | relative_url }}"
    width="100%"
    height="320"
    frameborder="0"
></iframe>

#### Univariate Analysis: Histogram of **ThreePointFreq**

<iframe
    src="{{ '/assets/threepointfreq_hist.html' | relative_url }}"
    width="100%"
    height="550"
    frameborder="0"
></iframe>

Teams cluster between **0.3 and 0.4** (so roughly 30–40 % of their shots are 3s), centered at 0.35, with a long right tail for the ultra 3 heavy teams. The distribution tells us that there’s enough variation to test whether “more threes → more wins.”

#### Bivariate Analysis: Scatterplot of **ThreePointFreq vs WinPct**

<iframe
    src="{{ '/assets/freq_vs_win_scatter.html' | relative_url }}"
    width="100%"
    height="550"
    frameborder="0"
></iframe>

Each dot is a team season. The slight upward trendline hints that higher 3 point frequency *does* correlate with winning, but the cloud of points is wide enough that it would seem that coaching, defense, and luck clearly matter too. This plot basically justifies building the classifier.

#### Quick aggregate: Pivot table → **Mean WinPct by ThreePointFreq quintile**

<iframe
    src="{{ '/assets/winpct_pivot.html' | relative_url }}"
    width="100%"
    height="350"
    frameborder="0"
></iframe>

Across seasons, the **Q5 (heaviest 3 shooting) teams average \~0.56 WinPct**, while Q1 averages \~0.48. Not every year is monotonic, but the overall bump supports the “live by the three” hypothesis which motivated me in adding that `High3P_Flag` feature.

Together these visuals tell us something: that 3 point volume *might* actually be the key to winning (unfortunately). Regardless, this setup works well for the modeling steps that follow.

---

## Framing a Prediction Problem

**Problem type.** This is a **binary classification** predictive problem: Classify whether an NBA team finishes the season with a winning record (`WinningRecord = 1` if WinPct > 0.500, else 0) using team level statistics from the 2014-15 to 2023-24 seasons.

**Predictive vs. inferential?** This is a *predictive* problem. To make sure to have consistent data overall, I made sure all of my input features were strictly from regular season games, so we don't accidentally sneak in any “future” info like playoff games. The goal is to answer, *given a team’s on‑court style, could we have guessed they’d cross .500?*.

**Evaluation metric.** I went with **accuracy**:

* Class balance is \~52 % winners vs 48 % non‑winners which wasn't skewed enough to justify using F1 or AUROC.
* Readers can understand “78 % accuracy” more intuitively than an F1 of “0.79.”
* I still show precision/recall in the notebook for completeness, but the main evaluation metric is accuracy.

---

## Baseline Model

### Pipeline architecture

1. **Numeric features (2 quantitative)**

   * `3PA` – raw 3 point attempts
   * `3P%` – 3 point accuracy  → `StandardScaler` so magnitude doesn’t affect the optimizer too much
2. **Categorical feature (1 nominal)**

   * `Conference` – East or West  → `OneHotEncoder` (two dummy columns, drop‑first handled automatically)
3. **Classifier**

   * `LogisticRegression` (`max_iter=1000`, default `C=1`)

Everything lives in a single `Pipeline`, so the train/test split (80/20, `stratify=y`, `random_state=42`) flows through preprocessing and modeling in one go.

| Metric            | Score       |
| ----------------- | ----------- |
| **Accuracy**      | **0.767**   |
| Precision (0 / 1) | 0.74 / 0.79 |
| Recall (0 / 1)    | 0.79 / 0.74 |

*Is 76.7 % “good”?* Compared to a **naive majority class guess (51.7 %)** it’s a pretty huge increase, which tells us that volume + efficiency + conference already captures a lot of information. But the confusion matrix (👇) shows 14 wrong predictions out of 60 test rows, so there’s still room for improvement.

<iframe
    src="{{ '/assets/baseline_conf_matrix.html' | relative_url }}"
    width="100%"
    height="450"
    frameborder="0"
></iframe>

---

## Final Model

### New Features

| New feature                    | Data‑generating intuition                                                          |
| ------------------------------ | ---------------------------------------------------------------------------------- |
| **ThreePointFreq = 3PA / FGA** | Pure “how 3 happy are we?” ratio (volume independent of pace). Teams that shoot more threes in their possessions are taking higher variance shots which could be beneficial or disadvantageous. It's possible for a team to have an advantage due to this metric if their team is built with all snipers.                 |
| **ThreePA\_per100**            | Normalizes 3PA to 100 possessions → that way we can see attemps without factoring in pace. Because pace is extremely variable across teams, scaling to per 100 gives us the ability to compare raw volume of shots without worrying about the pace which gives the model a less skewed statistic to predict on. |
| **TS%**                        | **TS%** = Points Scored / [2 × (FGA + 0.44 × FTA)] so it accounts for teams' efficiency in shooting. It takes into account 3s, 2s, and free throws and just how much they're all worth. This feature goes along well ThreePointFreq, because if both ThreePointFreq and TS% are high then this could be a huge factor in winning (and losing vice versa)                |
| **High3P\_Flag**               | Season relative binary flag → lets the model learn a nonlinear “threshold” bump. Being over league median threshold tells us a bit about the playstyle of the team. A high 3 point shooting team is more likely to be playing small ball like the rockets telling us they have defensive liabilities. This dummy variable lets logistic regression pick up any step function effect that raw frequency alone might just smoooth over.  |
| **Coach**                      | Accounts for Team Strategy → some coaches are 3 point merchants. Also some coaches are just known to have great success in the regular season (look at J.B. Bickerstaff vs Monty Williams, Pistons head coaches). Just 1 year apart, you can see the impact that the head coach had on the success of the team (playoff-caliber team vs the worst team in NBA History).                  |

These sit alongside the original 3PA / 3P% / Conference, bringing the tally to **5 quantitative + 3 categorical** (counting the dummy for `High3P_Flag`).

### Model & Tuning

* **Pre‑processing**
  * Quantitative → `StandardScaler` **+** `PolynomialFeatures(degree=2)` to let interactions (e.g., high volume and high accuracy) show up better.
  * Categoricals → `OneHotEncoder(handle_unknown='ignore')`
* **Classifier:** `LogisticRegression` with L2 penalty.
* **Hyperparameters:** Searched `C ∈ {0.01, 0.1, 1, 10, 100}` via 5‑fold `GridSearchCV` (still inside the 80 % training fold).
  The **best C = 0.01** was a heavier regularizer which fought off the extra polynomial noise.

### Results

| Model                | Features                | Accuracy  | Δ vs. baseline |
| -------------------- | ----------------------- | --------- | -------------- |
| Baseline             | 2 numerical + 1 categorical           | 0.767     | —              |
| **Final (C = 0.01)** | 5 numerical + 3 categorical (+ polynomial) | **0.783** | **+1.6 pp**    |

Confusion‑matrix shift:

* True negatives up from **23 → 25**
* False positives down from **6 → 4**

<iframe
    src="{{ '/assets/final_conf_matrix.html' | relative_url }}"
    width="100%"
    height="450"
    frameborder="0"
></iframe>

### How did we Increase?

* **ThreePointFreq / ThreePA\_per100** isolate playstyle from raw attempts → teams that chucked threes and ran the ball faster got a boost.
* **TS%** rewards efficient scoring → a team can chuck up 45 threes and still brick their way below .500.
* **Coach One-Hot Encoding** lets the model spot patterns like “Mike D’Antoni teams = green light/small ball/defensive liability”
* Polynomial terms catch combined effects (like high volume × high accuracy) that a plain linear boundary would miss.

The improvement honestly wasn't much as it turns out, basketball outcomes are pretty noisy. But the feature engineering narrative lines up with how the modern NBA actually plays.

### Conclusion

Over ten seasons’ worth of data, three distinct stages of analysis led us from the question of whether teams that chuck threes win more, to a bit more clear understanding of the nuances using code, visuals, and two predictive models.

| Stage              | Key moves                                                                                                                  | Takeaways                                                                                                                |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Cleaning & EDA** | Fixed percent strings, patched missing *Pace*, engineered more features, visualized distributions and scatterplots | 3 point share has widened every year and already shows a slightly positive slope against **WinPct**                             |
| **Baseline model** | `3PA + 3P%` (scaled) ⊕ `Conference` (OHE) → *LogReg*                                                                       | **76.7 %** accuracy ⇒ volume + efficiency alone beat a naive 52 % majority guess by > 24 pp                                    |
| **Final model**    | Added pace adjusted volume, true shooting, binary “heavy 3” flag, coach OHE, degree = 2 interactions, tuned *C*              | **78.3 %** accuracy, sharper confusion matrix (FP ↓ 33 %), feature importances confirmed *ThreePointFreq* & *TS%* as useful signals for the model |

#### Does "living by the 3" put you over .500?

**Yes, but only if you combine volume with efficiency.**
Teams in the top 20 % of *ThreePointFreq* averaged **0.56 WinPct**, a full 8 pp above the bottom quintile. The classifier’s coefficients and SHAP summaries (in the notebook) showed the same thing: high volume + high accuracy teams normally have a higher win percentage than teams that lack either. Yet, the wide scatter and still modest test accuracy (78 %) remind us there are probably many more factors that play a part in a teams ability to have a winning season. Defense, injuries, and coaching philosophies can have a huge impact on teams, especially those who rely on big stars like Steph Curry and Lebron James. Threes definitely help, but they don’t guarantee wins. Remember folks...Celtics got cooked this year in the playoffs (2025).

#### More Highlights

* Pace adjusted 3PA (*ThreePA\_per100*) exposed fast teams that looked trigger happy but actually just ran more possessions.
* True Shooting (*TS%*) was better than raw 3P% once we let the model weigh both 2s and free throws along with the 3s.
* OHE Coaches improved the model in very subtle ways like how D’Antoni/Houser lineups flagged as perennial green lights.

#### Limitations & next steps

* **Small N**: 300 team seasons give useful signal but cap model complexity.
* **Features not looked at**: opponent 3 point defense, travel fatigue, roster age, injury games missed.

#### TL;DR for Reddit debates

*Chuck ≠ chuck & pray.* Teams that **shoot a lot *and* shoot well** end up winners far more often than losers which is basically common sense. While the NBA’s 3 point revolution isn’t foolproof, across a decade of data it’s definitely become a bigger factor in winning games.

