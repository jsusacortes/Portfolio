# Data Science Portfolio Projects

This repository contains three end-to-end machine learning projects completed as part of the **MIT Applied Data Science & Machine Learning Certification**. Each project tackles a real‑world business problem using a different machine learning paradigm: unsupervised clustering, supervised classification, and recommendation systems.

---

## 📁 Project Overview

| Project | Problem Type | Key Techniques |
|---------|--------------|----------------|
| **Customer Personality Segmentation** | Unsupervised Learning | K‑Means Clustering, Feature Scaling, Elbow Method, Silhouette Analysis |
| **Lead Conversion Prediction** | Supervised Classification | Decision Tree, Random Forest, Hyperparameter Tuning, Cost‑Complexity Pruning |
| **Amazon Product Recommendation** | Recommendation Systems | User‑Based & Item‑Based Collaborative Filtering, Matrix Factorization (SVD) |

All projects are implemented in **Python** using popular libraries (pandas, scikit‑learn, matplotlib, seaborn) and are provided as **Jupyter notebooks** compatible with **Google Colab**.

---

## 📊 Project 1: Customer Personality Segmentation

### Business Context
A leading retail company wants to better understand its customers to create personalized marketing campaigns, improve retention, and optimize resource allocation. Using customer demographics, spending habits, and campaign response data, we segment customers into distinct groups.

### Dataset
- **Source:** 2,240 customers, 28 features (demographics, spending, campaign responses)
- **Key Features:** Income, Spending per product category (wine, meat, etc.), Campaign acceptance, Purchase channels

### Approach
1. **Data Cleaning:** Median imputation for missing `Income` values (right‑skewed distribution).
2. **Feature Scaling:** StandardScaler applied to all numeric features.
3. **Clustering:** K‑Means with optimal *K* determined by Elbow Method and Silhouette Score → *K=3*.
4. **Cluster Profiling:** Analyzed mean/median values and boxplots to interpret segments.

### Results & Recommendations
- **Cluster 0 – Budget Shoppers:** Lower income, frequent website visits but low conversion. *Action:* Discounts, loyalty points, and web‑only deals.
- **Cluster 1 – Mid‑Tier:** Average income and spending, balanced channels. *Action:* Personalized product recommendations, moderate‑value campaigns.
- **Cluster 2 – Premium:** High income, high spending on wine/meat, strong catalog response. *Action:* VIP loyalty program, exclusive early access, premium campaigns.

**Key Insight:** The strongest predictors of segment membership are income and spending on wine/meat. The model provides a clear blueprint for targeted marketing.

---

## 📈 Project 2: Lead Conversion Prediction (ExtraaLearn)

### Business Context
ExtraaLearn, an EdTech startup, generates a high volume of leads daily but struggles to identify which ones are most likely to convert to paying customers. We build a machine learning model to predict conversion and uncover the key drivers.

### Dataset
- **Source:** 4,612 leads, 15 features
- **Target:** `status` – Converted (1) / Not Converted (0)
- **Features:** Age, occupation, first interaction channel, profile completion, website activity, marketing channels

### Approach
1. **Data Cleaning:** Removed outliers in `website_visits` and `page_views_per_visit` using IQR.
2. **Feature Engineering:** One‑hot encoding of categorical variables.
3. **Modeling:**
   - Baseline Decision Tree (overfit)
   - Tuned Decision Tree via GridSearchCV (improved generalization)
   - Pruned Decision Tree using cost‑complexity pruning
   - Random Forest (ensemble, robust performance)
4. **Evaluation:** Accuracy, recall (most important for minimizing missed conversions), and feature importance analysis.

### Results & Recommendations
- **Top Features:** `time_spent_on_website`, `first_interaction_Website`, `profile_completed`, `age`.
- **Best Model:** Random Forest with strong recall (0.80+) and balanced precision.
- **Actionable Insights:**
  1. Increase time on website (interactive demos, chatbots, video).
  2. Prioritize website UX to convert first‑time visitors.
  3. Incentivize profile completion (progress bars, exclusive content).
  4. Age‑targeted messaging (ROI for professionals, community for younger leads).
  5. Deploy the model as a lead scoring system to prioritize high‑intent leads for sales outreach.

---

## 🛒 Project 3: Amazon Product Recommendation System

### Business Context
E‑commerce platforms need to recommend relevant products to users to reduce information overload and increase sales. We implement and compare multiple recommendation algorithms on an Amazon Electronics dataset.

### Dataset
- **Source:** 4.2 million ratings from Amazon Electronics
- **Filtered to** 11,315 high‑quality interactions (users with ≥50 ratings, products with ≥5 ratings)
- **Fields:** `userId`, `productId`, `rating` (1‑5)

### Approach
1. **Data Preprocessing:** Filtered low‑activity users and products to ensure meaningful similarity.
2. **Models:**
   - **Rank‑Based:** Popularity – baseline for cold‑start users.
   - **User‑User Collaborative Filtering:** Finds similar users and recommends their liked items.
   - **Item‑Item Collaborative Filtering:** Recommends items similar to those the user liked.
   - **SVD (Matrix Factorization):** Latent factor model capturing hidden patterns.
3. **Evaluation Metrics:** RMSE, Precision@10, Recall@10, F1@10 (threshold = 3.5).
4. **Hyperparameter Tuning:** GridSearchCV for KNN and SVD.

### Results
| Model | RMSE | Precision | Recall | F1 |
|-------|------|-----------|--------|----|
| User‑User (Baseline) | 1.02 | 0.79 | 0.88 | 0.83 |
| User‑User (Optimized) | 0.97 | 0.84 | 0.93 | 0.88 |
| Item‑Item (Baseline) | 1.05 | 0.77 | 0.86 | 0.81 |
| Item‑Item (Optimized) | 1.00 | 0.82 | 0.91 | 0.86 |
| SVD (Baseline) | 0.94 | 0.84 | 0.92 | 0.88 |
| **SVD (Optimized)** | **0.92** | **0.84** | **0.92** | **0.88** |

**Key Takeaways:**
- Optimized SVD delivers the best overall performance.
- Collaborative filtering (user‑user) is more effective than item‑item for this dataset.
- Rank‑based serves as a cold‑start fallback.
- Production system could combine rank‑based for new users and SVD for existing users.

---

## 🚀 How to Run the Notebooks

All notebooks are designed to run in **Google Colab** (recommended) or locally with Jupyter.

### Prerequisites
- Google account (for Colab)
- Python 3.7+
- Required libraries (see `requirements.txt`)

### Steps for Each Project
1. Open the notebook in Google Colab:
   - Upload the `.ipynb` file to your Drive or open directly from GitHub.
2. Mount your Google Drive (if needed for data access):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Ensure the dataset file is in the specified path. The notebooks assume the dataset is in the Colab environment:
   - For **Customer Personality Segmentation:** `Customer_Personality_Segmentation.csv`
   - For **Lead Conversion Prediction:** `ExtraaLearn.csv`
   - For **Recommendation System:** `ratings_Electronics.csv`
   - *You may need to upload these files to your Colab session or adjust the paths.*
4. Run all cells sequentially.

### Installation (Local)
If running locally, install dependencies:
```bash
pip install -r requirements.txt
```
Then run Jupyter:
```bash
jupyter notebook
```

### Datasets
The datasets used in these projects are publicly available:
- **Customer Personality Segmentation:** [Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)
- **Lead Conversion Prediction:** [Kaggle](https://www.kaggle.com/datasets/knightbearr/lead-conversion-prediction) (modified for this project)
- **Amazon Electronics Ratings:** [UCSD Amazon Product Data](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) – subset used for this project.

*Due to size constraints, only filtered versions are included in the repository. Please download the raw data from the links above if you wish to reproduce the filtering steps.*

---

## 📦 Dependencies

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit‑learn
- scikit‑surprise (for recommendation system)
- yellowbrick (optional, for clustering visualizations)

See `requirements.txt` for exact versions.

---

## 👤 Author

**Your Name**  
MIT Applied Data Science & Machine Learning Certification  
[LinkedIn Profile URL] | [GitHub Profile URL]  

*These projects were completed as part of the MIT Professional Education program. They demonstrate practical skills in data preprocessing, exploratory data analysis, model building, evaluation, and deriving business‑relevant insights.*

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
