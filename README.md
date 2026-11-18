# Customer Segmentation using K-Means Clustering

**SkillCraft Technology – Machine Learning Internship**

**Task 02: Customer Segmentation**

## Project Overview
This project uses the K-Means Clustering algorithm to group mall customers based on their annual income and spending scores.

## Objective
To identify different customer groups and understand their spending behaviour using unsupervised machine learning.

## Dataset
- Dataset: Mall Customers
- Total Records: 200
- Total Columns: 5

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Input Features
- Annual Income (k$)
- Spending Score (1-100)

## Machine Learning Algorithm
K-Means Clustering

## Number of Clusters
5

## Model Evaluation
- Silhouette Score: 0.5547

## Customer Cluster Distribution

| Cluster | Number of Customers |
|---|---:|
| 0 | 81 |
| 1 | 39 |
| 2 | 22 |
| 3 | 35 |
| 4 | 23 |

## Visualizations
- `elbow_method.png` – Determines the optimal number of clusters.
- `customer_segmentation.png` – Displays the five customer groups.

## Project Files
- `customer_segmentation.py` – Main Python program.
- `Mall_Customers.csv` – Dataset.
- `customer_segments.csv` – Clustered customer data.
- `elbow_method.png` – Elbow graph.
- `customer_segmentation.png` – Customer segmentation graph.
- `README.md` – Project documentation.

## How to Run

Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn
```

Run the program:

```bash
python customer_segmentation.py
```

## Conclusion
The project successfully applies K-Means Clustering to divide 200 mall customers into five groups based on annual income and spending scores. The results help identify different customer spending patterns.

**Internship:** SkillCraft Technology – Machine Learning Internship

**Task 02:** Customer Segmentation using K-Means Clustering
