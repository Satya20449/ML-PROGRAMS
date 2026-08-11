"""Placement prediction dataset: exploratory data analysis (15 tasks)."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Common setup
sns.set_theme(style="whitegrid")
OUTPUT_DIR = Path("eda_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def save_plot(filename):
    """Save and close the current plot."""
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


# 1. Load Data
df = pd.read_csv("placementprediction.csv")
print("\n1. LOAD DATA")
print("Shape:", df.shape)
print(df.head())


# 2. Basic Info / Structure
print("\n2. BASIC INFO / STRUCTURE")
df.info()
print("\nData types:\n", df.dtypes)
print("\nNumeric summary:\n", df.describe())
print("\nCategorical summary:\n", df.describe(include="object"))


# 3. Missing Values
print("\n3. MISSING VALUES")
missing_report = pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing Percentage": (df.isnull().mean() * 100).round(2),
})
print(missing_report[missing_report["Missing Count"] > 0])

plt.figure(figsize=(14, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis")
plt.title("Missing Values Heatmap")
save_plot("03_missing_values_heatmap.png")

# Fill missing values in numeric columns with their median.
numeric_columns = df.select_dtypes(include="number").columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
print("Numeric missing values after median fill:", df[numeric_columns].isnull().sum().sum())


# 4. Duplicate Rows
print("\n4. DUPLICATE ROWS")
print("Duplicate records:", df.duplicated().sum())


# 5. Target Variable Distribution
print("\n5. TARGET VARIABLE DISTRIBUTION")
print(df["PlacementStatus"].value_counts())
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="PlacementStatus")
plt.title("Placement Status Distribution")
save_plot("05_placement_status_distribution.png")


# 6. Numeric Feature Distributions
print("\n6. NUMERIC FEATURE DISTRIBUTIONS")
score_columns = ["CGPA", "AttendancePercent", "AptitudeTestScore", "CodingTestScore",
                 "SoftSkillsRating", "MockInterviewScore"]
score_columns = [column for column in score_columns if column in df.columns]
df[score_columns].hist(figsize=(14, 8), bins=20, edgecolor="black", color="steelblue")
plt.suptitle("Distributions of Academic and Assessment Scores")
save_plot("06_numeric_feature_distributions.png")


# 7. Outlier Detection (Boxplots)
print("\n7. OUTLIER DETECTION")
boxplot_columns = ["CGPA", "AptitudeTestScore", "CodingTestScore", "SoftSkillsRating",
                   "MockInterviewScore", "Salary Package"]
boxplot_columns = [column for column in boxplot_columns if column in df.columns]
plt.figure(figsize=(14, 6))
sns.boxplot(data=df[boxplot_columns])
plt.title("Boxplots of Key Numeric Features")
plt.xticks(rotation=30, ha="right")
save_plot("07_outlier_boxplots.png")


# 8. Correlation Analysis
print("\n8. CORRELATION ANALYSIS")
correlation = df.select_dtypes(include="number").corr()
print("Correlation with PlacementStatus:\n", correlation["PlacementStatus"].sort_values(ascending=False))
plt.figure(figsize=(16, 12))
sns.heatmap(correlation, cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Heatmap")
save_plot("08_correlation_heatmap.png")


# 9. Relationship Plots
print("\n9. RELATIONSHIP PLOTS")
placed_df = df[df["PlacementStatus"] == 1]
plt.figure(figsize=(8, 5))
sns.regplot(data=placed_df, x="CGPA", y="Salary Package", scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})
plt.title("CGPA vs Salary Package (Placed Students)")
save_plot("09_cgpa_vs_salary_package.png")

plt.figure(figsize=(8, 5))
sns.regplot(data=df, x="AptitudeTestScore", y="CodingTestScore", scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})
plt.title("Aptitude Test Score vs Coding Test Score")
save_plot("09_aptitude_vs_coding.png")


# 10. Categorical Feature Counts
print("\n10. CATEGORICAL FEATURE COUNTS")
categorical_columns = ["Gender", "City", "CollegeTier", "Stream", "Specialisation",
                       "Hostel", "HistoryOfBacklogs", "CGPA_Tier"]
for column in categorical_columns:
    print(f"\n{column}:\n", df[column].value_counts())
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x=column, order=df[column].value_counts().index)
    plt.title(f"Count of {column}")
    plt.xticks(rotation=45, ha="right")
    save_plot(f"10_count_{column.lower()}.png")


# 11. Gender vs Placement Status
print("\n11. GENDER VS PLACEMENT STATUS")
print(pd.crosstab(df["Gender"], df["PlacementStatus"]))
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Gender", hue="PlacementStatus")
plt.title("Placement Status by Gender")
save_plot("11_gender_vs_placement.png")


# 12. College Tier / Stream vs Placement Status
print("\n12. COLLEGE TIER / STREAM VS PLACEMENT STATUS")
print("College Tier:\n", pd.crosstab(df["CollegeTier"], df["PlacementStatus"]))
print("Stream:\n", pd.crosstab(df["Stream"], df["PlacementStatus"]))
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sns.countplot(data=df, x="CollegeTier", hue="PlacementStatus", ax=axes[0])
axes[0].set_title("Placement Status by College Tier")
sns.countplot(data=df, x="Stream", hue="PlacementStatus", ax=axes[1])
axes[1].set_title("Placement Status by Stream")
axes[1].tick_params(axis="x", rotation=45)
save_plot("12_college_tier_stream_vs_placement.png")


# 13. SGPA Trend Across Semesters
print("\n13. SGPA TREND ACROSS SEMESTERS")
semester_columns = [f"SGPA_Sem{i}" for i in range(1, 9)]
average_sgpa = df[semester_columns].mean()
print(average_sgpa)
plt.figure(figsize=(10, 5))
plt.plot(range(1, 9), average_sgpa.values, marker="o")
plt.xticks(range(1, 9), [f"Sem {i}" for i in range(1, 9)])
plt.xlabel("Semester")
plt.ylabel("Average SGPA")
plt.title("Average SGPA Trend Across Semesters")
save_plot("13_sgpa_trend.png")


# 14. Salary Package Analysis
print("\n14. SALARY PACKAGE ANALYSIS")
print(placed_df["Salary Package"].describe())
plt.figure(figsize=(8, 5))
sns.histplot(data=placed_df, x="Salary Package", bins=20, kde=True)
plt.title("Salary Package Distribution for Placed Students")
save_plot("14_salary_distribution.png")

plt.figure(figsize=(7, 5))
sns.boxplot(data=placed_df, x="CollegeTier", y="Salary Package")
plt.title("Salary Package by College Tier")
save_plot("14_salary_by_college_tier.png")


# 15. Pairplot
print("\n15. PAIRPLOT")
pairplot_columns = ["CGPA", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore", "PlacementStatus"]
pairplot = sns.pairplot(df[pairplot_columns], hue="PlacementStatus", diag_kind="hist", corner=True)
pairplot.fig.suptitle("Pairwise Relationships by Placement Status", y=1.02)
pairplot.savefig(OUTPUT_DIR / "15_pairplot.png", dpi=150, bbox_inches="tight")
plt.close("all")

print(f"\nAll EDA plots saved to: {OUTPUT_DIR.resolve()}")
