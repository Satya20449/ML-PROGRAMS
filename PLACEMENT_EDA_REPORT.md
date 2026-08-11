# Placement EDA report

The analysis is implemented in `EDA.py`. It uses `pandas` for data loading and summaries, `matplotlib.pyplot` for figure setup/saving, and `seaborn` for statistical plots.

| Task | Tools used | What the code does |
|---|---|---|
| 1. Load data | `pd.read_csv`, `.shape`, `.head()` | Reads `placementprediction.csv`, then prints its dimensions and first five records. |
| 2. Basic structure | `.info()`, `.dtypes`, `.describe()` | Prints column types, non-null counts, numeric statistics, and categorical statistics. |
| 3. Missing values | `.isnull()`, `sns.heatmap`, `.fillna()` | Prints missing counts and percentages, saves a missing-data heatmap, and replaces numeric NaNs with each column median. |
| 4. Duplicate rows | `.duplicated().sum()` | Counts fully duplicate student records. |
| 5. Target distribution | `.value_counts()`, `sns.countplot` | Prints and plots the number of placed and not-placed students. |
| 6. Numeric distributions | `DataFrame.hist` | Creates histograms for CGPA, attendance, aptitude, coding, soft-skills, and mock-interview scores. |
| 7. Outliers | `sns.boxplot` | Draws boxplots for CGPA, assessment scores, and salary package to identify extreme values. |
| 8. Correlation | `.corr()`, `sns.heatmap` | Produces a numeric correlation heatmap and prints each numeric feature's correlation with `PlacementStatus`. |
| 9. Relationship plots | `sns.regplot` | Draws fitted regression scatterplots for CGPA vs salary (placed students) and aptitude vs coding score. |
| 10. Categorical counts | `.value_counts()`, `sns.countplot` | Prints and plots frequencies for gender, city, tier, stream, specialisation, hostel, backlogs, and CGPA tier. |
| 11. Gender vs placement | `pd.crosstab`, `sns.countplot(hue=...)` | Shows placement-status counts separately for each gender. |
| 12. Tier/stream vs placement | `pd.crosstab`, `sns.countplot(hue=...)` | Shows placement-status counts by college tier and by stream. |
| 13. SGPA trend | `.mean()`, `plt.plot` | Calculates mean SGPA for Sem1–Sem8 and plots the semester trend line. |
| 14. Salary analysis | `.describe()`, `sns.histplot`, `sns.boxplot` | Describes salary for placed students, plots its distribution, and compares it between college tiers. |
| 15. Pairplot | `sns.pairplot` | Displays pairwise CGPA, aptitude, coding, and mock-interview relationships, coloured by placement status. |

## Generated output

Running `EDA.py` saves PNG plots in `eda_output/`. Console output contains the tables, counts, duplicate check, summaries, and correlations.
