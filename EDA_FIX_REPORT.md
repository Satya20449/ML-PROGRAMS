# EDA error-fix report

## Error

Running `EDA.py` stopped with:

```text
KeyError: "['AttendancePercentage'] not in index"
```

The numeric-distribution section selected `AttendancePercentage`, but that column is not present in `placementprediction.csv`.

## Cause

The dataset header is `AttendancePercent`, not `AttendancePercentage`. Pandas raises a `KeyError` whenever code selects a column name that does not exist in the DataFrame.

## Fix

Updated the numeric histogram column list in `EDA.py`:

```python
hist_cols = ["CGPA", "AttendancePercent", "AptitudeTestScore"]
```

Also replaced `plt.show("Numeric Feature Distribution")` with `plt.suptitle("Numeric Feature Distribution")` followed by `plt.show()`. `show()` displays the chart; it is not a title-setting function.

## Plot-display problem

After the column name was fixed, the script appeared to generate only the first histogram. This was caused by the individual `plt.show()` calls throughout the script. In a standard Python run, `plt.show()` is blocking: execution waits at the first call until its plot window is closed. The code for the later plots has therefore not run yet.

## Fix for complete output

The script now saves each generated plot under `eda_output/` and calls `plt.show()` only once, after all plotting sections have run. This means:

- all eight plots are generated in a single run;
- the files remain available even after their plot windows are closed; and
- all open plot windows display together at the end of the script.

Generated filenames are:

- `01_numeric_feature_distribution.png`
- `02_cgpa_distribution_with_mean.png`
- `03_cgpa_vs_salary_package.png`
- `04_coding_test_vs_aptitude_test.png`
- `05_count_gender.png`
- `05_count_city.png`
- `05_count_hostel.png`
- `05_count_historyofbacklogs.png`

## Verification

Confirmed that the CSV contains `AttendancePercent`, and ran the entire EDA script in non-interactive mode. All eight expected plot files were created and the script completed without an exception.
