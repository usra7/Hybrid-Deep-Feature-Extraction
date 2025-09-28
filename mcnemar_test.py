import pandas as pd
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from itertools import combinations

# Load your predictions from the saved CSV
predictions_df = pd.read_csv('final_results/all_predictions.csv')

# Get predictions from first run
y_test = predictions_df['True_Labels']
y_svm = predictions_df['SVM_Predictions']
y_rf = predictions_df['RandomForest_Predictions'] 
y_lr = predictions_df['LogisticRegression_Predictions']
y_knn = predictions_df['KNN_Predictions']
y_dt = predictions_df['DecisionTree_Predictions']

# Store classifiers in a dictionary
classifiers = {
    'SVM': y_svm,
    'RandomForest': y_rf, 
    'LogisticRegression': y_lr,
    'KNN': y_knn,
    'DecisionTree': y_dt
}

print(f"Total test samples: {len(y_test)}")

# Calculate McNemar's test for all pairs
print("Calculating McNemar's test...")
results = []

for clf1, clf2 in combinations(classifiers.keys(), 2):
    pred1 = classifiers[clf1]
    pred2 = classifiers[clf2]
    
    # Create 2x2 contingency table
    table = [[0, 0], [0, 0]]
    
    for i in range(len(y_test)):
        correct1 = (pred1[i] == y_test[i])
        correct2 = (pred2[i] == y_test[i])
        table[int(correct1)][int(correct2)] += 1
    
    # Perform McNemar's test
    try:
        result = mcnemar(table, exact=False, correction=True)
        
        results.append({
            'Classifier 1': clf1,
            'Classifier 2': clf2,
            'Statistic': f"{result.statistic:.2f}",
            'p-value': result.pvalue,
            'Significance': 'Significant' if result.pvalue < 0.05 else 'Not significant'
        })
    except Exception as e:
        print(f"Error with {clf1} vs {clf2}: {e}")

# Create results DataFrame
mcnemar_df = pd.DataFrame(results)

# Save to CSV
mcnemar_df.to_csv('final_results/mcnemar_test_results.csv', index=False)
print("Results saved to: final_results/mcnemar_test_results.csv")

# ============================================================================
# WORD-FRIENDLY TABLE OUTPUT
# ============================================================================

print("\n" + "="*80)
print("WORD-FRIENDLY TABLE FOR YOUR PAPER")
print("="*80)

# Create a nicely formatted table for Word
word_table = "TABLE III: PAIRWISE MCNEMAR'S TEST COMPARING CLASSIFIER PERFORMANCE ON THE TEST SET. SIGNIFICANCE INDICATES P < 0.05.\n\n"

# Table header
word_table += "| Classifier 1       | Classifier 2       | Statistic | p-value    | Significance    |\n"
word_table += "|--------------------|--------------------|-----------|------------|-----------------|\n"

# Table rows
for _, row in mcnemar_df.iterrows():
    clf1 = row['Classifier 1'].ljust(18)
    clf2 = row['Classifier 2'].ljust(18)
    stat = row['Statistic'].ljust(9)
    
    # Format p-value nicely
    pval = float(row['p-value'])
    if pval < 0.0001:
        pval_str = f"{pval:.2e}".ljust(10)
    else:
        pval_str = f"{pval:.4f}".ljust(10)
    
    sig = row['Significance'].ljust(15)
    
    word_table += f"| {clf1} | {clf2} | {stat} | {pval_str} | {sig} |\n"

print(word_table)

# Also create a simple text version for easy copying
print("\n" + "="*80)
print("SIMPLE TEXT VERSION (Copy-paste into Word)")
print("="*80)

simple_text = "TABLE III: PAIRWISE MCNEMAR'S TEST COMPARING CLASSIFIER PERFORMANCE ON THE TEST SET. SIGNIFICANCE INDICATES P < 0.05.\n\n"

for _, row in mcnemar_df.iterrows():
    pval = float(row['p-value'])
    if pval < 0.0001:
        pval_str = f"{pval:.2e}"
    else:
        pval_str = f"{pval:.4f}"
    
    simple_text += f"{row['Classifier 1']} vs {row['Classifier 2']}: Statistic = {row['Statistic']}, p-value = {pval_str}, {row['Significance']}\n"

print(simple_text)

# ============================================================================
# RESULTS SUMMARY FOR YOUR DISCUSSION SECTION
# ============================================================================

print("\n" + "="*80)
print("SUMMARY FOR YOUR DISCUSSION SECTION")
print("="*80)

significant_pairs = []
not_significant_pairs = []

for _, row in mcnemar_df.iterrows():
    pair = f"{row['Classifier 1']} vs {row['Classifier 2']}"
    pval = float(row['p-value'])
    
    if pval < 0.05:
        significant_pairs.append((pair, pval))
    else:
        not_significant_pairs.append((pair, pval))

print("SIGNIFICANT DIFFERENCES (p < 0.05):")
if significant_pairs:
    for pair, pval in significant_pairs:
        if pval < 0.0001:
            pval_str = f"{pval:.2e}"
        else:
            pval_str = f"{pval:.4f}"
        print(f"• {pair} (p = {pval_str})")
else:
    print("No significant differences found.")

print("\nNOT SIGNIFICANT DIFFERENCES:")
if not_significant_pairs:
    for pair, pval in not_significant_pairs:
        pval_str = f"{pval:.4f}"
        print(f"• {pair} (p = {pval_str})")

# ============================================================================
# SAVE WORD TABLE TO TEXT FILE
# ============================================================================

with open('final_results/mcnemar_word_table.txt', 'w') as f:
    f.write(word_table)
    f.write("\n\n")
    f.write(simple_text)
    f.write("\n\nSUMMARY:\n")
    f.write("Significant differences (p < 0.05):\n")
    for pair, pval in significant_pairs:
        if pval < 0.0001:
            pval_str = f"{pval:.2e}"
        else:
            pval_str = f"{pval:.4f}"
        f.write(f"• {pair} (p = {pval_str})\n")
    
    f.write("\nNot significant differences:\n")
    for pair, pval in not_significant_pairs:
        pval_str = f"{pval:.4f}"
        f.write(f"• {pair} (p = {pval_str})\n")

print(f"\nWord table saved to: final_results/mcnemar_word_table.txt")

print("\n" + "="*80)
print("INSTRUCTIONS FOR WORD:")
print("="*80)
print("1. Copy the table from above into your Word document")
print("2. Select the table text")
print("3. Go to Insert → Table → Convert Text to Table")
print("4. Set 'Number of columns' to 5")
print("5. Check 'Separate text at: Tabs'")
print("6. Click OK")
print("7. Format the table with your preferred style")