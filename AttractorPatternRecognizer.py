import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the synthetic data
data = pd.read_csv("synthetic_web_content.csv")
print(f"Loaded {len(data)} examples")

# Prepare features and target
X = data['content']
y_topic = data['topic']
y_attractor = data['has_attractor']

# Create train/test split
X_train, X_test, y_topic_train, y_topic_test, y_attractor_train, y_attractor_test = train_test_split(
    X, y_topic, y_attractor, test_size=0.2, random_state=42
)

# Create text features
vectorizer = TfidfVectorizer(max_features=500)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("\n===== EXPERIMENT 1: STANDARD CLASSIFICATION =====")
print("Training a standard classifier for topic prediction")

# Train a basic classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_vec, y_topic_train)

# Evaluate
y_pred = clf.predict(X_test_vec)
print("\nTopic Classification Report:")
print(classification_report(y_topic_test, y_pred))

# Analyze errors by attractor presence
errors = y_pred != y_topic_test
attractor_error_rate = (errors & (y_attractor_test == 1)).sum() / (y_attractor_test == 1).sum()
non_attractor_error_rate = (errors & (y_attractor_test == 0)).sum() / (y_attractor_test == 0).sum()

print(f"\nError rate on examples with attractors: {attractor_error_rate:.4f}")
print(f"Error rate on examples without attractors: {non_attractor_error_rate:.4f}")
print(f"Difference: {abs(attractor_error_rate - non_attractor_error_rate):.4f}")

print("\n===== EXPERIMENT 2: ATTRACTOR DETECTION =====")
print("Training a classifier to detect attractor patterns")

# Train an attractor detector
detector = LogisticRegression(max_iter=1000, random_state=42)
detector.fit(X_train_vec, y_attractor_train)

# Evaluate
y_attractor_pred = detector.predict(X_test_vec)
print("\nAttractor Detection Report:")
print(classification_report(y_attractor_test, y_attractor_pred))

# Visualize attractor detection features
feature_importance = np.abs(detector.coef_[0])
feature_names = vectorizer.get_feature_names_out()

# Get the top features
top_features_idx = np.argsort(feature_importance)[-20:]
top_features = [(feature_names[i], feature_importance[i]) for i in top_features_idx]

plt.figure(figsize=(10, 8))
y_pos = np.arange(len(top_features))
values = [f[1] for f in top_features]
labels = [f[0] for f in top_features]

plt.barh(y_pos, values)
plt.yticks(y_pos, labels)
plt.xlabel('Feature Importance')
plt.title('Top 20 Features for Attractor Detection')
plt.tight_layout()
plt.savefig('attractor_features.png')
plt.close()

print("\n===== EXPERIMENT 3: TAXONOMY-AWARE CLASSIFICATION =====")
print("Incorporating taxonomic information into the model")

# Create a feature set that includes taxonomic information
data['taxonomy_features'] = data['topic'] + '/' + data['subtopic']
taxonomy_encoder = TfidfVectorizer(max_features=50)
taxonomy_features_train = taxonomy_encoder.fit_transform(data.loc[X_train.index, 'taxonomy_features'])
taxonomy_features_test = taxonomy_encoder.transform(data.loc[X_test.index, 'taxonomy_features'])

# Combine text features with taxonomy features
from scipy.sparse import hstack
X_train_combined = hstack([X_train_vec, taxonomy_features_train])
X_test_combined = hstack([X_test_vec, taxonomy_features_test])

# Train a taxonomy-aware classifier
tax_clf = RandomForestClassifier(n_estimators=100, random_state=42)
tax_clf.fit(X_train_combined, y_topic_train)

# Evaluate
y_tax_pred = tax_clf.predict(X_test_combined)
print("\nTaxonomy-Aware Classification Report:")
print(classification_report(y_topic_test, y_tax_pred))

# Analyze errors by attractor presence for taxonomy-aware model
tax_errors = y_tax_pred != y_topic_test
tax_attractor_error_rate = (tax_errors & (y_attractor_test == 1)).sum() / (y_attractor_test == 1).sum()
tax_non_attractor_error_rate = (tax_errors & (y_attractor_test == 0)).sum() / (y_attractor_test == 0).sum()

print(f"\nError rate on examples with attractors (taxonomy-aware): {tax_attractor_error_rate:.4f}")
print(f"Error rate on examples without attractors (taxonomy-aware): {tax_non_attractor_error_rate:.4f}")
print(f"Difference: {abs(tax_attractor_error_rate - tax_non_attractor_error_rate):.4f}")

print("\n===== EXPERIMENT 4: ATTRACTOR-AWARE CLASSIFICATION =====")
print("Using attractor predictions to improve classification")

# Get attractor probabilities
attractor_probs = detector.predict_proba(X_test_vec)[:, 1].reshape(-1, 1)

# Create a combined feature set with attractor probabilities
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
attractor_probs_scaled = scaler.fit_transform(attractor_probs)

# Combine all features
X_test_full = hstack([X_test_vec, taxonomy_features_test, attractor_probs_scaled])

# Train on combined features in training set
attractor_probs_train = detector.predict_proba(X_train_vec)[:, 1].reshape(-1, 1)
attractor_probs_train_scaled = scaler.fit_transform(attractor_probs_train)
X_train_full = hstack([X_train_vec, taxonomy_features_train, attractor_probs_train_scaled])

# Train the fully aware classifier
full_clf = RandomForestClassifier(n_estimators=100, random_state=42)
full_clf.fit(X_train_full, y_topic_train)

# Evaluate
y_full_pred = full_clf.predict(X_test_full)
print("\nFully-Aware Classification Report:")
print(classification_report(y_topic_test, y_full_pred))

# Compare all models
models = [
    ("Standard", y_pred),
    ("Taxonomy-Aware", y_tax_pred),
    ("Fully-Aware", y_full_pred)
]

print("\n===== MODEL COMPARISON =====")
for name, preds in models:
    accuracy = (preds == y_topic_test).mean()
    attractor_accuracy = (preds[y_attractor_test == 1] == y_topic_test[y_attractor_test == 1]).mean()
    non_attractor_accuracy = (preds[y_attractor_test == 0] == y_topic_test[y_attractor_test == 0]).mean()
    
    print(f"{name} Model:")
    print(f"  Overall Accuracy: {accuracy:.4f}")
    print(f"  Accuracy on examples with attractors: {attractor_accuracy:.4f}")
    print(f"  Accuracy on examples without attractors: {non_attractor_accuracy:.4f}")
    print(f"  Gap: {abs(attractor_accuracy - non_attractor_accuracy):.4f}")
    print("")

# Visualize the comparison
model_names = [m[0] for m in models]
overall_acc = [(m[1] == y_topic_test).mean() for m in models]
attractor_acc = [(m[1][y_attractor_test == 1] == y_topic_test[y_attractor_test == 1]).mean() for m in models]
non_attractor_acc = [(m[1][y_attractor_test == 0] == y_topic_test[y_attractor_test == 0]).mean() for m in models]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(model_names))
width = 0.25

ax.bar(x - width, overall_acc, width, label='Overall')
ax.bar(x, attractor_acc, width, label='With Attractors')
ax.bar(x + width, non_attractor_acc, width, label='Without Attractors')

ax.set_ylabel('Accuracy')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.legend()
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

print("Analysis complete. Visualizations saved to 'attractor_features.png' and 'model_comparison.png'")
