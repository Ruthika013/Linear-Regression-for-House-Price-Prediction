"""
Linear Regression for House Price Prediction (Area -> Price)

Part A: the 5-row sample from the problem statement
Part B: the full 2000-row dataset (House Price Prediction Dataset.csv)

Run:  python house_price_linear_regression.py [path/to/dataset.csv]
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_theme(style="whitegrid")
CSV = sys.argv[1] if len(sys.argv) > 1 else "House Price Prediction Dataset.csv"


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def show(title, m):
    print(f"\n{title}")
    for k, v in m.items():
        print(f"  {k:5s}: {v:,.4f}")


def plot_fit(df, model, title, fname, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.scatterplot(data=df, x="Area", y="Price", ax=ax, alpha=0.6 if len(df) > 50 else 1, s=40 if len(df) > 50 else 70)
    xs = pd.DataFrame({"Area": np.linspace(df["Area"].min(), df["Area"].max(), 100)})
    ax.plot(xs["Area"], model.predict(xs), color="crimson", lw=2, label="Regression line")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- Part A
print("=" * 60)
print("PART A: 5-row sample from the problem statement")
print("=" * 60)
sample = pd.DataFrame({"Area": [1400, 1600, 1700, 1875, 1100],
                       "Price": [245, 312, 279, 308, 199]})
X, y = sample[["Area"]], sample["Price"]
m = LinearRegression().fit(X, y)
print(f"Equation: Price = {m.intercept_:.4f} + {m.coef_[0]:.4f} * Area")
show("Fit on all 5 points (training fit)", evaluate(y, m.predict(X)))
# With 5 points a train/test split is meaningless, so use leave-one-out CV
loo_pred = cross_val_predict(LinearRegression(), X, y, cv=LeaveOneOut())
show("Leave-one-out cross-validation (honest estimate)", evaluate(y, loo_pred))
for a in (1500, 2000):
    print(f"Predicted price for {a} sq.ft: {m.predict(pd.DataFrame({'Area': [a]}))[0]:.2f}")
plot_fit(sample, m, "Sample data: Area vs Price", "sample_fit.png", "Area (sq.ft)", "Price (as given)")

# ---------------------------------------------------------------- Part B
print("\n" + "=" * 60)
print("PART B: full dataset")
print("=" * 60)
df = pd.read_csv(CSV)
print(f"Rows: {len(df)}, missing values: {int(df.isna().sum().sum())}, duplicates: {int(df.duplicated().sum())}")
print(f"Correlation(Area, Price) = {df['Area'].corr(df['Price']):.4f}")

X, y = df[["Area"]], df["Price"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
m2 = LinearRegression().fit(X_tr, y_tr)
print(f"Equation: Price = {m2.intercept_:,.2f} + {m2.coef_[0]:,.4f} * Area")
show("Train", evaluate(y_tr, m2.predict(X_tr)))
show("Test", evaluate(y_te, m2.predict(X_te)))
baseline = np.full(len(y_te), y_tr.mean())
show("Baseline (always predict mean price) on test", evaluate(y_te, baseline))
for a in (1500, 2500, 4000):
    print(f"Predicted price for {a} sq.ft: {m2.predict(pd.DataFrame({'Area': [a]}))[0]:,.0f}")

plot_fit(df, m2, "Full dataset: Area vs Price", "dataset_fit.png", "Area (sq.ft)", "Price")

res = y_te - m2.predict(X_te)
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.histplot(res, bins=30, kde=True, ax=ax)
ax.set(title="Residuals (test set)", xlabel="Actual - Predicted")
fig.tight_layout(); fig.savefig("dataset_residuals.png", dpi=150); plt.close(fig)

# Which features actually relate to price? (context for the near-zero R2)
print("\nMean price by Location / Condition / Garage:")
for c in ("Location", "Condition", "Garage"):
    print(df.groupby(c)["Price"].mean().round(0).to_string(), "\n")
num = df[["Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt", "Price"]].corr()["Price"].drop("Price")
print("Correlation of numeric features with Price:\n", num.round(4).to_string())
# ---- Predict with the sample model (Part A) ----
user_area = float(input("\nEnter house area in sq.ft: "))
pred = m.predict(pd.DataFrame({"Area": [user_area]}))[0]
print(f"Predicted price for {user_area:.0f} sq.ft = {pred:.2f}")
