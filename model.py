from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
import joblib


def encode_dataframe(encoder, dataframe):
    encoded_df = pd.DataFrame()
    for col in dataframe.columns:
        if dataframe[col].dtype == "object":
            encoded_data = encoder.fit_transform(dataframe[[col]]).toarray()
            encoded_cols = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([col]),
                                        index=dataframe.index)
            encoded_df = pd.concat([encoded_df, encoded_cols], axis=1)
        else:
            encoded_df = pd.concat([encoded_df, dataframe[col]], axis=1)
    return encoded_df


train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")

train_df = encode_dataframe(OneHotEncoder(), train_df)
test_df = encode_dataframe(OneHotEncoder(), test_df)

all_cols = pd.Series(train_df.columns.tolist() + test_df.columns.tolist()).unique()
train_df = train_df.reindex(columns=all_cols).fillna(0)
test_df = test_df.reindex(columns=all_cols).fillna(0)

X_train = np.array(train_df.drop("SalePrice", axis=1))
y_train = np.array(train_df["SalePrice"])
X_test = np.array(test_df.drop("SalePrice", axis=1))

model = Ridge(alpha=10)
model.fit(X_train, y_train)

joblib.dump(model, "model.joblib")

scores = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error")
print(f"Training set negative root mean squared error: {np.mean(scores):.4f}")
scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
print(f"Training set r2 score: {np.mean(scores):.4f}")

predictions = model.predict(X_test)

with open("submission.csv", "w") as submission:
    submission.write("Id,SalePrice")
    for i, j in enumerate(predictions):
        submission.write(f"\n{i + 1461},{j}")
