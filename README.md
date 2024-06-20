# House Price Estimator
This project is a Ridge regression model used to estimate house prices. 
It was developed for a Kaggle competition.

## How it Works
1. The code reads in the training and testing data from CSV files.
2. It encodes the categorical features using OneHotEncoder from scikit-learn.
3. The encoded data is combined with the numerical features, and any missing values are filled with 0.
4. The training data is split into features (X_train) and target variable (y_train).
5. A Ridge regression model is created with an alpha value of 10.
6. The model is trained on the training data using the fit() method.
7. The trained model is saved to a file named "model.joblib" using joblib.
8. The performance of the model is evaluated on the training data using cross-validation with two metrics: negative root mean squared error and R-squared score.
9. Predictions are made on the testing data using the predict() method.
10. The predictions are written to a CSV file named "submission.csv".

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
