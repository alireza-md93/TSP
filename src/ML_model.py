import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error

with open('data/data.pkl', 'rb') as f:
    X, Y1, Y2 = pickle.load(f)

# print(sum(Y1_train))
# Train ML model
X_train, X_test, Y1_train, Y1_test = train_test_split(X, Y1, test_size=0.2, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, Y1_train)

joblib.dump(rf_classifier, 'model/rf_classifier.pkl')

# # Test model accuracy
Y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(Y1_test, Y_pred)
print("ML Model Accuracy for Branch Prioritization:", accuracy)

X_train, X_test, Y2_train, Y2_test = train_test_split(X, Y2, test_size=0.2, random_state=42)
mlp_best_cost_model = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
mlp_best_cost_model.fit(X_train, Y2_train)

joblib.dump(mlp_best_cost_model, 'model/mlp_regressor.pkl')

Y_pred = mlp_best_cost_model.predict(X_test)
mae = mean_absolute_error(Y2_test, Y_pred)
print("Mean Absolute Error for Best Achievable Cost Prediction:", mae)

