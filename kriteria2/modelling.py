import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import mlflow
import mlflow.sklearn

import warnings
warnings.filterwarnings("ignore")

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Autologging Random Forest")

df = pd.read_csv("preprocess.csv")


x = df.drop("quality_label", axis=1)
y = df["quality_label"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

## Biasanya disini dilakukan sampling/scalling untuk data train

mlflow.sklearn.autolog()

with mlflow.start_run(run_name="Autologging SVM"):
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print (classification_report(y_test, y_pred))

    print (f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print (f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
    print (f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
    print (f"F1 Score: {f1_score(y_test, y_pred, average='weighted')}")