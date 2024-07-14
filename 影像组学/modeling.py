from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

class Modeling:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def build_and_evaluate_model(self, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=test_size)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return model, accuracy

if __name__ == "__main__":
    # 假设features和labels已定义
    features = np.random.rand(100, 10)
    labels = np.random.randint(2, size=100)
    
    modeling = Modeling(features, labels)
    model, accuracy = modeling.build_and_evaluate_model()
    print("Model Accuracy:", accuracy)
