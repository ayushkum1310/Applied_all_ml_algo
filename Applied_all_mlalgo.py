import tkinter as tk
from tkinter import filedialog
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
from sklearn.metrics import accuracy_score

class App:
    def __init__(self, master):
        self.master = master
        master.title("Machine Learning Model Accuracy")

        self.dataset_label = tk.Label(master, text="Select a dataset:")
        self.dataset_label.pack()
        # self.dataset_button = tk.Button(master, text="Choose Dataset", command=self.choose_dataset)
        # self.dataset_button.pack()

        self.algorithm_label = tk.Label(master, text="Select an algorithm:")
        self.algorithm_label.pack()
        self.algorithm_options = ["Logistic Regression", "K-Nearest Neighbors", "Decision Tree", "Random Forest", "Support Vector Machine", "Naive Bayes", "Multi-layer Perceptron","All"]
        self.algorithm_dropdown = tk.OptionMenu(master, tk.StringVar(), *self.algorithm_options)
        self.algorithm_dropdown.pack()

        self.accuracy_label = tk.Label(master, text="Accuracy")
        self.accuracy_label.pack()

        self.run_button = tk.Button(master, text="Run Model", command=self.run_model)
        self.run_button.pack()

        self.quit_button = tk.Button(master, text="Quit", command=master.quit)
        self.quit_button.pack()

    def choose_dataset(self):
        filename = filedialog.askopenfilename()
        self.dataset_label.configure(text=f"Dataset: {filename}")

    def run_model(self):
        algorithm = self.algorithm_dropdown.cget("text")
        dataset = self.dataset_label.cget("text")[9:]  # remove "Dataset: " from the label text
        if dataset == "":
            self.accuracy_label.configure(text="Please choose a dataset first.")
            return
        elif algorithm == "":
            self.accuracy_label.configure(text="Please choose an algorithm.")
            return

        # Load the dataset
        if dataset == "Iris":
            data = load_iris()
            X = data.data
            y = data.target
        else:
            filename = filedialog.askopenfilename()
            if filename.endswith('.csv'):
                data=pd.read_csv(filename)
                

            elif filename.endswith('.xlsx'):
                data=pd.read_excel(filename)
            else:
                self.accuracy_label.configure(text="Unsupported file format")
                return
            
            X=data.iloc[:,:-1]
            y=data.iloc[:,-1]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,shuffle=True,random_state=0)
        
        if len(X_train) == 0 or len(X_test) == 0:
            self.accuracy_label.configure(text="Not enough data to split. Please adjust the test_size parameter.")
            return

        # Train and evaluate the selected algorithm
        
        results = []
        
        
        if algorithm == "Logistic Regression":
            model = LogisticRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results.append((algorithm, accuracy))
        elif algorithm == "K-Nearest Neighbors":
            model = KNeighborsClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results.append((algorithm, accuracy))
        elif algorithm == "Decision Tree":
            model = DecisionTreeClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results.append((algorithm, accuracy))
        elif algorithm == "Random Forest":
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results.append((algorithm, accuracy))
        elif algorithm == "Support Vector Machine":
            model = SVC()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results.append((algorithm, accuracy))
        elif algorithm == "Naive Bayes":
            model = GaussianNB()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results.append((algorithm, accuracy))
        elif algorithm == "Multi-layer Perceptron":
            model=MLPClassifier(hidden_layer_sizes=(100,),max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results.append((algorithm, accuracy))
        elif algorithm=="All":
            logreg = LogisticRegression()
            knn = KNeighborsClassifier()
            dtc = DecisionTreeClassifier()
            rfc = RandomForestClassifier()
            svm = SVC()
            nb = GaussianNB()
            mlp = MLPClassifier()
            models_list=[logreg,knn,dtc,rfc,svm,nb,mlp]
            for model in models_list:
        
                model.fit(X_train,y_train)
                y_hat=model.predict(X_test)
                accuracy=accuracy_score(y_test,y_hat)
                results.append(accuracy)
            list1=["Logistic Regression", "K-Nearest Neighbors", "Decision Tree", "Random Forest", "Support Vector Machine", "Naive Bayes", "Multi-layer Perceptron"]


        # Update the labels
        self.accuracy_label.configure(text="")
        i=0
        if algorithm=="All":
            for result in results:
                algorithm_label = tk.Label(self.master, text=f"The accuracy of  {list1[i]} on  dataset is {result}",fg="red")
                i=i+1
                algorithm_label.pack()
        else:
            for result in results:
                algorithm_label = tk.Label(self.master, text=f"The accuracy of on the dataset is {result}",fg="red")
                i=i+1
                algorithm_label.pack()


root = tk.Tk()
app = App(root)
root.mainloop()

