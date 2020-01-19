import tkinter as tk
from functools import partial
import sys


print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

root = tk.Tk()
root.title('Diabetes Predictor')

def call_result(label_result, n1, n2):
    num1 = (n1.get())
    num2 = (n2.get())
    result = int(num1)+int(num2)
    label_result.config(text="Result = %d" % result)
    logreg = LogisticRegression().fit(X_train, y_train)
    print("Training set accuracy: {:.3f}".format(logreg.score(X_train, y_train)))
    print("Test set accuracy: {:.3f}".format(logreg.score(X_test, y_test)))
    return


# create string objects
Preg = tk.StringVar()
Glucose = tk.StringVar()
BloodPressure = tk.StringVar()
SkinThickness = tk.StringVar()
Insulin = tk.StringVar()
BMI = tk.StringVar()
DiabetesPedigreeFunction = tk.StringVar()
Age = tk.StringVar()
labelResult = tk.Label(root)


labelNum1 = tk.Label(root, text="Pregnancies").grid(row=1, column=0)
labelNum2 = tk.Label(root, text="Glucose").grid(row=2, column=0)
labelNum3 = tk.Label(root, text="Blood Pressure").grid(row=3, column=0)
labelNum4 = tk.Label(root, text="Skin Thickness").grid(row=4, column=0)
labelNum5 = tk.Label(root, text="Insulin").grid(row=5, column=0)
labelNum6 = tk.Label(root, text="BMI").grid(row=6, column=0)
labelNum7 = tk.Label(root, text="Pedigree Function").grid(row=7, column=0)
labelNum8 = tk.Label(root, text="Age").grid(row=8, column=0)
labelResult.grid(row=9, column=2)

entryNum1 = tk.Entry(root, textvariable=Preg).grid(row=1, column=2)
entryNum2 = tk.Entry(root, textvariable=Glucose).grid(row=2, column=2)
entryNum3 = tk.Entry(root, textvariable=BloodPressure).grid(row=3, column=2)
entryNum4 = tk.Entry(root, textvariable=SkinThickness).grid(row=4, column=2)
entryNum5 = tk.Entry(root, textvariable=Insulin).grid(row=5, column=2)
entryNum6 = tk.Entry(root, textvariable=BMI).grid(row=6, column=2)
entryNum7 = tk.Entry(
    root, textvariable=DiabetesPedigreeFunction).grid(row=7, column=2)
entryNum8 = tk.Entry(root, textvariable=Age).grid(row=8, column=2)

call_result = partial(call_result, labelResult, Preg, Glucose)

buttonCal = tk.Button(root, text="Calculate",
                      command=call_result).grid(row=9, column=1)

root.mainloop()
