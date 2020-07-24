import os
from preprocessing import get_data, vectorize_data
from gridsearchcv import train
from evaluate import get_acc, print_report, caculate_confidence, predict


# Path to file 
root_path   = os.path.dirname(__file__)

model_path  = os.path.join(root_path, "result/model.sav")
report_path = os.path.join(root_path, "result/report.xlsx")

train_file  = os.path.join(root_path, "data/train.txt")
test_file  = os.path.join(root_path, "data/test.txt")

# Get data
X_train, y_train = get_data(train_file)
X_test, y_test   = get_data(test_file)

# Vectorizer
X_train, y_train, X_test, y_test, vectorizer, le = vectorize_data(X_train, y_train, X_test, y_test)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test : {X_test.shape}\n")

print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test : {y_test.shape}\n")

print(f"Ratio: {len(X_train)/len(X_test)}")

# Training
print("Start training...")
model = train(X_train, y_train, model_path)
print("Training done!!")

#  Evaluate
## Acc
print("Train Acc")
get_acc(model, X_train, y_train)

print("\nTest Acc")
get_acc(model, X_test, y_test)

## Print result
print_report(model, X_test, y_test, le.classes_, report_path)

## Caculate Confidence
caculate_confidence(model, X_test, y_test)

## Predict
text = 'who are the actresses in the movies'
predict(model, vectorizer, le, text)


