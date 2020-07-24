import pandas as pd
import numpy as np
np.random.seed(1998)

from sklearn.metrics import classification_report, accuracy_score
from itertools import repeat

import time


def get_acc(model, X, y):
    y_pred = model.predict(X)
    print(classification_report(y, y_pred))
    train_accuracy=accuracy_score(y_pred,y)
    print("Accuracy {}".format(train_accuracy))


def print_report(model, X, y, target_names, report_path):
    result_dict = {"label": [], "precision": [], "recall": [], "f1_score": [], "support": []}

    report = classification_report(y_test, y_test_pred, target_names=target_names, output_dict=True)

    not_labels = ["accuracy", "macro avg", "weighted avg"]
    eval_label_report = {key : value for key, value in report.items() if key not in not_labels}
    eval_label_report = sorted(eval_label_report.items(), key=lambda x: x[1]["f1-score"])

    print("{:<40} {:<25} {:<25} {:<25} {:<25}\n".format('Label','Precision','Recall', 'F1-score', 'Support'))
    for item in eval_label_report:
        key       = item[0]
        precision = item[1]['precision']
        recall    = item[1]['recall']
        f1_score  = item[1]['f1-score']
        support   = item[1]['support']

        result_dict["label"].append(key)
        result_dict["precision"].append(precision)
        result_dict["recall"].append(recall)
        result_dict["f1_score"].append(f1_score)
        result_dict["support"].append(support)

        print("{:<40} {:<25} {:<25} {:<25} {:<25}".format(key, precision, recall, f1_score, support))

    pd.DataFrame.from_dict(result_dict).to_excel(report_path)


def caculate_confidence(model, X, y):
    y_pp = model.predict_proba(X)

    count_more_than_90 = 0
    count_80_to_90 = 0
    count_70_to_80 = 0
    count_60_to_70 = 0
    count_50_to_60 = 0
    count_20_to_50 = 0
    count_less_than_20 = 0
    confidence_T = []
    confidence_F = []
    for i in range(len(y_pp)):
        if np.max(y_pp[i]) >= 0.9:
            count_more_than_90 += 1
        elif np.max(y_pp[i]) >= 0.8:
            count_80_to_90 += 1
        elif np.max(y_pp[i]) >= 0.7:
            count_70_to_80 += 1
        elif np.max(y_pp[i]) >= 0.6:
            count_60_to_70 += 1
        elif np.max(y_pp[i]) >= 0.5:
            count_50_to_60 += 1
        elif np.max(y_pp[i]) >= 0.2:
            count_20_to_50 += 1
        else:
            count_less_than_20 += 1
        if np.argmax(y_pp[i]) != y[i]:
            confidence_F.append(np.max(y_pp[i]))
        else:
            confidence_T.append(np.max(y_pp[i]))
    print("count_more_than_90: {}".format(count_more_than_90))
    print("count_80_to_90: {}".format(count_80_to_90))
    print("count_70_to_80: {}".format(count_70_to_80))
    print("count_60_to_70: {}".format(count_60_to_70))
    print("count_50_to_60: {}".format(count_50_to_60))
    print("count_20_to_50: {}".format(count_20_to_50))
    print("count_less_than_20: {}".format(count_less_than_20))
    print("confidence_T: {}".format(sum(confidence_T) / len(confidence_T)))
    print("confidence_F: {}".format(sum(confidence_F) / len(confidence_F)))


def predict(model, vectorizer, le, text):
    start_time = time.time()

    input_vector = vectorizer.transform([text]).A
    labels = model.predict_proba(input_vector)
    new_labels = []

    for row in labels:
        prob_per_class = list(zip(model.classes_, row))
        prob_per_class = sorted(prob_per_class)
        new_labels.append([i[1] for i in prob_per_class])
    intent_index = np.argmax(new_labels[0])

    print("Predict \'{}\' as intent {} with confidence {}".format(text, le.classes_[intent_index], np.max(new_labels[0])))
    end_time = time.time()
    print('Predict Time: ' + str(end_time - start_time))