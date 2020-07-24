# -*- coding: utf-8 -*-
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import time
import pickle

def train(X_train, y_train, model_path):
    C                = [1, 2, 5, 10]
    gamma            = [0.1]
    kernels          = ["linear"]
    num_threads      = -1
    cv_splits        = 5
    tuned_parameters = [{"C": C, "gamma": gamma, "kernel": [str(k) for k in kernels]}]

    model = GridSearchCV(
                SVC(C=1, probability=True, class_weight="balanced"),
                param_grid=tuned_parameters,
                scoring="f1_weighted",
                n_jobs=40,
                cv=cv_splits,
                verbose=1,
                iid=False,
            )

    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    print('Training Time: ' + str(end_time - start_time))

    print("Best params: {}".format(model.best_params_))

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    return model