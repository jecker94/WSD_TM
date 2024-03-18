import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from imblearn.metrics import classification_report_imbalanced


# functions to replace "all" labels
def replace_forbl(row):
    if row["Gold Standard"] == "all":
        return row["Baseline"]
    if row["Gold Standard"].count(',') >= 1:
        if row["Baseline"] in row["Gold Standard"]:
            return row["Baseline"]
        else:  # take the first if all of them do not fit
            return row["Gold Standard"].split(", ")[0]
    else:  # same value as before
        return row["Gold Standard"]


def replace_forjc(row):
    if row["Gold Standard"] == "all":
        return row["JC-sim"]
    if row["Gold Standard"].count(',') >= 1:
        if row["JC-sim"] in row["Gold Standard"]:
            return row["JC-sim"]
        else:  # take the first if all of them do not fit
            return row["Gold Standard"].split(", ")[0]
    else:  # same value as before
        return row["Gold Standard"]


def replace_forvec(row):
    if row["Gold Standard"] == "all":
        return row["Vector_hf"]
    if row["Gold Standard"].count(',') >= 1:
        if row["Vector_hf"] in row["Gold Standard"]:
            return row["Vector_hf"]
        else:  # take the first if all of them do not fit
            return row["Gold Standard"].split(", ")[0]
    else:  # same value as before
        return row["Gold Standard"]


if __name__ == '__main__':

    # load data
    data = pd.read_excel("Evaluation.xlsx",
                         header=0)
    # delete all rows with a zero
    data_new = data[~(data == 0).any(axis=1)]
    print(data_new)
    # data_new = data_new[~(data_new == "all").any(axis=1)]

    # create new dfs for all sets and change data in gold standard
    baseline = data_new[["Gold Standard", "Baseline"]]
    bl = baseline.copy().apply(replace_forbl, axis=1)
    print("Accuracy baseline: ", accuracy_score(baseline.iloc[:, 1].to_list(), bl.to_list()))
    # print(classification_report(baseline.iloc[:, 1].to_list(), bl.to_list()))
    # print(classification_report_imbalanced(baseline.iloc[:, 1].to_list(), bl.to_list()))

    jc_sim = data_new[["Gold Standard", "JC-sim"]]
    jc = jc_sim.copy().apply(replace_forjc, axis=1)
    print("Accuracy jc-sim: ", accuracy_score(jc_sim.iloc[:, 1].to_list(), jc.to_list()))
    # print(classification_report(jc_sim.iloc[:, 1].to_list(), jc.to_list()))
    # print(classification_report_imbalanced(jc_sim.iloc[:, 1].to_list(), jc.to_list()))

    vector = data_new[["Gold Standard", "Vector_hf"]]
    vec = vector.copy().apply(replace_forvec, axis=1)
    print("Accuracy vector: ", accuracy_score(vector.iloc[:, 1].to_list(), vec.to_list()))
    # print(classification_report(vector.iloc[:, 1].to_list(), vec.to_list()))
    # print(classification_report_imbalanced(vector.iloc[:, 1].to_list(), vec.to_list()))
