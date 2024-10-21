"""
membership attack inference algorithms
"""

import numpy as np
from sklearn import linear_model, model_selection


def mia_true_nonmember_rate(
    mia_data_X: np.ndarray,
    mia_data_Y: np.ndarray,
    forget_X: np.ndarray,
    forget_Y: np.ndarray,
    attack_model=None,
):
    """Calculates the MIA (Membership Inference Attack) scores.

    MIA-fpr = TN / |Df|
    where TN refers to the number of the forgetting samples (label = 0) predicted as non-training examples (label = 0),
    and |Df| refers to the size of the forgetting dataset.

    Parameters:
    - mia_data_X (np.ndarray): the MIA training data.
    - mia_data_Y (np.ndarray): Labels for the MIA training data (should only contain 0 and 1).
    - forget_X (np.ndarray):  the forgetting data.
    - forget_Y (np.ndarray): Labels for the forgetting data (should only contain 0).

    Returns:
    - float: MIA score (proportion of TN in the forgetting dataset).
    """

    # Check that mia_data_Y contains only 0s and 1s
    if not np.all(np.isin(mia_data_Y, np.array([0, 1]))):
        raise ValueError("mia_data labels should only have 0 and 1s")

    # Initialize the attack model
    if not attack_model:
        attack_model = linear_model.LogisticRegression()
        # Fit the attack model on the MIA training data
        attack_model.fit(mia_data_X, mia_data_Y)

    # Predict on the forgetting data
    forget_pred = attack_model.predict(forget_X)

    # Calculate confusion matrix components
    tn, fp, fn, tp = confusion_matrix(forget_Y, forget_pred)

    # Calculate and return the MIA score
    # return fp / len(forget_Y)
    return tn / len(forget_Y)


def confusion_matrix(y_true, y_pred):
    """
    Computes the confusion matrix for binary classification.

    Args:
      y_true : array_like of shape (n,)
        True labels.
      y_pred : array_like of shape (n,)
        Predicted labels.

    Returns:
      tn, fp, fn, tp : int
        True negatives, false positives, false negatives, and true positives.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Print the confusion matrix components and derived metrics
    # for header in ["tp", "fp", "tn", "fn"]:
    #     print("{:<10}".format(header), end="|")
    # print()
    # for value in [tp, fp, tn, fn]:
    #     print("{:<10.3f}".format(value), end="|")
    # print()

    return tn, fp, fn, tp


def mia_crossval(
    mia_data_X: np.ndarray,
    mia_data_Y: np.ndarray,
    forget_X: np.ndarray,
    forget_Y: np.ndarray,
    n_splits=10,
    random_state=0,
):
    """
    Computes cross-validation average accuracy of a membership inference attack.

    MIA-score = predicted member / actual members
    apply this formula to forgotten set
    forget_MIA-rate = FP / |Df|
    where FP refers to the number of the forget samples (label = 0) predicted as training member examples (label = 1),
    and |Df| refers to the size of the forgetting set

    Parameters:
    - mia_data_X (np.ndarray): the MIA training data, include members and non-members.
    - mia_data_Y (np.ndarray): Labels for the MIA training data (should only contain 0 and 1).
    - forget_X (np.ndarray):  the forgetting data.
    - forget_Y (np.ndarray): Labels for the forgetting data (should only contain 0, becuase is indeeded should be non-members).
    - n_splits: int
        number of splits to use in the cross-validation.

    Returns:

    """

    if not np.all(np.isin(np.unique(forget_Y), np.array([0, 1]))):
        raise ValueError("forget labels should only have 0 and 1s")
    if not np.all(np.isin(np.unique(mia_data_Y), np.array([0, 1]))):
        raise ValueError("mia data labels should only have 0 and 1s")

    attack_model = linear_model.LogisticRegression()

    # use stratified cross-validation for imbalanced dataset
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )

    tn_crossval, fp_crossval, fn_crossval, tp_crossval = 0, 0, 0, 0
    tn_forget, fp_forget, fn_forget, tp_forget = 0, 0, 0, 0
    for train_idx, test_idx in cv.split(mia_data_X, mia_data_Y):
        train_X, test_X = mia_data_X[train_idx], mia_data_X[test_idx]
        train_Y, test_Y = mia_data_Y[train_idx], mia_data_Y[test_idx]

        attack_model.fit(train_X, train_Y)

        test_preds = attack_model.predict(test_X)
        forget_preds = attack_model.predict(forget_X)

        # Calculate confusion matrix components
        results = confusion_matrix(forget_Y, forget_preds)
        tn_forget, fp_forget, fn_forget, tp_forget = [
            cur + next
            for cur, next in zip((tn_forget, fp_forget, fn_forget, tp_forget), results)
        ]

        results = confusion_matrix(test_Y, test_preds)
        tn_crossval, fp_crossval, fn_crossval, tp_crossval = [
            cur + next
            for cur, next in zip(
                (tn_crossval, fp_crossval, fn_crossval, tp_crossval), results
            )
        ]

    # Print the confusion matrix components and derived metrics
    # for header in ["tp", "fp", "tn", "fn"]:
    #     print("{:<10}".format(header), end="|")
    # print()
    # for value in [tp_crossval, fp_crossval, tn_crossval, fn_crossval]:
    #     print("{:<10}".format(value), end="|")
    # print()

    # # Print the confusion matrix components and derived metrics
    # for header in ["tp", "fp", "tn", "fn"]:
    #     print("{:<10}".format(header), end="|")
    # print()
    # for value in [tp_forget, fp_forget, tn_forget, fn_forget]:
    #     print("{:<10}".format(value), end="|")
    # print()

    # Calculate and return the MIA score

    # true predicted rate
    val_false_rate = (fn_crossval + fp_crossval) / (
        tn_crossval + fp_crossval + tp_crossval + fn_crossval
    )

    #  false predicted members / actual non-members
    val_false_member_rate = fp_crossval / (tn_crossval + fp_crossval)
    #  true predicted nonmembers / actual non-members
    val_true_nonmember_rate = tn_crossval / (tn_crossval + fp_crossval)

    forget_false_member_rate = fp_forget / (tn_forget + fp_forget)
    forget_true_nonmember_rate = tn_forget / (tn_forget + fp_forget)

    # For a successful membership inference attack
    # High true nonmember rate: The attack should correctly identify most non-members samples as non-members.

    print(
        f"cross validation:\n \
            true nonmember / actual nonmembers is {val_true_nonmember_rate:.3f}.\n \
            total false rate is {val_false_rate:.3f}"
    )
    if val_true_nonmember_rate > 0.9:
        print(
            "successful membership inference attack on cross validation, This attack model can be appllied to forgotten data"
        )
    else:
        print(
            "This membership inference attack on cross validation is unsuccessful, so it should be appllied to forgotten data"
        )

    print(
        f"Average true non-member / actual non-members of cross validation is {forget_true_nonmember_rate:.3f}\n"
    )
    return attack_model, forget_true_nonmember_rate
