import pandas as pd
from imblearn.under_sampling import NeighbourhoodCleaningRule, RandomUnderSampler


def get_sampler(sampler_name):
    if sampler_name == "srs":
        return srs_sampler
    elif sampler_name == "ncr":
        return ncr_sampler
    else:
        return none_sampler


def none_sampler(data):
    return data


def srs_sampler(data):
    """
    Simple random sampling. Balances classes based on minority class cardinality.
    :param data: dataset with 'label' column
    :return: balanced dataset
    """
    srs = RandomUnderSampler(sampling_strategy="majority", random_state=7810)
    labels = data["label"]
    data.drop(columns=["label"], inplace=True)
    data, y = srs.fit_resample(data, labels)
    data["label"] = y

    return data


def ncr_sampler(data):
    """
    Uses neighboor cleanning rule sampler to undersample majority class
    :param data: dataset with 'label' column
    :return: balanced dataset
    """

    ncr = NeighbourhoodCleaningRule(sampling_strategy="majority", n_jobs=24)
    labels = data["label"]
    data.drop(columns=["label"], inplace=True)
    data, y = ncr.fit_resample(data, labels)
    data["label"] = y

    return data


def main():
    return


if __name__ == "__main__":
    main()
