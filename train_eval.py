#!/usr/bin/env python
"""Homograph disambiguation training and evaluation."""

import csv
import glob
import logging
import statistics

from typing import Dict, List, Tuple

import nltk  # type: ignore
import sklearn.feature_extraction  # type: ignore
import sklearn.linear_model  # type: ignore


FeatureVector = Dict[str, str]
FeatureVectors = List[FeatureVector]


TRAIN_TSV = "data/train/*.tsv"


def _token_feature(tokens: List[str], index: int) -> str:
    if index < 0:
        return "[$]"
    if index >= len(tokens):
        return "[^]"
    else:
        token = tokens[index]
        return "[NUMERIC]" if token.isnumeric() else token.casefold()


def extract_features(
    sentence: str, homograph: str, start: int, end: int
) -> FeatureVector:
    """Extracts  feature vector for a single sentence."""
    # There is some tricky stuff to find the target homograph word here.
    sentence_b = sentence.encode("utf8")
    left = sentence_b[:start]
    target = b"^" + sentence_b[start:end] + b"^"
    right = sentence_b[end:]
    sentence = (left + target + right).decode("utf8")
    tokens = nltk.word_tokenize(sentence)
    t = -1
    for (i, token) in enumerate(tokens):
        if token.count("^") == 2:
            t = i
            break
    assert t != -1, f"target homograph {homograph!r} not found"
    target = tokens[t].replace("_", "")
    # Now onto feature extraction.
    features: Dict[str, str] = {}
    # TODO: add features to the feature dictionary here using `token`, its
    # index `t`, and the list of tokens `tokens`.
    features["t-1"] = _token_feature(tokens, t - 1)
    features["t-2"] = _token_feature(tokens, t - 2)
    features["t+1"] = _token_feature(tokens, t + 1)
    features["t+2"] = _token_feature(tokens, t + 2)
    features["t-2^t+1"] = f"{features['t-2']}^{features['t-2']}"
    features["t+1^t+2"] = f"{features['t+1']}^{features['t+2']}"
    features["t-1^t+1"] = f"{features['t-1']}^{features['t+1']}"
    if target.isupper():
        features["cap(t)"] = "upper"
    elif target.islower():
        features["cap(t)"] = "lower"
    elif target.istitle():
        features["cap(t)"] = "title"
    else:
        features["cap(t)"] = "na"
    return features


def extract_features_file(path: str) -> Tuple[FeatureVectors, List[str]]:
    """Extracts feature vectors for an entire TSV file."""
    features: FeatureVectors = []
    labels: List[str] = []
    with open(path, "r") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            labels.append(row["wordid"])
            features.append(
                extract_features(
                    row["sentence"],
                    row["homograph"],
                    int(row["start"]),
                    int(row["end"]),
                )
            )
    return features, labels


def main() -> None:
    logging.basicConfig(format="%(levelname)s: %(message)s", level="INFO")
    correct: List[int] = []
    size: List[int] = []
    for train_path in glob.iglob(TRAIN_TSV):
        vectorizer = sklearn.feature_extraction.DictVectorizer(dtype=bool)
        # Training.
        (feature_vectors, y) = extract_features_file(train_path)
        x = vectorizer.fit_transform(feature_vectors)
        model = sklearn.linear_model.LogisticRegression(
            penalty="l1",
            C=100,
            solver="liblinear",
        )
        model = sklearn.linear_model.LogisticRegression(
            penalty="l1",
            C=10,
            solver="liblinear",
        ) 
        model.fit(x, y)
        test_path = train_path.replace("/train/", "/test/")
        # Evaluation.
        (feature_vectors, y) = extract_features_file(test_path)
        x = vectorizer.transform(feature_vectors)
        yhat = model.predict(x)
        assert len(y) == len(yhat), "Mismatched lengths"
        correct.append(sum(y == yhat))
        size.append(len(y))
    # Accuracies.
    logging.info("Micro-average accuracy:\t%.4f", sum(correct) / sum(size))
    accuracies = [c / s for (c, s) in zip(correct, size)]
    logging.info("Macro-average accuracy:\t%.4f", statistics.mean(accuracies))


if __name__ == "__main__":
    main()
