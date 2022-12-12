from typing import Tuple
import re


def levenshtein_distance(source: Tuple[str], target: Tuple[str]):
    """
    Compute the Levenshtein distance between two sequences.
    """

    n, m = len(source), len(target)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        source, target = target, source
        n, m = m, n

    current_row = range(n + 1)  # Keep current and previous row, not entire matrix
    for i in range(1, m + 1):
        previous_row, current_row = current_row, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete, change = (
                previous_row[j] + 1,
                current_row[j - 1] + 1,
                previous_row[j - 1],
            )
            if source[j - 1] != target[i - 1]:
                change += 1
            current_row[j] = min(add, delete, change)

    distance = current_row[n]

    del current_row
    del previous_row

    return distance


def word_error_rate(prediction: str, transcript: str):
    pattern = r"\W+"

    prediction = re.split(pattern, prediction)
    transcript = re.split(pattern, transcript)

    return levenshtein_distance(prediction, transcript) / len(transcript)


def character_error_rate(prediction: str, transcript: str):
    return levenshtein_distance(prediction, transcript) / len(transcript)


def batch_wer(predictions: Tuple[str], transcipts: Tuple[str]):
    wer = 0

    for pred, trans in zip(predictions, transcipts):
        wer += word_error_rate(pred, trans)

    return wer / len(transcipts)


def batch_cer(predictions: Tuple[str], transcipts: Tuple[str]):
    cer = 0

    for pred, trans in zip(predictions, transcipts):
        cer += character_error_rate(pred, trans)

    return cer / len(transcipts)
