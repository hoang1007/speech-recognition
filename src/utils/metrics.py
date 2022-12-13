from typing import Tuple, Union
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


def word_error_rate(
    predicted: Union[str, Tuple[str]], transcript: Union[str, Tuple[str]]
):
    if isinstance(predicted, str):
        predicted = (predicted,)
    if isinstance(transcript, str):
        transcript = (transcript,)

    pattern = r"\W+"

    err, total = 0, 0

    for pred, tgt in zip(predicted, transcript):
        pred_tokens = re.split(pattern, pred)
        tgt_tokens = re.split(pattern, tgt)
        err += levenshtein_distance(pred_tokens, tgt_tokens)
        total += len(tgt_tokens)

    return err / total


def character_error_rate(
    predicted: Union[str, Tuple[str]], transcript: Union[str, Tuple[str]]
):
    if isinstance(predicted, str):
        predicted = (predicted,)
    if isinstance(transcript, str):
        transcript = (transcript,)

    err, total = 0, 0

    for pred, tgt in zip(predicted, transcript):
        err += levenshtein_distance(pred, tgt)
        total += len(tgt)

    return err / total
