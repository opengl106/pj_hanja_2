from typing import Any, Callable, Dict, Tuple

import evaluate
import numpy as np
from torch import Tensor

seqeval = evaluate.load("seqeval")

def compute_metrics_uncurried(p: Tuple[Tensor, Tensor], labels_to_hanja_words: Dict[int, str]) -> Dict[str, Any]:
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [labels_to_hanja_words[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [labels_to_hanja_words[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def compute_metrics_currier(labels_to_hanja_words: Dict[int, str]) -> Callable[[Tuple[Tensor, Tensor]], Dict[str, Any]]:
    def compute_metrics(p: Tuple[Tensor, Tensor]) -> Dict[str, Any]:
        return compute_metrics_uncurried(p, labels_to_hanja_words)

    return compute_metrics
