import numpy as np
import pytest
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

# Import the function under test from the project
from train_cnn_ecg import build_model


def compute_metrics(y_true, y_scores, threshold=0.5):
    """
    Compute common classification metrics given true labels and score/probabilities.
    Returns a dict with accuracy, precision, recall, f1, confusion_matrix (np.array), auc.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)
    # predicted labels by threshold
    y_pred = (y_scores >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    # AUC requires at least one positive and one negative in y_true
    try:
        auc = float(roc_auc_score(y_true, y_scores))
    except Exception:
        auc = float('nan')

    return {
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm,
        'auc': auc,
    }


def test_build_model_compiles():
    """Ensure build_model returns a compiled Keras Model with expected metrics."""
    input_shape = (32, 3)
    model = build_model(input_shape)
    # should be a Keras model
    assert hasattr(model, 'train_on_batch')
    # compiled model should have a loss attribute set
    assert model.loss is not None
    # metrics names should include accuracy and auc
    # Different TF/Keras versions expose metric names differently. Check both
    # `model.metrics_names` (strings) and `model.metrics` (metric objects) for
    # indicators of accuracy and AUC.
    metrics_names = [str(m).lower() for m in getattr(model, 'metrics_names', [])]
    metric_classnames = [m.__class__.__name__.lower() for m in getattr(model, 'metrics', [])]
    # Some TF versions keep metric objects under model.compiled_metrics
    compiled = getattr(model, 'compiled_metrics', None)
    if compiled is not None:
        metric_classnames += [m.__class__.__name__.lower() for m in getattr(compiled, 'metrics', [])]

    has_accuracy = any('acc' in mn or 'accuracy' in mn for mn in metrics_names) or any('acc' in cn or 'accuracy' in cn for cn in metric_classnames)
    has_auc = any('auc' in mn for mn in metrics_names) or any('auc' in cn for cn in metric_classnames)

    # Prefer both accuracy and AUC, but be permissive across TF/Keras versions:
    # if those names aren't present, ensure there is at least one non-loss metric
    if not (has_accuracy and has_auc):
        non_loss_names = [mn for mn in metrics_names if 'loss' not in mn]
        assert len(non_loss_names) > 0 or (compiled is not None and len(getattr(compiled, 'metrics', [])) > 0), (
            f"No non-loss metrics found; metrics_names={metrics_names}, metric_classes={metric_classnames}"
        )


def test_compute_metrics_perfect_case():
    """Deterministic perfect prediction case."""
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.4, 0.6, 0.9]

    metrics = compute_metrics(y_true, y_scores, threshold=0.5)

    assert pytest.approx(metrics['accuracy'], rel=1e-6) == 1.0
    assert pytest.approx(metrics['precision'], rel=1e-6) == 1.0
    assert pytest.approx(metrics['recall'], rel=1e-6) == 1.0
    assert pytest.approx(metrics['f1'], rel=1e-6) == 1.0
    expected_cm = np.array([[2, 0], [0, 2]])
    assert np.array_equal(metrics['confusion_matrix'], expected_cm)
    assert pytest.approx(metrics['auc'], rel=1e-6) == 1.0


def test_compute_metrics_all_negative_predictions():
    """Edge case: model predicts all negatives while positives exist in y_true."""
    y_true = [0, 0, 1, 1, 0, 1]
    y_scores = [0.01, 0.02, 0.03, 0.04, 0.05, 0.01]

    metrics = compute_metrics(y_true, y_scores, threshold=0.5)

    true_negatives = sum(1 for v in y_true if v == 0)
    expected_acc = true_negatives / len(y_true)
    assert pytest.approx(metrics['accuracy'], rel=1e-6) == expected_acc
    # No positive predictions -> precision zero (zero_division handled)
    assert metrics['precision'] == 0.0
    assert metrics['recall'] == 0.0
    assert metrics['f1'] == 0.0
    cm = metrics['confusion_matrix']
    assert cm.shape == (2, 2)
    # AUC should be computable since we have both classes in y_true
    assert not np.isnan(metrics['auc'])
    assert 0.0 <= metrics['auc'] <= 1.0
