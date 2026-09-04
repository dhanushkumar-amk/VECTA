"""
Recall@k evaluation utility for approximate and exact nearest neighbor search.

Formula:
    For a given query q:
        Recall@k(q) = |Predicted_q[:k] ∩ GroundTruth_q[:k]| / k

    Averaged across all queries Q:
        Recall@k = (1 / |Q|) * Σ_{q in Q} Recall@k(q)

Worked Example:
    Query 1:
        Predicted top-3 IDs:    [10, 20, 30]
        Ground truth top-3 IDs: [10, 20, 40]
        Overlap: {10, 20} -> size = 2
        Recall@3(q1) = 2 / 3 ≈ 0.6667

    Query 2:
        Predicted top-3 IDs:    [5, 6, 7]
        Ground truth top-3 IDs: [5, 6, 7]
        Overlap: {5, 6, 7} -> size = 3
        Recall@3(q2) = 3 / 3 = 1.0

    Mean Recall@3 = (0.6667 + 1.0) / 2 = 0.8333 (83.33%)
"""

from typing import List, Sequence, Union


def recall_at_k(
    predicted_ids: Sequence[Sequence[Union[int, int]]],
    ground_truth_ids: Sequence[Sequence[Union[int, int]]],
    k: int,
) -> float:
    """
    Compute Recall@k between predicted neighbor IDs and ground truth IDs.

    Args:
        predicted_ids: List of retrieved neighbor IDs per query, sorted best-to-worst.
        ground_truth_ids: List of true nearest neighbor IDs per query, sorted best-to-worst.
        k: Number of nearest neighbors to evaluate (cutoff rank).

    Returns:
        Average Recall@k score across all queries in [0.0, 1.0].
        Returns 0.0 if either input is empty or k == 0.

    Raises:
        ValueError: If len(predicted_ids) != len(ground_truth_ids).
    """
    if k <= 0 or not predicted_ids or not ground_truth_ids:
        return 0.0

    if len(predicted_ids) != len(ground_truth_ids):
        raise ValueError(
            f"Query count mismatch: predicted ({len(predicted_ids)}) != "
            f"ground truth ({len(ground_truth_ids)})"
        )

    total_recall = 0.0

    for pred, gt in zip(predicted_ids, ground_truth_ids):
        # Take top-k slice for both
        pred_top_k = set(pred[:k])
        gt_top_k = set(gt[:k])

        # Overlap fraction
        intersection = pred_top_k.intersection(gt_top_k)
        total_recall += len(intersection) / float(k)

    return total_recall / len(predicted_ids)


if __name__ == "__main__":
    # Test 1: Worked example from docstring
    pred = [[10, 20, 30], [5, 6, 7]]
    gt = [[10, 20, 40], [5, 6, 7]]
    r = recall_at_k(pred, gt, k=3)
    expected = (2.0 / 3.0 + 3.0 / 3.0) / 2.0
    assert abs(r - expected) < 1e-6, f"Expected {expected}, got {r}"

    # Test 2: Perfect recall (k=3)
    pred_perfect = [[1, 2, 3], [4, 5, 6]]
    gt_perfect = [[1, 2, 3], [4, 5, 6]]
    assert abs(recall_at_k(pred_perfect, gt_perfect, k=3) - 1.0) < 1e-6

    # Test 3: Zero recall
    pred_zero = [[10, 11, 12]]
    gt_zero = [[1, 2, 3]]
    assert recall_at_k(pred_zero, gt_zero, k=3) == 0.0

    # Test 4: Hand-crafted prompt test: predicted=[[1,2,3]], ground_truth=[[1,2,4]], k=3 -> 2/3
    sample_pred = [[1, 2, 3]]
    sample_gt = [[1, 2, 4]]
    assert abs(recall_at_k(sample_pred, sample_gt, k=3) - (2.0 / 3.0)) < 1e-6

    print("All recall.py self-tests passed successfully!")
