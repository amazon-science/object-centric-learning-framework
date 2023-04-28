# Evaluation clustering config

Configuration to evaluate a trained model for object discovery by clustering object representations.

Given a set of images, each with a set of ground truth masks and a set of object masks and
representations, we perform the following steps:

  1. Assign each object a cluster id by clustering the corresponding representations over all
     images.
  2. Merge object masks with the same cluster id on the same image to form a semantic mask.
  3. Compute IoU between masks of predicted clusters and ground truth classes over all images.
  4. Assign clusters to classes based on the IoU and a matching strategy.

In contrast to most other configurations, the clustering evaluation
configuration is defined programmatically in python.  The definition can be
found below:

```python title="ocl/cli/eval_cluster_metrics.py:EvaluationClusteringConfig"
--8<-- "ocl/cli/eval_cluster_metrics.py:EvaluationClusteringConfig"
```
