import torch
import torch.nn as nn
import numpy as np

class LinearRankingModel(nn.Module):
    """
    A linear model for ranking translations.

    score = w1 * bleu + w2 * chrf + w3 * ter + w4 * bertscore + w5 * bleurt + w6 * comet
    """
    def __init__(self, n_metrics=6):
        super(LinearRankingModel, self).__init__()
        # Initialize weights for each metric
        self.weights = nn.Parameter(torch.ones(n_metrics) / n_metrics)
        with torch.no_grad():
            self.weights.fill_(1.0 / n_metrics) # Start with equal weights for all metrics

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute the score for a batch of translations.

        Args:
            features: Tensor of shape (batch_size, 6) containing the metric scores for each translation.

        Returns:
            Tensor of shape (batch_size,) containing the computed scores.
        """
        # Matrix multiplication which gives us a vector of shape (batch_size,)
        # where each element is the weighted sum of the metrics for one translation.
        return torch.matmul(features, self.weights)


# If you want to play around, here's a neural network model!
# MLP stands for Multi-Layer Perceptron. They invent these names to sound cool.
class MLPRankingModel(nn.Module):
    """
    A neural model for ranking translations.
    Our default architecture is a 2-layer network with ReLU activation.
    The advantage MLPs is in capturing non-linear relationships between the metrics and the translation quality.
    """
    def __init__(self, n_metrics=6, hidden_size=16):
        super(MLPRankingModel, self).__init__()
        self.fc1 = nn.Linear(n_metrics, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute the score for a batch of translations.

        Args:
            features: Tensor of shape (batch_size, 6) containing the metric scores for each translation.

        Returns:
            Tensor of shape (batch_size,) containing the computed scores.
        """
        x = self.fc1(features)
        x = self.relu(x)
        x = self.fc2(x)
        return x.squeeze()


class MLPRouterModel(nn.Module):
    """
    Routes translations to the appropriate ranking model based on the metric scores.
    """
    def __init__(self, n_metrics=6, hidden_size=16, n_error_types=10):
        super(MLPRouterModel, self).__init__()
        self.fc1 = nn.Linear(n_metrics, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, n_error_types)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute the routing probabilities for a batch of translations.

        Args:
            features: Tensor of shape (batch_size, 6) containing the metric scores for each translation.

        Returns:
            Tensor of shape (batch_size, n_error_types) containing the routing probabilities for each error type.
        """
        x = self.fc1(features)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class SoftConditionalRankingModel(nn.Module):
    """
    A soft conditional ranking model that combines the router and the ranking models.
    The idea is, given the metrics for some translation, we use the router to determine the probabilities for each error type, and then we compute a weighted sum of the scores from the corresponding ranking models.
    """
    def __init__(self, n_metrics=6, hidden_size=16, n_error_types=10, ranking_models=None, router=None):
        super(SoftConditionalRankingModel, self).__init__()

        if router is None:
            self.router = MLPRouterModel(n_metrics, hidden_size, n_error_types)
        else:
            self.router = router
            self._freeze_router()

        if ranking_models is None:
            self.ranking_models = nn.ModuleList([LinearRankingModel(n_metrics) for _ in range(n_error_types)])
        else:
            self.ranking_models = nn.ModuleList(ranking_models)
            self._freeze_ranking_models()

    def _freeze_ranking_models(self):
        """
        Freeze the parameters of the ranking models, i.e. we don't update their weights when we train this model.
        This is for when we have already-trained ("pre-trained") ranking models and we just want to learn how to route to them.
        """
        print('Ranker sub-models are now frozen (not being trained)')
        for model in self.ranking_models:
            for param in model.parameters():
                param.requires_grad = False

    def _freeze_router(self):
        print('Router sub-model is now frozen (not being trained)')
        for param in self.router.parameters():
            param.requires_grad = False

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        routing_out = self.router(features)  # Shape: (batch_size, n_error_types)
        routing_probs = torch.softmax(routing_out, dim=1)
        scores = torch.stack([model(features) for model in self.ranking_models], dim=1)  # Shape: (batch_size, n_error_types)
        weighted_scores = routing_probs * scores  # Element-wise multiplication
        return weighted_scores.sum(dim=1)  # Sum over error types to get final score
