import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

from loss import PairwiseRankingLoss, L2RegularizationLoss
from model import LinearRankingModel, MLPRankingModel, SoftConditionalRankingModel, MLPRouterModel
from data import TranslationRankingDataset
from evaluation import evaluate_metrics


def train_model(model, train_dataset, val_dataset, num_epochs=10, batch_size=32, learning_rate=0.01, margin=0.1, lambda_reg=0.01):
    """
    Train the linear ranking model using pairwise ranking loss and L2 regularization.

    Args:
        model: The linear ranking model to be trained.
        train_dataset: A dataset containing pairs of translations and their metric scores.
        val_dataset: A dataset for validation to monitor performance during training.
        num_epochs: Number of training epochs.
        batch_size: Size of each training batch. Balance between stability and speed.
        learning_rate: Learning rate for the optimizer. This controls how much to update the model's weights.
        margin: Margin for the pairwise ranking loss.
        lambda_reg: Regularization strength for L2 regularization.

    Returns:
        The trained model.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # The optimizer is responsible for updating the model's weights based on the computed gradients.
    # Adam is a popular choice, and was developed by UofT!
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize the loss functions
    ranking_loss_fn = PairwiseRankingLoss(margin=margin)
    regularization_loss_fn = L2RegularizationLoss(lambda_reg=lambda_reg)

    # An epoch is one pass through the entire training dataset.
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        # Instead of iterating over each row, we batch together batch_size rows, and calculate their loss together.
        # This is more efficient and leads to more stable training.
        for batch in train_loader:
            # Each batch is represented as a tuple of two tensors:
            # one for the correct translations and one for the incorrect translations.
            features_correct, features_incorrect = batch

            # Compute scores for correct and incorrect translations
            scores_correct = model.forward(features_correct)
            scores_incorrect = model.forward(features_incorrect)

            # Calculate the sum of ranking loss and regularization loss
            ranking_loss = ranking_loss_fn(scores_correct, scores_incorrect)
            regularization_loss = regularization_loss_fn(model)
            loss = ranking_loss + regularization_loss

            # Compute the gradients (loss w.r.t. model parameters) and update the model's weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # This represents the average training loss for this epoch. Training loss always decreases.
        avg_loss = total_loss / len(train_loader)

        # Validation step. Validation loss may decrease at first but can increase if the model starts to "overfit".
        # We want to stop training when validation loss starts to increase.
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                features_correct, features_incorrect = batch
                scores_correct = model.forward(features_correct)
                scores_incorrect = model.forward(features_incorrect)
                ranking_loss = ranking_loss_fn(scores_correct, scores_incorrect)
                regularization_loss = regularization_loss_fn(model)
                val_loss += (ranking_loss + regularization_loss).item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    return model


def evaluate_model(model, test_dataset, label_column, metric_columns):
    """
    Evaluate the trained model on the test dataset and compare its performance to individual metrics.
    We return the pairwise ranking accuracy,
    i.e. the proportion of times the model correctly ranks the correct translation higher than the incorrect translation.

    Args:
        model: The trained linear ranking model.
        test_dataset: A dataset containing pairs of translations and their metric scores for testing.
        label_column: The name of the column in the test dataset that contains the labels ('correct' or 'incorrect').
        metric_columns: List of column names that contain the metric scores.

    Returns:
        A dictionary containing the evaluation results for each metric and the meta-metric.
    """
    # First, we want to get the meta-metric scores for the test dataset using our trained model.
    test_data = {
        "label": [],
        "meta_metric": []
    }
    for metric in metric_columns:
        test_data[metric] = []

    for i in range(len(test_dataset)):
        features_correct, features_incorrect = test_dataset[i]
        score_correct = model.forward(features_correct.unsqueeze(0)).item()
        score_incorrect = model.forward(features_incorrect.unsqueeze(0)).item()

        test_data["label"].append("correct")
        test_data["meta_metric"].append(score_correct)
        for j, metric in enumerate(metric_columns):
            test_data[metric].append(features_correct[j].item())

        test_data["label"].append("incorrect")
        test_data["meta_metric"].append(score_incorrect)
        for j, metric in enumerate(metric_columns):
            test_data[metric].append(features_incorrect[j].item())

    test_df = pd.DataFrame(test_data)
    return evaluate_metrics(test_df, label_column=label_column, metrics=metric_columns + ["meta_metric"])


if __name__ == "__main__":
    metrics = ["bleu-score", "chrf-score", "ter-score", "bertscore", "bleurt", "comet"]
    file_name = "metric_data_2_with_labels.csv"
    label_column = "label"  # The name of the column in the CSV containing labels
    error_types = sorted(
        {'addition', 'do-not-translate', 'mistranslation', 'omission', 'overtranslation', 'punctuation',
         'real-world-knowledge', 'undertranslation', 'untranslated', 'wrong-language'})
    model_configuration = 3 # Change this variable to switch between different model architectures (1 = linear, 2 = MLP, 3 = soft conditional)
    soft_configuration = 2  # Change this variable to switch between different configurations (1, 2, 3, or 4)

    if model_configuration == 1:
        model = LinearRankingModel()
        output_model_name = "linear_ranking_model.pth"
    elif model_configuration == 2:
        model = MLPRankingModel()
        output_model_name = "mlp_ranking_model.pth"
    elif model_configuration == 3:
        # Soft conditional ranking model is a more complex model that can learn to route different types of mistranslations to different sub-models
        if soft_configuration >= 3:
            router = MLPRouterModel(n_metrics=len(metrics), hidden_size=16, n_error_types=len(error_types))
            router.load_state_dict(torch.load("router_model.pth"))
        if soft_configuration % 2 == 0:
            ranking_models = []
            for error_type in error_types:
                model = LinearRankingModel()
                model.load_state_dict(torch.load(f"linear_ranking_model_{error_type}.pth"))
                ranking_models.append(model)
        if soft_configuration == 1:
            # Configuration 1: No pre-trained models
            model = SoftConditionalRankingModel(n_metrics=len(metrics), hidden_size=16, n_error_types=10)
        elif soft_configuration == 2:
            # Configuration 2: Pre-trained ranking models
            model = SoftConditionalRankingModel(n_metrics=len(metrics), hidden_size=16, n_error_types=10, ranking_models=ranking_models)
        elif soft_configuration == 3:
            # Configuration 3: Pre-trained router
            model = SoftConditionalRankingModel(n_metrics=len(metrics), hidden_size=16, n_error_types=len(error_types), router=router)
        elif soft_configuration == 4:
            # Configuration 4: Pre-trained ranking models and router
            model = SoftConditionalRankingModel(n_metrics=len(metrics), ranking_models=ranking_models, router=router)
            # In this case, there is nothing to train
            trained_model = model
        output_model_name = "soft_conditional_ranking_model.pth"

    dataset = TranslationRankingDataset(csv_file=file_name, label_column="label", metric_columns=metrics)
    # Split the dataset into train - val - test sets (80% train, 10% val, 10% test)
    train, temp = train_test_split(dataset, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    print(f"Train size: {len(train)}, Val size: {len(val)}, Test size: {len(test)}")

    hyperparameters = {
        'num_epochs': 20,   # Number of passes through the dataset
        'batch_size': 64,   # Number of rows to process together. Larger size -> more stable but slower updates.
        'learning_rate': 0.0001,    # How much to update the weights. Too high -> unstable training, too low -> slow.
        'margin': 0.1,  # Minimum difference we want between the score for the correct and incorrect translations. Larger margin -> more aggressive ranking, smaller margin -> more lenient.
        'lambda_reg': 0.001  # Strength of penalty for large weights. Too high -> underfit (i.e. doesn't perform well on training data), too low -> overfit (i.e. performs well on training data but poorly on unseen data). Both are equally bad.
    }
    # Train the model
    if soft_configuration != 4:
        trained_model = train_model(model, train, val, **hyperparameters)
    torch.save(trained_model.state_dict(), output_model_name)

    # Evaluate each metric, and our new meta-metric, on the test set.
    results = evaluate_model(trained_model, test, label_column=label_column, metric_columns=metrics)
    for metric, result in results.items():
        print(f"Metric: {metric}")
        print(f"  Pairwise Ranking Accuracy: {result['pairwise_accuracy']:.4f}")
        print(f"  AUC-ROC: {result['auc_roc']:.4f}")
        print(f"  Mean Margin: {result['mean_margin']:.4f}")
        print(f"  Paired t-test p-value: {result['p_value']:.4e}")
