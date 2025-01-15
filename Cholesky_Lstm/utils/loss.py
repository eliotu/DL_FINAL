import torch
import torch.nn as nn

"""RMSE"""


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


"""Penalized RMSE"""


class SignSensitiveRMSELoss(nn.Module):
    def __init__(self, lambda_weight):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_weight = lambda_weight

    def forward(self, yhat, y):
        assert isinstance(yhat, torch.Tensor), "yhat must be a tensor"
        assert isinstance(y, torch.Tensor), "y must be a tensor"
        assert yhat.shape == y.shape, "Shapes of yhat and y must match"

        mse_loss = self.mse(yhat, y)
        rmse_loss = torch.sqrt(mse_loss)

        def reconstruct_cov(cholesky_vec):
            batch_size = cholesky_vec.size(0)
            n = int(
                (
                    -1
                    + torch.sqrt(
                        torch.tensor(1 + 8 * cholesky_vec.size(1), dtype=torch.float32)
                    )
                )
                // 2
            )
            L = torch.zeros(batch_size, n, n, device=cholesky_vec.device)
            tril_indices = torch.tril_indices(row=n, col=n, device=cholesky_vec.device)
            L[:, tril_indices[0], tril_indices[1]] = cholesky_vec
            return L @ L.transpose(1, 2)

        true_cov = reconstruct_cov(y)
        pred_cov = reconstruct_cov(yhat)

        sign_diff = torch.sign(true_cov) - torch.sign(pred_cov)
        sign_penalty = torch.sum((sign_diff**2) / 4, dim=(1, 2))
        sign_penalty = torch.mean(sign_penalty)

        total_loss = rmse_loss + self.lambda_weight * sign_penalty
        return total_loss


"""Eucledian Distance"""


class EuclideanLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yhat, y):
        def reconstruct_cov(cholesky_vec):
            batch_size = cholesky_vec.size(0)
            n = int(
                (
                    -1
                    + torch.sqrt(
                        torch.tensor(1 + 8 * cholesky_vec.size(1), dtype=torch.float32)
                    )
                )
                // 2
            )
            L = torch.zeros(batch_size, n, n, device=cholesky_vec.device)
            tril_indices = torch.tril_indices(row=n, col=n, device=cholesky_vec.device)
            L[:, tril_indices[0], tril_indices[1]] = cholesky_vec
            return L @ L.transpose(1, 2)

        true_cov = reconstruct_cov(y)
        pred_cov = reconstruct_cov(yhat)
        diff = pred_cov - true_cov

        batch_size = diff.size(0)
        tril_indices = torch.tril_indices(
            row=diff.size(1), col=diff.size(2), device=diff.device
        )
        vech_diff = diff[:, tril_indices[0], tril_indices[1]]

        euclidean_distance = torch.sum(vech_diff**2, dim=1)
        return torch.mean(euclidean_distance)


class FrobeniusLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yhat, y):
        def reconstruct_cov(cholesky_vec):
            batch_size = cholesky_vec.size(0)
            n = int(
                (
                    -1
                    + torch.sqrt(
                        torch.tensor(1 + 8 * cholesky_vec.size(1), dtype=torch.float32)
                    )
                )
                // 2
            )
            L = torch.zeros(batch_size, n, n, device=cholesky_vec.device)
            tril_indices = torch.tril_indices(row=n, col=n, device=cholesky_vec.device)
            L[:, tril_indices[0], tril_indices[1]] = cholesky_vec
            return L @ L.transpose(1, 2)

        true_cov = reconstruct_cov(y)
        pred_cov = reconstruct_cov(yhat)
        frobenius_distance = torch.norm(pred_cov - true_cov, p="fro", dim=(1, 2))
        return torch.mean(frobenius_distance)


class FrobeniusLossWithSignPenalty(nn.Module):
    def __init__(self, lambda_penalty=0.1):
        """
        Initializes the FrobeniusLossWithSignPenalty.

        Parameters:
        - lambda_penalty: Regularization parameter for the sign penalty term.
        """
        super().__init__()
        self.lambda_penalty = lambda_penalty

    def forward(self, yhat, y):
        def reconstruct_cov(cholesky_vec):
            batch_size = cholesky_vec.size(0)
            n = int(
                (
                    -1
                    + torch.sqrt(
                        torch.tensor(1 + 8 * cholesky_vec.size(1), dtype=torch.float32)
                    )
                )
                // 2
            )
            L = torch.zeros(batch_size, n, n, device=cholesky_vec.device)
            tril_indices = torch.tril_indices(row=n, col=n, device=cholesky_vec.device)
            L[:, tril_indices[0], tril_indices[1]] = cholesky_vec
            return L @ L.transpose(1, 2)

        true_cov = reconstruct_cov(y)
        pred_cov = reconstruct_cov(yhat)

        frobenius_distance = torch.norm(pred_cov - true_cov, p="fro", dim=(1, 2))

        tril_indices = torch.tril_indices(
            row=true_cov.size(1), col=true_cov.size(2), device=true_cov.device
        )
        true_signs = torch.sign(true_cov[:, tril_indices[0], tril_indices[1]])
        pred_signs = torch.sign(pred_cov[:, tril_indices[0], tril_indices[1]])

        sign_diff = (true_signs - pred_signs) / 2
        sign_penalty = torch.sum(sign_diff**2, dim=1)

        total_loss = torch.mean(frobenius_distance + self.lambda_penalty * sign_penalty)
        return total_loss


class SignAccuracyLoss(nn.Module):
    def __init__(self):
        """
        Initializes the SignAccuracyLoss class.
        This loss calculates the mean accuracy of the predicted signs
        compared to the true signs for the lower triangular part of covariance matrices.
        """
        super().__init__()

    def forward(self, yhat, y):
        def reconstruct_cov(cholesky_vec):
            batch_size = cholesky_vec.size(0)
            n = int(
                (
                    -1
                    + torch.sqrt(
                        torch.tensor(1 + 8 * cholesky_vec.size(1), dtype=torch.float32)
                    )
                )
                // 2
            )
            L = torch.zeros(batch_size, n, n, device=cholesky_vec.device)
            tril_indices = torch.tril_indices(row=n, col=n, device=cholesky_vec.device)
            L[:, tril_indices[0], tril_indices[1]] = cholesky_vec
            return L @ L.transpose(1, 2)

        true_cov = reconstruct_cov(y)
        pred_cov = reconstruct_cov(yhat)

        tril_indices = torch.tril_indices(
            row=true_cov.size(1), col=true_cov.size(2), device=true_cov.device
        )

        true_signs = torch.sign(true_cov[:, tril_indices[0], tril_indices[1]])
        pred_signs = torch.sign(pred_cov[:, tril_indices[0], tril_indices[1]])

        correct_signs = (true_signs == pred_signs).float()
        sign_accuracy = torch.mean(correct_signs)

        return 1 - sign_accuracy
