import torch
from remag.losses import BarlowTwinsLoss

def test_barlow_twins_loss():
    """Test the BarlowTwinsLoss."""
    batch_size = 16
    proj_dim = 32
    torch.manual_seed(42)
    z1 = torch.randn(batch_size, proj_dim)
    z2 = z1.clone()  # Identical views

    loss_fn = BarlowTwinsLoss(lambda_param=5e-3, eps=1e-6)
    loss, stats = loss_fn(z1, z2, return_stats=True)

    # Invariance loss should be close to 0 for identical inputs
    assert stats["invariance_loss"] < 0.2

    # Check that it returns a single scalar when return_stats=False
    loss_scalar = loss_fn(z1, z2)
    assert isinstance(loss_scalar, torch.Tensor)
    assert loss_scalar.ndim == 0

    # Different inputs should have higher invariance loss
    z3 = torch.randn(batch_size, proj_dim)
    loss_diff, stats_diff = loss_fn(z1, z3, return_stats=True)
    assert stats_diff["invariance_loss"] > stats["invariance_loss"]
