"""Core shared modules: model, dataset, metrics, config, and training utilities."""

from core.config import Config
from core.model import GrokNet
from core.dataset import TASKS, make_dataset
from core.metrics import weight_norms, gradient_norms, compute_ipr, compute_accuracy
from core.utils import get_device, make_optimizer, make_targets_onehot
