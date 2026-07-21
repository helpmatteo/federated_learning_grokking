import pytest
from fedgrok.training.runner import should_abort, RunConfig


class TestShouldAbort:
    def test_no_abort_normal_training(self):
        assert not should_abort(step=5000, train_acc=100.0, test_acc=1.0, t_base=8000, t_max=30000)

    def test_abort_memorization_failure(self):
        assert should_abort(step=16000, train_acc=30.0, test_acc=1.0, t_base=8000, t_max=30000)

    def test_no_abort_memorization_before_deadline(self):
        assert not should_abort(step=10000, train_acc=30.0, test_acc=1.0, t_base=8000, t_max=30000)

    def test_abort_generalization_hopeless(self):
        assert should_abort(step=30000, train_acc=100.0, test_acc=2.0, t_base=8000, t_max=30000)

    def test_no_abort_generalization_progressing(self):
        assert not should_abort(step=30000, train_acc=100.0, test_acc=10.0, t_base=8000, t_max=30000)

    def test_no_abort_not_fully_memorized(self):
        """99.5% train acc should NOT trigger rule 2 (threshold is 99.9)."""
        assert not should_abort(step=30000, train_acc=99.5, test_acc=2.0, t_base=8000, t_max=30000)


class TestRunConfig:
    def test_s_fl_formula(self):
        rc = RunConfig(t_base=8000, t_max=30000)
        assert rc.s_fl == 45000

    def test_s_rescue_formula(self):
        rc = RunConfig(t_base=8000, t_max=30000)
        assert rc.s_rescue == 60000

    def test_s_fl_cap(self):
        rc = RunConfig(t_base=8000, t_max=50000)
        assert rc.s_fl == 50000

    def test_s_rescue_cap(self):
        rc = RunConfig(t_base=8000, t_max=50000)
        assert rc.s_rescue == 80000
