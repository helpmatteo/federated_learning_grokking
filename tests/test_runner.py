import pytest
from fedgrok.training.runner import RunConfig

# The TestShouldAbort class that lived here tested a function no experiment ever
# called. Both have been removed -- see the note in fedgrok/training/runner.py.
# Tests that pass while guarding nothing are worse than no tests: they made the
# early-abort protocol described in the README look implemented.


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
