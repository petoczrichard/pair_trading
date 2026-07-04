"""
workflow consists from a number of steps
each step saves some attribute to the workflow (its "return value") that other
steps can use
the validation is done at an attribute level, so step interconnected validation
is avoided
some steps my write to disc to persist analysis
"""

class Step:
    ...


class Workflow:
    def __init__(self, steps):
        self.steps = steps


        self.data = None
        self.metadata = None

        self.period_data = None
        self.period_metadata = None
        self.period_groups = None
        self.period_trades = None

        self.result = None


        self.data_loader_step = None  # self.data out
        # set up rolling periods
        # within the period:
        self.data_cleaner_stop = None  # self.data in, self.period_data out
        self.pair_grouper_step = None  # self.data, self.metadata in, self.period_groups out
        self.pair_selection_step = None  # self.period_groups, self.period_data in, self.period_trades out

        self.backtest_step = None  # all self.period_trades in, self.result out
        self.analyser_step = None  # self.result in, writes analysis to disc
