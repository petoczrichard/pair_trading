from pair_trading.pair.abstract import AbstractPair


class DistancePair(AbstractPair):

    alias = 'distance'

    def calculate(self, zero_mean: bool = True):
        start_price1 = self.price1_values[0]
        start_price2 = self.price2_values[0]

        self.hedge_ratio = start_price1 / start_price2
        self.spread = self.create_spread()

        if zero_mean:
            self.spread_mean = 0

        self.ssd = self.calculate_ssd()

        return self
