from typing import Iterator, Self
import pandas as pd

from pair_trading.catalog import PairTradingCatalog
from pair_trading.utils import date_offset


class RollingPeriod(metaclass=PairTradingCatalog):

    alias = 'rolling_period'

    def __init__(
        self,
        formation_start: str | pd.Timestamp,
        formation_end: str | pd.Timestamp,
        trading_start: str | pd.Timestamp,
        trading_end: str | pd.Timestamp,
        time_step: str,
        number_of_periods: int | None = None,
        first_valid_date: str | pd.Timestamp | None = None,
        last_valid_date: str | pd.Timestamp | None = None,
    ) -> None:
        self.formation_start = pd.Timestamp(formation_start)
        self.formation_end = pd.Timestamp(formation_end)
        self.trading_start = pd.Timestamp(trading_start)
        self.trading_end = pd.Timestamp(trading_end)

        self.date_properties = {
            "formation_start": self.formation_start,
            "formation_end": self.formation_end,
            "trading_start": self.trading_start,
            "trading_end": self.trading_end,
        }

        self.time_step = time_step
        self.number_of_periods = number_of_periods or float("inf")

        self.first_valid_date = pd.Timestamp(
            first_valid_date or pd.Timestamp.min
        )
        self.last_valid_date = pd.Timestamp(
            last_valid_date or pd.Timestamp.max
        )

    def __iter__(self) -> Iterator[Self]:
        if self.number_of_periods > 0 and self._validate_dates():
            self.number_of_periods -= 1
            yield self
        else:
            return

        while self.number_of_periods > 0:
            for key, date in self.date_properties.items():
                self.date_properties[key] = date_offset(date, self.time_step)

            if self._validate_dates():
                self.number_of_periods -= 1
                yield self
            else:
                return

    def _validate_dates(self) -> bool:
        return all(
            self.first_valid_date <= date <= self.last_valid_date
            for date in self.date_properties.values()
        )
