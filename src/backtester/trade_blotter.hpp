#pragma once

#include <vector>

#include "trade.hpp"


class TradeBlotter {
public:
    void add_trade(
        int source_id,
        int ticker,
        int date,
        double amount)
    {
        trades_.push_back({
            next_trade_id_++,
            source_id,
            ticker,
            date,
            amount
        });
    }

    const Trade& get_trade(int trade_id) const {
        return trades_.at(trade_id);
    }

    const std::vector<Trade>& trades() const {
        return trades_;
    }

private:
    int next_trade_id_ = 0;
    std::vector<Trade> trades_;
};
