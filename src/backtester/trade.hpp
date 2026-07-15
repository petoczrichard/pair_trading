#pragma once

#include <limits>
#include <ostream>


struct Trade {
    int trade_id = std::numeric_limits<int>::max();
    int source_id;
    int ticker;
    int date;
    double amount;
};

inline std::ostream& operator<<(std::ostream& os, const Trade& trade) {
    return os
        << "Trade("
        << "trade_id=" << trade.trade_id
        << ", source_id=" << trade.source_id
        << ", ticker=" << trade.ticker
        << ", date=" << trade.date
        << ", amount=" << trade.amount
        << ")";
}
