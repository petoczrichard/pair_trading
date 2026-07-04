#pragma once

#include <limits>


struct Trade {
    int trade_id = std::numeric_limits<int>::max();
    int source_id;
    int ticker;
    int date;
    double amount;
};
