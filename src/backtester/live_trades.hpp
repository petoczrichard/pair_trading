#pragma once

#include <unordered_map>

#include "trade.hpp"


class LiveTrades {
public:
    void add(int source_id, Trade trade) {
        mapping_[source_id].push_back(trade);
    }

    std::vector<Trade>& get(int source_id) {
        return mapping_.at(source_id);
    }

    void erase(int source_id) {
        auto it = mapping_.find(source_id);
        mapping_.erase(it);
    }

private:
    std::unordered_map<int, std::vector<Trade>> mapping_;
};
