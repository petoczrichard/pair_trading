#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>


enum class TradeType {
    Entry,
    Exit,
};

enum class BasketWeightType {
    Shares,
    Notional,
};

inline std::ostream& operator<<(
    std::ostream& os,
    const TradeType& type
) {
    switch (type) {
        case TradeType::Entry:
            return os << "TradeType.Entry";
        case TradeType::Exit:
            return os << "TradeType.Exit";
    }
}

inline std::ostream& operator<<(
    std::ostream& os,
    const BasketWeightType& weight_type
) {
    switch (weight_type) {
        case BasketWeightType::Shares:
            return os << "BasketWeightType.Shares";
        case BasketWeightType::Notional:
            return os << "BasketWeightType.Notional";
    }
}

template <typename T>
std::ostream& write_vector(std::ostream& os, const std::vector<T>& values) {
    os << "[";

    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            os << ", ";
        }

        os << values[i];
    }

    return os << "]";
}

struct Basket {
    std::vector<Eigen::Index> asset_ids;
    std::vector<double> weights;
    BasketWeightType weight_type;
};

inline std::ostream& operator<<(std::ostream& os, const Basket& basket) {
    os << "Basket(asset_ids=";
    write_vector(os, basket.asset_ids);
    os << ", weights=";
    write_vector(os, basket.weights);
    return os
        << ", weight_type=" << basket.weight_type
        << ")";
}

struct TradeSource {
    uint64_t source_id;
    TradeType type;
    Eigen::Index date;
    std::optional<Basket> basket = std::nullopt;
    double max_adv_participation_rate = 1.0;
    double max_ratio_of_portfolio_value = 1.0;

    TradeSource(
        uint64_t source_id,
        TradeType type,
        Eigen::Index date,
        std::optional<Basket> basket = std::nullopt,
        double max_adv_participation_rate = 1.0,
        double max_ratio_of_portfolio_value = 1.0
    )
        : source_id(source_id),
          type(type),
          date(date),
          basket(std::move(basket)),
          max_adv_participation_rate(max_adv_participation_rate),
          max_ratio_of_portfolio_value(max_ratio_of_portfolio_value)
    {
        if (type == TradeType::Entry && !this->basket) {
            throw std::invalid_argument(
                "Entry trades require a basket"
            );
        }

        if (type == TradeType::Exit && this->basket) {
            throw std::invalid_argument(
                "Exit trades must not have a basket"
            );
        }
    }

    std::vector<double> shares_to_buy(
        double max_position,
        const Eigen::VectorXd& prices,
        const Eigen::VectorXd& adv
    ) const {
        if (type != TradeType::Entry) {
            throw std::logic_error(
                "shares_to_buy is only valid for Entry trades"
            );
        }

        auto shares = basket->weights;

        if (basket->weight_type == BasketWeightType::Notional) {
            for (size_t i = 0; i < shares.size(); ++i) {
                shares[i] /= prices[i];
            }
        }

        double basket_price = 0.0;
        for (size_t i = 0; i < shares.size(); ++i) {
            basket_price += std::abs(shares[i]) * prices[i];
        }
        double max_baskets = max_position / basket_price;

        double biggest_participation = max_adv_participation_rate;
        for (size_t i = 0; i < shares.size(); ++i) {
            double current_participation =
                max_baskets * std::abs(shares[i]) * prices[i] / adv[i];

            if (current_participation > biggest_participation) {
                biggest_participation = current_participation;
            }
        }

        double deleveraged_baskets =
            max_adv_participation_rate / biggest_participation * max_baskets;

        std::vector<double> shares_to_buy_(shares.size());
        for (size_t i = 0; i < shares.size(); ++i) {
            shares_to_buy_[i] = deleveraged_baskets * shares[i];
        }

        return shares_to_buy_;
    }
};

inline std::ostream& operator<<(
    std::ostream& os,
    const TradeSource& trade_source
) {
    os
        << "TradeSource("
        << "source_id=" << trade_source.source_id
        << ", type=" << trade_source.type
        << ", date=" << trade_source.date
        << ", basket=";

    if (trade_source.basket.has_value()) {
        os << *trade_source.basket;
    } else {
        os << "None";
    }

    return os
        << ", max_adv_participation_rate="
        << trade_source.max_adv_participation_rate
        << ", max_ratio_of_portfolio_value="
        << trade_source.max_ratio_of_portfolio_value
        << ")";
}
