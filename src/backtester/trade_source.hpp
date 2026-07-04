#pragma once

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

struct Basket {
    std::vector<Eigen::Index> asset_ids;
    std::vector<double> weights;
    BasketWeightType weight_type;
};

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
