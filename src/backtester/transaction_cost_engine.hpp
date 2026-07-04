#pragma once

#include <Eigen/Dense>
#include <cmath>


class TransactionCostEngine {
public:
    const double commission_;
    const double borrowing_cost_;

    TransactionCostEngine(
        double commission,
        double borrowing_cost,
        Eigen::MatrixXd spread,
        Eigen::MatrixXd slippage,
        Eigen::MatrixXd market_impact,
        Eigen::MatrixXd adv
    )
        : commission_(commission),
          borrowing_cost_(borrowing_cost),
          spread_(std::move(spread)),
          slippage_(std::move(slippage)),
          market_impact_(std::move(market_impact)),
          adv_(std::move(adv))
    {}

    double spread(int date, int ticker) const {
        return spread_(date, ticker);
    }

    double slippage(int date, int ticker) const {
        return slippage_(date, ticker);
    }

    double market_impact(int date, int ticker, double currency_volume) const {
        double participation_rate = currency_volume / adv_(date, ticker);
        return market_impact_(date, ticker) * std::sqrt(participation_rate);
    }

    template <typename RowA, typename RowB>
    double short_exposure_impl(
        const Eigen::ArrayBase<RowA>& weights,
        const Eigen::ArrayBase<RowB>& prices
    ) const {
        return (weights < 0.0)
            .select(weights.abs() * prices, 0.0)
            .sum();
    }

    // Python-facing concrete overload
    double short_exposure(
        const Eigen::Ref<const Eigen::VectorXd>& weights,
        const Eigen::Ref<const Eigen::VectorXd>& prices
    ) const {
        return short_exposure_impl(weights.array(), prices.array());
    }

private:
    const Eigen::MatrixXd spread_;
    const Eigen::MatrixXd slippage_;
    const Eigen::MatrixXd market_impact_;
    const Eigen::MatrixXd adv_;
};
