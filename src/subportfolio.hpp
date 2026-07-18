#pragma once

#include <cmath>
#include <cstddef>
#include <future>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include "backtester/trade.hpp"
#include "backtester/transaction_cost_engine.hpp"


inline Eigen::MatrixXd calculate_subportfolio(
    const std::vector<Trade>& trades,
    const Eigen::MatrixXd& prices,
    const TransactionCostEngine& tc_engine
) {
    constexpr Eigen::Index accounting_column_count = 6;
    if (prices.cols() < accounting_column_count) {
        throw std::invalid_argument(
            "prices must contain the six accounting columns"
        );
    }

    Eigen::MatrixXd holdings = Eigen::MatrixXd::Zero(prices.rows(), prices.cols());

    const Eigen::Index cash_col = prices.cols() - accounting_column_count;
    const Eigen::Index commission_col = cash_col + 1;
    const Eigen::Index borrowing_cost_col = cash_col + 2;
    const Eigen::Index spread_col = cash_col + 3;
    const Eigen::Index slippage_col = cash_col + 4;
    const Eigen::Index market_impact_col = cash_col + 5;

    std::vector<std::vector<const Trade*>> trades_sorted_by_date(
        static_cast<std::size_t>(holdings.rows())
    );
    for (const auto& trade : trades) {
        if (trade.date < 0 || trade.date >= holdings.rows()) {
            throw std::out_of_range("trade date is outside the price matrix");
        }
        if (trade.ticker < 0 || trade.ticker >= cash_col) {
            throw std::out_of_range("trade ticker is outside the asset columns");
        }

        trades_sorted_by_date[static_cast<std::size_t>(trade.date)].push_back(
            &trade
        );
    }

    for (Eigen::Index current_date = 1;
         current_date < holdings.rows();
         ++current_date) {
        holdings.row(current_date) = holdings.row(current_date - 1);

        for (const Trade* trade :
             trades_sorted_by_date[static_cast<std::size_t>(current_date)]) {
            const double currency_amount =
                trade->amount * prices(current_date, trade->ticker);
            const double abs_currency_amount = std::abs(currency_amount);

            holdings(current_date, trade->ticker) += trade->amount;
            holdings(current_date, cash_col) -= currency_amount;
            holdings(current_date, commission_col) +=
                abs_currency_amount * tc_engine.commission_;
            holdings(current_date, spread_col) +=
                abs_currency_amount * tc_engine.spread(current_date, trade->ticker);
            holdings(current_date, slippage_col) +=
                abs_currency_amount * tc_engine.slippage(current_date, trade->ticker);
            holdings(current_date, market_impact_col) +=
                abs_currency_amount * tc_engine.market_impact(
                    current_date,
                    trade->ticker,
                    abs_currency_amount
                );
        }

        const double short_exposure = tc_engine.short_exposure_impl(
            holdings.row(current_date).array(),
            prices.row(current_date).array()
        );
        holdings(current_date, borrowing_cost_col) +=
            short_exposure * tc_engine.borrowing_cost_ / 252;
    }

    return holdings;
}


inline std::vector<Eigen::MatrixXd> calculate_subportfolios(
    const std::vector<std::vector<Trade>>& trades,
    const Eigen::MatrixXd& prices,
    const TransactionCostEngine& tc_engine
) {
    std::vector<std::future<Eigen::MatrixXd>> futures;
    futures.reserve(trades.size());

    for (const auto& subportfolio_trades : trades) {
        const auto* subportfolio_trades_ptr = &subportfolio_trades;
        futures.push_back(std::async(
            std::launch::async,
            [subportfolio_trades_ptr, &prices, &tc_engine] {
                return calculate_subportfolio(
                    *subportfolio_trades_ptr,
                    prices,
                    tc_engine
                );
            }
        ));
    }

    std::vector<Eigen::MatrixXd> subportfolios;
    subportfolios.reserve(futures.size());
    for (auto& future : futures) {
        subportfolios.push_back(future.get());
    }

    return subportfolios;
}
