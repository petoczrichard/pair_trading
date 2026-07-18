#pragma once

#include <vector>
#include <Eigen/Dense>

#include "backtester/trade.hpp"
#include "backtester/trade_source.hpp"
#include "backtester/trade_blotter.hpp"
#include "backtester/live_trades.hpp"
#include "backtester/transaction_cost_engine.hpp"


class BacktesterEngine {
public:
    BacktesterEngine(
        double initial_portfolio_value,
        double leverage,
        Eigen::MatrixXd prices,
        Eigen::MatrixXd adv,
        std::vector<TradeSource> trade_sources,
        TransactionCostEngine tc_engine
    )
        : prices_(prices.rows(), prices.cols() + 6),

          cash_col_(prices.cols()),
          commision_col_(prices.cols() + 1),
          borrowing_cost_col_(prices.cols() + 2),
          spread_col_(prices.cols() + 3),
          slippage_col_(prices.cols() + 4),
          market_impact_col_(prices.cols() + 5),

          initial_portfolio_value_(initial_portfolio_value),
          leverage_(leverage),

          adv_(std::move(adv)),
          trade_sources_(std::move(trade_sources)),
          tc_engine_(std::move(tc_engine)),

          blotter_{},
          live_trades_{}
    {
        prices_.leftCols(prices.cols()) = std::move(prices);
        prices_.col(cash_col_).setOnes();
        prices_.rightCols(5).setConstant(-1.0);

        holdings_ = Eigen::MatrixXd::Zero(prices_.rows(), prices_.cols());
        holdings_(0, cash_col_) = initial_portfolio_value;
    }

    void run() {
        std::vector<std::vector<TradeSource>>
            exit_trade_sources_sorted_by_date(holdings_.rows());
        std::vector<std::vector<TradeSource>>
            entry_trade_sources_sorted_by_date(holdings_.rows());

        for (const auto& trade_source : trade_sources_) {
            if (trade_source.type == TradeType::Exit) {
                exit_trade_sources_sorted_by_date[trade_source.date].push_back(trade_source);
            }
            else if (trade_source.type == TradeType::Entry) {
                entry_trade_sources_sorted_by_date[trade_source.date].push_back(trade_source);
            }
        }

        for (size_t current_date = 1; current_date < holdings_.rows(); ++current_date) {
            holdings_.row(current_date) = holdings_.row(current_date - 1);

            auto holdings_row = holdings_.row(current_date);
            auto prices_row = prices_.row(current_date);
            if (!holdings_row.allFinite()) {
                throw std::runtime_error("holdings row contains NaN or infinity");
            }
            if (!prices_row.allFinite()) {
                throw std::runtime_error("prices row contains NaN or infinity");
            }
            double portfolio_value = holdings_row.dot(prices_row);

            for (const auto& trade_source : exit_trade_sources_sorted_by_date[current_date]) {
                auto trades_to_close = live_trades_.get(trade_source.source_id);

                for (const auto& trade : trades_to_close) {
                    auto amount = -trade.amount;

                    blotter_.add_trade(
                        trade.source_id,
                        trade.ticker,
                        current_date,
                        amount
                    );

                    holdings_update(amount, current_date, trade.ticker);
                }

                live_trades_.erase(trade_source.source_id);
            }

            for (const auto& trade_source : entry_trade_sources_sorted_by_date[current_date]) {
                auto asset_ids = trade_source.basket->asset_ids;

                std::vector<double> shares_to_buy = trade_source.shares_to_buy(
                    portfolio_value * leverage_ * trade_source.max_ratio_of_portfolio_value,
                    prices_(current_date, asset_ids),                    
                    adv_(current_date, asset_ids)
                );
                for (size_t i = 0; i < asset_ids.size(); ++i) {
                    auto current_asset_id = asset_ids[i];

                    blotter_.add_trade(
                        trade_source.source_id,
                        current_asset_id,
                        current_date,
                        shares_to_buy[i]
                    );
                    live_trades_.add(
                        trade_source.source_id,
                        blotter_.trades().back()
                    );
                    holdings_update(shares_to_buy[i], current_date, current_asset_id);
                }
            }
            
            double short_exposure = tc_engine_.short_exposure(
                holdings_.row(current_date),
                prices_.row(current_date)
            );
            holdings_(current_date, borrowing_cost_col_) += short_exposure * tc_engine_.borrowing_cost_ / 252;
        }

    }

    const std::vector<Trade>& blotter_trades() const {
        return blotter_.trades();
    }

    const std::vector<TradeSource>& trade_sources() const {
        return trade_sources_;
    }

    const Eigen::MatrixXd& prices() const {
        return prices_;
    }

    const Eigen::MatrixXd& holdings() const {
        return holdings_;
    }

    const TransactionCostEngine& tc_engine() const {
        return tc_engine_;
    }

private:
    const double leverage_;
    const double initial_portfolio_value_;

    const int cash_col_;
    const int commision_col_;
    const int borrowing_cost_col_;
    const int spread_col_;
    const int slippage_col_;
    const int market_impact_col_;

    Eigen::MatrixXd holdings_;
    Eigen::MatrixXd prices_;
    const Eigen::MatrixXd adv_;

    const std::vector<TradeSource> trade_sources_;
    const TransactionCostEngine tc_engine_;

    TradeBlotter blotter_;
    LiveTrades live_trades_;

    void holdings_update(double shares, size_t date, int asset_id) {
        double currency_amount = shares * prices_(date, asset_id);
        double abs_currency_amount = std::abs(currency_amount);

        holdings_(date, asset_id) += shares;
        holdings_(date, cash_col_) -= currency_amount;

        holdings_(date, commision_col_) += abs_currency_amount * tc_engine_.commission_;
        holdings_(date, spread_col_) += abs_currency_amount * tc_engine_.spread(date, asset_id);
        holdings_(date, slippage_col_) += abs_currency_amount * tc_engine_.slippage(date, asset_id);
        holdings_(date, market_impact_col_) += abs_currency_amount * tc_engine_.market_impact(date, asset_id, abs_currency_amount);
    }
};
