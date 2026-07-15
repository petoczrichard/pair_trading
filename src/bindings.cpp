#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <sstream>
#include <vector>

#include "backtester/trade.hpp"
#include "backtester/trade_source.hpp"
#include "backtester/transaction_cost_engine.hpp"
#include "backtester/backtester_engine.hpp"


namespace nb = nanobind;

template <typename T>
std::string repr(const T& value) {
    std::ostringstream os;
    os << value;
    return os.str();
}


NB_MODULE(trading_core, m) {
    nb::class_<Trade>(m, "Trade")
        .def(nb::init<>())
        .def_rw("trade_id", &Trade::trade_id)
        .def_rw("source_id", &Trade::source_id)
        .def_rw("ticker", &Trade::ticker)
        .def_rw("date", &Trade::date)
        .def_rw("amount", &Trade::amount)
        .def("__repr__", &repr<Trade>);

    nb::enum_<TradeType>(m, "TradeType")
        .value("Entry", TradeType::Entry)
        .value("Exit", TradeType::Exit);

    nb::enum_<BasketWeightType>(m, "BasketWeightType")
        .value("Shares", BasketWeightType::Shares)
        .value("Notional", BasketWeightType::Notional);

    nb::class_<Basket>(m, "Basket")
        .def(
            nb::init<
                std::vector<Eigen::Index>,
                std::vector<double>,
                BasketWeightType
            >(),
            nb::arg("asset_ids"),
            nb::arg("weights"),
            nb::arg("weight_type")
        )
        .def_rw("asset_ids", &Basket::asset_ids)
        .def_rw("weights", &Basket::weights)
        .def_rw("weight_type", &Basket::weight_type)
        .def("__repr__", &repr<Basket>);

    nb::class_<TradeSource>(m, "TradeSource")
        .def(
            nb::init<
                uint64_t,
                TradeType,
                Eigen::Index,
                std::optional<Basket>,
                double,
                double
            >(),
            nb::arg("source_id"),
            nb::arg("type"),
            nb::arg("date"),
            nb::arg("basket") = std::nullopt,
            nb::arg("max_adv_participation_rate") = 1.0,
            nb::arg("max_ratio_of_portfolio_value") = 1.0
        )
        .def_rw("source_id", &TradeSource::source_id)
        .def_rw("type", &TradeSource::type)
        .def_rw("date", &TradeSource::date)
        .def_rw("basket", &TradeSource::basket)
        .def_rw("max_adv_participation_rate",
                &TradeSource::max_adv_participation_rate)
        .def_rw("max_ratio_of_portfolio_value",
                &TradeSource::max_ratio_of_portfolio_value)
        .def("__repr__", &repr<TradeSource>)
        .def(
            "shares_to_buy",
            [](TradeSource& self,
               double max_position,
               const Eigen::Ref<const Eigen::VectorXd>& prices,
               const Eigen::Ref<const Eigen::VectorXd>& adv) {
                return self.shares_to_buy(max_position, prices, adv);
            }
        );

    nb::class_<TransactionCostEngine>(m, "TransactionCostEngine")
        .def(
            nb::init<
                double,
                double,
                Eigen::MatrixXd,
                Eigen::MatrixXd,
                Eigen::MatrixXd,
                Eigen::MatrixXd
            >(),
            nb::arg("commission"),
            nb::arg("borrowing_cost"),
            nb::arg("spread"),
            nb::arg("slippage"),
            nb::arg("market_impact"),
            nb::arg("adv")
        )
        .def_ro("commission", &TransactionCostEngine::commission_)
        .def_ro("borrowing_cost", &TransactionCostEngine::borrowing_cost_)
        .def("spread", &TransactionCostEngine::spread)
        .def("slippage", &TransactionCostEngine::slippage)
        .def("market_impact", &TransactionCostEngine::market_impact)
        .def("short_exposure", &TransactionCostEngine::short_exposure);

    nb::class_<BacktesterEngine>(m, "BacktesterEngine")
        .def(
            nb::init<
                double,
                double,
                Eigen::MatrixXd,
                Eigen::MatrixXd,
                std::vector<TradeSource>,
                TransactionCostEngine
            >(),
            nb::arg("initial_portfolio_value"),
            nb::arg("leverage"),
            nb::arg("prices"),
            nb::arg("adv"),
            nb::arg("trade_sources"),
            nb::arg("tc_engine")
        )
        .def("run", &BacktesterEngine::run)
        .def(
            "blotter_trades",
            &BacktesterEngine::blotter_trades,
            nb::rv_policy::reference_internal
        )
        .def(
            "trade_sources",
            &BacktesterEngine::trade_sources,
            nb::rv_policy::reference_internal
        )
        .def(
            "prices",
            &BacktesterEngine::prices,
            nb::rv_policy::reference_internal
        )
        .def(
            "holdings",
            &BacktesterEngine::holdings,
            nb::rv_policy::reference_internal
        );
}
