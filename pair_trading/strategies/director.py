engle_granger = StrategyBuilder(
    data_source=StaticDataSource(),
    data_cleaner=StandardDataCleaner(),
    valid_grouper=IndustryGrouper(),
    pair_selecter=EngleGrangerPairSelector(),
    transaction_cost_model=FixedTransactionCostModel(),
)
