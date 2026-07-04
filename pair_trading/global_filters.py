from pair_trading.catalog import PairTradingCatalog


def _resolve_key(pair, property_name, **kwargs):
    property_ = getattr(pair, property_name)

    if callable(property_):
        property_ = property_(**kwargs)

    return property_


@PairTradingCatalog.register()
def asset_limit(pairs, maximum, sort_by, ascending=True, **kwargs):
    asset_counter = {}
    selected_pairs = []

    sorted_pairs = sorted(
        pairs,
        key=lambda pair: _resolve_key(pair, sort_by, **kwargs),
        reverse=not ascending,
    )

    for pair in sorted_pairs:
        name1_count = asset_counter.get(pair.name1, 0)
        name2_count = asset_counter.get(pair.name2, 0)

        if name1_count < maximum and name2_count < maximum:
            asset_counter[pair.name1] = name1_count + 1
            asset_counter[pair.name2] = name2_count + 1
            selected_pairs.append(pair)

    return selected_pairs


@PairTradingCatalog.register()
def asset_limit_ratio(pairs, ratio, sort_by, ascending=True, **kwargs):
    target = len(pairs) * ratio

    if not pairs or ratio <= 0:
        return []

    low, high = 1, len(pairs)
    best_pairs = []
    best_diff = float("inf")

    while low <= high:
        midpoint = (low + high) // 2

        selected = asset_limit(
            pairs=pairs,
            maximum=midpoint,
            sort_by=sort_by,
            ascending=ascending,
            **kwargs,
        )

        diff = abs(target - len(selected))

        if diff < best_diff:
            best_diff = diff
            best_pairs = selected

        if len(selected) < target:
            low = midpoint + 1
        else:
            high = midpoint - 1

    return best_pairs


@PairTradingCatalog.register()
def select_top(pairs, number, sort_by, ascending=True, **kwargs):
    return sorted(
        pairs,
        key=lambda pair: _resolve_key(pair, sort_by, **kwargs),
        reverse=not ascending,
    )[:number]
