from abc import ABCMeta, ABC
from typing import overload, Any


class PairTradingCatalog(type):
    catalog = {}

    def __new__(mcls, name, bases, namespace):
        cls = super().__new__(mcls, name, bases, namespace)

        mro = cls.__mro__

        if ABC in mro:
            abc_index = mro.index(ABC)
            root_class = mro[abc_index - 1]
            is_root_class = root_class == cls

            if not is_root_class:
                mcls.catalog.setdefault(root_class.alias, {})[cls.alias] = cls

        else:
            mcls.catalog[cls.alias] = cls

        return cls

    @classmethod
    def register(cls, alias=None):
        def decorator(func):
            cls.catalog[alias or func.__name__] = func
            return func
        return decorator

    @classmethod
    def get_catalog(mcls):
        return mcls.catalog

    @overload
    @classmethod
    def invoke(
        mcls,
        *,
        name: str,
        **kwargs,
    ) -> Any:
        ...

    @overload
    @classmethod
    def invoke(
        mcls,
        *,
        category: str,
        variant: str,
        **kwargs,
    ) -> Any:
        ...

    @classmethod
    def invoke(
        mcls,
        *,
        name: str | None = None,
        category: str | None = None,
        variant: str | None = None,
        **kwargs,
    ) -> Any:
        if name is not None:
            return mcls.catalog[name](**kwargs)

        if category is not None and variant is not None:
            return mcls.catalog[category][variant](**kwargs)

        raise ValueError(
            "Provide either `name` or (`category` and `variant`)."
        )


class PairTradingMeta(ABCMeta, PairTradingCatalog):
    ...
