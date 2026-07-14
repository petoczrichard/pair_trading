import inspect
import logging
import time
from functools import wraps


logger = logging.getLogger(__name__)


DEFAULT_PROPERTIES = ("__str__",)
ARGUMENTS_TO_SKIP = {'self', 'cls'}


def _resolve_dunder(name, property_name, value):
    if property_name == '__str__':
        return f"{name}={value}"
    return f"{property_name[2:-2]}({name})={value}"


def _format_arguments(bound_signature, formatter):
    for arguemnt_name, value in bound_signature.arguments.items():
        if arguemnt_name in ARGUMENTS_TO_SKIP:
            continue

        property_names = formatter.get(arguemnt_name, DEFAULT_PROPERTIES)

        for property_name in property_names:
            property_value = getattr(value, property_name)

            if callable(property_value):
                property_value = property_value()

            if property_name.startswith("__") and property_name.endswith("__"):
                yield _resolve_dunder(
                    arguemnt_name,
                    property_name,
                    property_value,
                )
            else:
                yield f"{arguemnt_name}.{property_name}={property_value}"


def logger_decorator(formatter=None):
    formatter = formatter or {}

    def decorator(func):
        signature = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()

            call_signature = ", ".join(_format_arguments(bound, formatter))

            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            except Exception:
                logger.exception(
                    "%.3fs - FAIL - %s(%s)",
                    time.perf_counter() - start,
                    func.__qualname__,
                    call_signature,
                )
                raise

            logger.info(
                "%.3fs - SUCCESS - %s(%s)",
                time.perf_counter() - start,
                func.__qualname__,
                call_signature,
            )

            return result

        return wrapper

    return decorator
