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


def _format_arguments(arguments, formatter):
    for argument_name, value in arguments.items():

        if argument_name in ARGUMENTS_TO_SKIP:
            continue

        property_names = formatter.get(argument_name, DEFAULT_PROPERTIES)

        for property_name in property_names:
            property_value = getattr(value, property_name)

            if callable(property_value):
                property_value = property_value()

            if property_name.startswith("__") and property_name.endswith("__"):
                yield _resolve_dunder(
                    argument_name,
                    property_name,
                    property_value,
                )
            else:
                yield f"{argument_name}.{property_name}={property_value}"


def logger_decorator(
    input_formatter=None,
    output_names=None,
    output_formatter=None,
):
    input_formatter = input_formatter or {}

    def decorator(func):
        signature = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()

            call_signature = ", ".join(
                _format_arguments(bound.arguments, input_formatter)
            )

            start = time.perf_counter()
            try:
                output = func(*args, **kwargs)
            except Exception:
                logger.exception(
                    "%.3fs - FAIL - %s(%s)",
                    time.perf_counter() - start,
                    func.__qualname__,
                    call_signature,
                )
                raise

            if output_names is None and output_formatter is None:
                logger.info(
                    "%.3fs - %s(%s)",
                    time.perf_counter() - start,
                    func.__qualname__,
                    call_signature,
                )

            else:
                tuple_output = (
                    output
                    if isinstance(output, tuple)
                    else (output,)
                )
                output_dict = {
                    key: value
                    for key, value
                    in zip(output_names, tuple_output)
                }
                output_string = ', '.join(
                    _format_arguments(output_dict, output_formatter)
                )

                logger.info(
                    "%.3fs - %s(%s) -> %s",
                    time.perf_counter() - start,
                    func.__qualname__,
                    call_signature,
                    output_string,
                )

            return output
        return wrapper
    return decorator
