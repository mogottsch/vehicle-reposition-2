from typing import Any, Callable, List
from timeit import default_timer as timer


class MeasureTimeTrait(object):
    exclude_methods: List = []

    def __getattribute__(self, name: str) -> Any:
        attribute = object.__getattribute__(self, name)
        if callable(attribute) and name not in self.exclude_methods:
            return object.__getattribute__(self, "measure_time")(attribute)
        return attribute

    def measure_time(self, cb: Callable) -> Callable:
        def decorated_callable(*args, **kwargs):
            start = timer()
            value = cb(*args, **kwargs)
            end = timer()
            print(f"{cb.__name__} finished in {end-start:.2f} seconds")
            return value

        return decorated_callable
