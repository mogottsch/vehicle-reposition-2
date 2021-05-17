from typing import Any, Callable, List
from timeit import default_timer as timer


class MeasureTimeTrait(object):
    # list of methods, which should NOT be measured
    exclude_methods: List = []

    # list of methods, which should be measured
    # using include_methods overwrites exlcude_methods, so that
    # every method, which is not in include_methods is excluded by default
    include_methods: List = []

    def __getattribute__(self, name: str) -> Any:
        attribute = object.__getattribute__(self, name)
        if not callable(attribute):
            return attribute

        if (not self.include_methods and name not in self.exclude_methods) or (
            self.include_methods and name in self.include_methods
        ):
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
