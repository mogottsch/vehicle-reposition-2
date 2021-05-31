from typing import Any, Callable, List
from timeit import default_timer as timer


class MeasureTimeTrait(object):
    """
    An (abstract) class to measure the time methods take to run.
    To use it make the class you want to monitor inherit from this class.

    Attributes
    ----------
    exclude_methods : list
        list of methods, which should NOT be measured
    include_list : list
        list of methods, which should be measured
        using include_methods overwrites exlcude_methods, so that
        every method, which is not in include_methods is excluded by default
    """

    exclude_methods: List = []
    include_methods: List = []

    def __getattribute__(self, name: str) -> Any:
        attribute = object.__getattribute__(self, name)
        if not callable(attribute):
            return attribute

        if (not self.include_methods and name not in self.exclude_methods) or (
            self.include_methods and name in self.include_methods
        ):
            return object.__getattribute__(self, "_measure_time")(attribute)

        return attribute

    def _measure_time(self, cb: Callable) -> Callable:
        def decorated_callable(*args, **kwargs):
            start = timer()
            value = cb(*args, **kwargs)
            end = timer()
            print(f"{cb.__name__} finished in {end-start:.2f} seconds")
            return value

        return decorated_callable
