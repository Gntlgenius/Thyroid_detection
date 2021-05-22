import pytest


class NotInRange(Exception):
    def __init__(self, message="values not in range"):
        self.message = message
        super().__init__(self.message)


def check_range():
    val = 5
    with pytest.raises(NotInRange):
        if val not in range(10, 20):
            raise NotInRange


