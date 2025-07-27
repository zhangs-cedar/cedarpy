from cedar.utils import try_except


@try_except
def test_try_except():
    raise


test_try_except()
