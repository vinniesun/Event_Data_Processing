from process_events import *
import unittest

WIDTH = 240
HEIGHT = 180

class TestEvent(unittest.TestCase):
    def __init__(self):
        super().__init__()

    def test_list_equal(self, a, b):
        return self.assertListEqual(a, b)

    def test_scalar_equal(self, a, b):
        return self.assertEqual(a, b)

    def test_tuple_equal(self, a, b):
        return self.assertTupleEqual(a, b)


def main():
    test_array = Events(10, WIDTH, HEIGHT)
    test_array.events[0] = (100, 100, 100, True)
    test_array.events[1] = (101, 101, 101, False)
    test_array.events[2] = (99, 101, 101, False)

    tests = TestEvent()

    tests.test_scalar_equal(len(test_array.events), 10)
    tests.test_scalar_equal(test_array.events["t"].all(), np.array([100, 101, 99, 0, 0, 0, 0, 0, 0, 0]).all())
    tests.test_scalar_equal(test_array.events[0]["t"], 100)

    aedat_file = "/Users/vincent/Desktop/CityUHK/EBBINNOT/EBBINNOT_AEDAT4/Recording/20180711_Site1_3pm_12mm_01.aedat4"
    bin_file = "/Users/vincent/Desktop/CityUHK/EBBINNOT/DAVIS_Events/Recording/20180711_Site1_3pm_12mm_01.bin"

    current = aedat_to_events(aedat_file)
    print(current.events[0])
    current = bin_to_events(bin_file)
    print(current.events[0])

if __name__ == "__main__":
    main()