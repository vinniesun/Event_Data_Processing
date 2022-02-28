from src.process_events import *
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
    #bin_file = "/Users/vincent/Desktop/CityUHK/EBBINNOT/DAVIS_Events/Recording/20180711_Site1_3pm_12mm_01.bin"

    current = aedat_to_events(aedat_file)
    print(current.events[0])
    print("-----------------------------------------")

    slices = time_window_slice(current)
    #slices = event_count_slice(current)
    print(slices[0])
    print("-----------------------------------------")
    print("Total number of event is: " + str(current.num_events) + ".\n" + 
        "Total number of slices at 33ms window is: " + str(len(slices)) + ".\n" + 
        "Number of events in the first slice is: " + str(len(slices[0])) + ".\n" +
        "Number of events in the second slice is: " + str(len(slices[1])) + ".\n" +
        "Smallest Timestamp: " + str(slices[0]["t"][0]) + ", Biggest Timestamp: " + str(slices[0]["t"][-1]) + ", delta_T is: " + str(slices[0]["t"][-1] - slices[0]["t"][0]))
    print("-----------------------------------------")

    images = accumulate_and_generate(slices, WIDTH, HEIGHT)
    print(images[0].shape)
    #current = bin_to_events(bin_file)
    #print(current.events[0])

if __name__ == "__main__":
    main()