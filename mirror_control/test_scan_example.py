import unittest
from unittest.mock import Mock, patch
from scan_example import scan, endpoints


class ScanTests(unittest.TestCase):
    def motor(self, **changes):
        state = dict(position=0, moving=False, referenced=True, lower=-100, upper=100)
        state.update(changes)
        motor = Mock()
        motor.status.return_value = state
        motor.goto.return_value = "job1"
        return motor

    @patch("scan_example.time.sleep")
    def test_cycles_are_absolute_and_stop(self, sleep):
        motor = self.motor(position=10)
        scan(motor, cycles=2)
        self.assertEqual([c.args[0] for c in motor.goto.call_args_list], [12, 8, 12, 8])
        self.assertEqual(motor.wait.call_count, 4)
        motor.stop.assert_called_once()
        motor.zero.assert_not_called()
        motor.set_limits.assert_not_called()

    def test_interrupt_stops(self):
        motor = self.motor()
        motor.wait.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            scan(motor)
        motor.stop.assert_called_once()

    def test_uncertain_send_stops(self):
        motor = self.motor()
        motor.goto.side_effect = TimeoutError
        with self.assertRaises(TimeoutError):
            scan(motor)
        motor.stop.assert_called_once()

    def test_preconditions_do_not_move_or_stop_other_client(self):
        for changes in (dict(moving=True), dict(referenced=False), dict(upper=1)):
            motor = self.motor(**changes)
            with self.assertRaises((ValueError, RuntimeError)):
                scan(motor)
            motor.goto.assert_not_called()
            motor.stop.assert_not_called()

    def test_resolution_and_invalid_values(self):
        self.assertEqual(endpoints(0, 3), (-2, 2, 3.6))
        for value in (0, -1, float("nan"), float("inf"), 0.1):
            with self.assertRaises(ValueError):
                endpoints(0, value)


if __name__ == "__main__":
    unittest.main()
