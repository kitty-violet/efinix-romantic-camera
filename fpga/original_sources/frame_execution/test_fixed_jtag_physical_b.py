"""USB layout guards and transport setup, offline; existing protocol tests retained."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import fixed_jtag_physical_b as binding


class Interface(list):
    def __init__(self, number, endpoints):
        super().__init__(SimpleNamespace(bEndpointAddress=e) for e in endpoints)
        self.bInterfaceNumber = number


class Config:
    def __init__(self, numbers):
        self.bNumInterfaces = len(numbers)
        self.interfaces = [Interface(n, [0x81+2*n, 2+2*n]) for n in numbers]
    def __getitem__(self, key):
        return self.interfaces[key[0]]


class Device:
    idVendor, idProduct, bus, address = 0x0403, 0x6011, 0, 2
    def __init__(self, numbers): self.config = Config(numbers)
    def __getitem__(self, key): return self.config


class Checks(unittest.TestCase):
    def test_virtual_b(self):
        self.assertEqual(binding.check_interface(Device([1]), 1)['physical_interface'], 1)
    def test_full_device_b(self):
        self.assertEqual(binding.check_interface(Device([0,1,2,3]), 2)['physical_interface'], 1)
    def test_reject_a_c_d(self):
        for number in (0,2,3):
            with self.assertRaises(ValueError): binding.check_interface(Device([number]), 1)
            with self.assertRaises(ValueError): binding.check_interface(Device([0,1,2,3]), number+1)
    def test_reject_wrong_endpoints(self):
        d = Device([1]);d.config.interfaces[0] = Interface(1, [0x81,2])
        with self.assertRaises(ValueError): binding.check_interface(d, 1)
    def test_reject_out_of_bounds(self):
        for index in (0,2,4):
            with self.assertRaises(ValueError): binding.check_interface(Device([1]), index)
    def test_reject_wrong_chip_and_layout(self):
        d = Device([1]);d.idProduct = 0x6010
        with self.assertRaises(ValueError): binding.check_interface(d, 1)
        with self.assertRaises(ValueError): binding.check_interface(Device([0,1]), 2)
    def test_guard_before_mpsse(self):
        factory = Mock()
        with patch.object(binding.host, 'usb_runtime', return_value=(factory, Mock())), patch.object(binding, 'verify_url', side_effect=ValueError):
            with self.assertRaises(ValueError): binding.PhysicalBTransport('rejected')
        factory.assert_not_called()
    def test_setup_and_id_guard(self):
        for identity in (0x10660a79, 0x00360a79, 0):
            engine = Mock();factory = Mock(return_value=engine)
            with patch.object(binding.host, 'usb_runtime', return_value=(factory, Mock())), patch.object(binding, 'verify_url', return_value={}), patch.object(binding.PhysicalBTransport, 'scan', return_value=identity):
                if identity:
                    transport = binding.PhysicalBTransport('ftdi://ftdi:4232h:0:2/1')
                    self.assertEqual(transport.idcode, identity)
                else:
                    with self.assertRaises(IOError): binding.PhysicalBTransport('ftdi://ftdi:4232h:0:2/1')
                    engine.close.assert_called_once()
            self.assertEqual(engine.controller.ftdi.timeouts, (500,500))
            engine.configure.assert_called_once()
            engine.controller.ftdi.set_latency_timer.assert_called_once_with(1)


if __name__ == '__main__': unittest.main(verbosity=2)
