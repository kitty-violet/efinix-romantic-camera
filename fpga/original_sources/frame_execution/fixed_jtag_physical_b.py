"""Physical-B USB binding extension; keep frozen loader and RTL unchanged."""
from urllib.parse import urlsplit
import fixed_jtag_loader as host


def check_interface(device, logical_interface):
    if (device.idVendor, device.idProduct) != (0x0403, 0x6011):
        raise ValueError('Require FT4232H 0403:6011')
    cfg = device[0]
    if cfg.bNumInterfaces not in (1, 4) or logical_interface < 1 or logical_interface > cfg.bNumInterfaces:
        raise ValueError('Unsupported USB interface layout or logical index')
    interface = cfg[(logical_interface-1, 0)]
    endpoints = sorted(ep.bEndpointAddress for ep in interface)
    if interface.bInterfaceNumber != 1 or endpoints != [0x04, 0x83]:
        raise ValueError('Selected USB interface is not physical B (MI01, endpoints 83/04)')
    return dict(physical_interface=1, logical_interface=logical_interface,
                bus=device.bus, address=device.address, endpoints=endpoints)


def verify_url(url):
    parsed = urlsplit(url)
    parts = parsed.netloc.split(':')
    if parsed.scheme != 'ftdi' or parsed.query or len(parts) not in (3, 4) or not all(parts):
        raise ValueError('Explicit FTDI serial or bus/address URL required')
    from pyftdi.ftdi import Ftdi
    from pyftdi.usbtools import UsbTools
    import usb.core
    desc, logical = Ftdi.get_identifiers(url)
    devices = [dev for dev in usb.core.find(find_all=True, idVendor=desc.vid, idProduct=desc.pid,
                                           backend=UsbTools.find_backend())
               if dev.bus == desc.bus and dev.address == desc.address]
    if len(devices) != 1:
        raise ValueError('URL must resolve to exactly one currently connected USB device')
    return check_interface(devices[0], logical)


class PhysicalBTransport(host.PyFtdiTransport):
    def __init__(self, url):
        JtagEngine, BitSequence = host.usb_runtime()
        self.usb_binding = verify_url(url)  # Reject A/C/D before any MPSSE setup.
        self.bits = BitSequence
        self.engine = JtagEngine(trst=False, frequency=1e6)
        self.engine.controller.ftdi.timeouts = (500, 500)
        try:
            host.configure_engine(self.engine, url)
            self.engine.reset()
            self.engine.go_idle()
            self.engine.write_ir(BitSequence(3, length=5))
            self.engine.go_idle()
            self.idcode = int(self.scan(0, 32))
            if self.idcode not in (0x00360a79, 0x10660a79):
                raise IOError(f'Unexpected Ti60 IDCODE {self.idcode:08x}; no payload written')
            self.engine.write_ir(BitSequence(8, length=5))
            self.engine.go_idle()
            self.engine.sync()
        except Exception:
            self.close()
            raise


if __name__ == '__main__':
    host.PyFtdiTransport = PhysicalBTransport
    host.main()
