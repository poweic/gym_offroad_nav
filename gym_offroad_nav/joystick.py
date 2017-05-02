from attrdict import AttrDict
import usb.core
import usb.util

class JoystickController(object):
    def __init__(self, callback):
        # Logitech Extreme 3D Pro's vendorID and productID: 1133:49685 (i.e. 046d:c215)
        # other joystick will have different vid and pid, use `lsusb` to find
        self.device = usb.core.find(idVendor=0x046d, idProduct=0xc215)

        # use the first/default configuration
        self.device.set_configuration()

        # first endpoint
        self.endpoint = self.device[0][(0,0)][0]

        self.callback = callback

    def start(self):

        # read a data packet
        data = None
        while True:
            try:
                data = self.device.read(
                    self.endpoint.bEndpointAddress,
                    self.endpoint.wMaxPacketSize
                )

                self.callback(self.decode(data))

            except usb.core.USBError as e:
                data = None
                if e.args == ('Operation timed out',):
                    continue

    def decode(self, data):

        # this is specifically designed to decode Extreme 3D Pro's controls
        return AttrDict(
            roll = ((data[1] & 0x03) << 8) + data[0],
            pitch = ((data[2] & 0x0f) << 6) + ((data[1] & 0xfc) >> 2),
            yaw = data[3],
            view = (data[2] & 0xf0) >> 4,
            throttle = -data[5] + 255,
            buttons = [
                (data[4] & 0x01) >> 0,
                (data[4] & 0x02) >> 1,
                (data[4] & 0x04) >> 2,
                (data[4] & 0x08) >> 3,
                (data[4] & 0x10) >> 4,
                (data[4] & 0x20) >> 5,
                (data[4] & 0x40) >> 6,
                (data[4] & 0x80) >> 7,

                (data[6] & 0x01) >> 0,
                (data[6] & 0x02) >> 1,
                (data[6] & 0x04) >> 2,
                (data[6] & 0x08) >> 3
            ]
        )

def test():
    def callback(controls):
        print controls

    joystick = JoystickController(callback)
    joystick.start()


if __name__ == '__main__':
    test()
