from PySide6.QtCore import QObject, Signal, Slot
from bleak import BleakClient
import time
import asyncio


class BLESensor(QObject):
    sigDebugMsg = Signal(str)
    sigBattery = Signal(dict)
    sigMotion = Signal(dict)
    sigConnection = Signal(dict)
    
    #与传感器通讯所需的服务UUID
    notify_uuid = "0000ffe4-0000-1000-8000-00805f9a34fb"
    write_uuid = "0000ffe9-0000-1000-8000-00805f9a34fb"
    #向传感器发送的信息
    voltage_req = b'\xFF\xAA\x27\x64\x00'
    freq_dict = {
        "10": b'\xFF\xAA\x03\x06\x00',
        "20": b'\xFF\xAA\x03\x07\x00',
        "50": b'\xFF\xAA\x03\x08\x00',
        "100": b'\xFF\xAA\x03\x09\x00',
        "200": b'\xFF\xAA\x03\x0A\x00',
    }

    def __init__(self, device) -> None:
        super().__init__()
        self.device = device
        self.active = True

    @Slot()
    def start(self):
        asyncio.create_task(self.connect_sensor())

    @Slot()
    def battery(self):
        asyncio.create_task(self._get_battery_percent())
        
    @Slot()
    def freq(self, freq=100):
        asyncio.create_task(self._change_freq(freq))

    @Slot()
    def stop(self):
        print(f"传感器{self.device.address}断开连接")
        self.stop_event.set()
        self.active = False
        self.sigConnection.emit({
            'address': self.device.address,
            'status': False
        })

    async def connect_sensor(self):
        self.stop_event = asyncio.Event()
        async with BleakClient(address_or_ble_device=self.device,
                                disconnected_callback=self._disconnected_callback) as self.client:
                self.sigConnection.emit({
                    'address': self.device.address,
                    'status': True
                })
                await self.client.start_notify(self.notify_uuid, self._notification_handler)
                self.freq("10")
                asyncio.create_task(self._battery_monitor())
                await self.stop_event.wait()

    def _notification_handler(self, sender, data: bytearray):
        # print(self.device.address, time.time())
        if data[1] == 0x61:
            self._decode_data(data)
        else:
            if data[2] == 0x64:
                self._decode_battery(data)
            else:
                print(data)
                
    async def _change_freq(self, freq):
        await self.client.write_gatt_char(
                self.write_uuid,
                self.freq_dict[freq])
                
    async def _battery_monitor(self):
        while not self.stop_event.is_set():
            self.battery()
            await asyncio.sleep(5)

    async def _get_battery_percent(self):
        try:
            self.client.address
            # 如果没有有效的client的话，函数不会执行
        except:
            return

        #对传感器发出的请求不一定会收到回应，因此要多次尝试
        self._battery_response = False
        while not self._battery_response:
            if self.stop_event.is_set():
                return
            await self.client.write_gatt_char(
                self.write_uuid,
                self.voltage_req)
            await asyncio.sleep(0.5)

    def _decode_data(self, data:bytearray):
        header_bit = data[0]
        assert header_bit == 0x55
        flag_bit = data[1] # 0x51 or 0x71
        assert flag_bit == 0x61 or flag_bit == 0x71
        # 2 byteずつ取り出して、後ろのbyteが上位ビットになるようにsigned shortに変換
        decoded = [int.from_bytes(data[i:i+2], byteorder='little', signed=True) for i in range(2, len(data), 2)]
        intstr = [int(i) for i in data]
        # print(bytearray(intstr))
        # signed shortの桁数なので、-32768~32767の範囲になる
        # なので一旦正規化してから各単位に合わせる
        ax    = decoded[0] / 32768.0 * 16 * 9.8
        ay    = decoded[1] / 32768.0 * 16 * 9.8
        az    = decoded[2] / 32768.0 * 16 * 9.8
        wx    = decoded[3] / 32768.0 * 2000
        wy    = decoded[4] / 32768.0 * 2000
        wz    = decoded[5] / 32768.0 * 2000
        roll  = decoded[6] / 32768.0 * 180
        pitch = decoded[7] / 32768.0 * 180
        yaw   = decoded[8] / 32768.0 * 180
        
        self.sigMotion.emit({
            'address': self.device.address,
            'timestamp': time.time(),
            'ax': ax,
            'ay': ay,
            'az': az,
            'wx': wx,
            'wy': wy,
            'wz': wz,
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw
        })
        # self.sigDebugMsg.emit(f"ax: {ax:.3f}, ay: {ay:.3f}, az: {az:.3f}, wx: {wx:.3f}, wy: {wy:.3f}, wz: {wz:.3f}, roll: {roll:.3f}, pitch: {pitch:.3f}, yaw: {yaw:.3f}")
        # self.current_data = (roll * np.pi / 180, pitch * np.pi / 180, yaw * np.pi / 180) # roll, pitch, yawをラジアンに変換して格納

    def _decode_battery(self, data):
        data = [str(hex(i)).split('x')[1] for i in data[4:6]]
        voltage = int(data[1]+data[0], base=16)
        self._battery_response = True
        self.sigBattery.emit({
            'address': self.device.address,
            'voltage': voltage
        })

    def _disconnected_callback(self, client):
        self.stop()