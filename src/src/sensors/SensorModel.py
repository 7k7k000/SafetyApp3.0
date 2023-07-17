from PySide6.QtCore import Signal, Slot, QObject
import numpy as np
class SensorModel(QObject):
    sigDataOutput = Signal(dict)
    ready = False
    last_data = None
    def __init__(self) -> None:
        super().__init__()
        pass
    
    @Slot()
    def calibrate(self):
        pass
    
    @Slot()
    def on_data_receive(self, res):
        pass
    
    def deg_of_two_vec(self, v1:np.array, v2:np.array) -> float:
        '''
        计算两个向量之间的欧拉角
        '''
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        cosine_angle = dot_product / (norm1 * norm2)
        angle_in_radians = np.arccos(cosine_angle)
        angle_in_degrees = np.degrees(angle_in_radians)
        return angle_in_degrees
    

class PedalModel(SensorModel):
    def __init__(self) -> None:
        super().__init__()
        self.pos0 = None
        self.pos1 = None
        self.stroke:float = 0
        self.last_data = None
        
    @Slot()
    def calibrate(self, pos):
        if self.last_data is not None:
            # print(f"vec0: {self.pos0}  vec1: {self.pos1}, cross: {self.cross}, stroke: {self.stroke}")
            res = self.last_data
            v = np.array([res['ax'], res['ay'], res['az']])
            print(v)
            if pos == 0:
                self.pos0 = v
            elif pos == 1:
                self.pos1 = v
            if self.pos0 is not None and self.pos1 is not None:
                self.cross = np.cross(self.pos0, self.pos1)
                self.cross = self.cross / np.linalg.norm(self.cross)
                self.stroke = self.deg_of_two_vec(self.pos0, self.pos1)
                
                self.ready = True
                
    count = 0
    @Slot()
    def on_data_receive(self, res):
        self.last_data = res
        if self.ready:
            v0 = np.array([res['ax'], res['ay'], res['az']])
            v1 = v0 - np.dot(v0, self.cross)
            deg1 = self.deg_of_two_vec(v1, self.pos0)
            deg2 = self.deg_of_two_vec(v1, self.pos1)
            if deg1 >= self.stroke:
                deg = self.stroke
            elif deg2 >= self.stroke:
                deg = 0
            else:
                deg = deg1
            res['pedal_deg'] = deg
            res['peal_percent'] = deg/self.stroke
            self.sigDataOutput.emit(res)
        pass
        
        
        
class SteeringWheelModel(SensorModel):
    def __init__(self) -> None:
        super().__init__()
        self.calibrated_rot = 0
        self.prev_deg = 0
        
    @Slot()
    def calibrate(self):
        if self.last_data is not None:
            self.prev_deg = 0
            self.calibrated_rot = self.raw_deg
            
    @Slot()
    def on_data_receive(self, res):
        # self.get_wheel_degs1(res)
        res['steeringwheel_deg'] = self.get_wheel_degs(res)
        self.last_data = res
        self.sigDataOutput.emit(self.last_data)
        
    def get_wheel_degs1(self, res):
        #Accelorometor Approach
        cross = np.array([0,0,1])
        v0 = np.array([1,0,0])
        v = np.array([res['ax'], res['ay'], res['az']])
        v1 = v - np.dot(v, cross)
        deg = self.deg_of_two_vec(v0, v1) * v1[1] / np.abs(v1[1])
        print(deg)
        
    def get_wheel_degs(self, data):
        y = np.sin(np.radians((float(data['pitch']))))
        x = np.sin(np.radians((float(data['roll']))))
        vec = np.array([y, x])
        deg = np.rad2deg(np.arccos(np.dot(vec, np.array([0, 1]))/np.linalg.norm(vec)))
        if y < 0:
            deg = -deg
        self.raw_deg = deg
        if self.calibrated_rot != 0:
            if deg < 0:
                deg+=360
            deg -= self.calibrated_rot
            if deg > 180:
                deg -= 360
        if self.prev_deg > 0:
            deg = self.steering_rotation(deg, self.prev_deg)
        else:
            deg = - self.steering_rotation(-deg, -self.prev_deg)
        self.prev_deg = deg
        # print(round(data['roll']), round(data['pitch']), round(data['yaw']), round(deg))
        return deg
    
    def steering_rotation(self, deg, prev_deg):
        '''
        实现方向盘打圈计数累加
        '''
        if deg >= 0:
            if deg < 50 and (360-(prev_deg%360) < 50):
                deg += (prev_deg//360+1) * 360
            else:
                deg += (prev_deg - prev_deg%360)
        if deg < 0:
            if prev_deg > 130:
                deg = (360+deg) + (prev_deg - prev_deg%360)
                if prev_deg % 360 < 50:
                    deg -= 360
        return deg
        
class DataModel(QObject):
    def __init__(self) -> None:
        super().__init__()
        self.wheel = SteeringWheelModel()
        self.brake = PedalModel()
        self.acc = PedalModel()
        self.type_dict = {}
        
    @Slot()
    def on_sensor_type_change(self, e):
        mac, t = e
        if t == "方向盘":
            self.type_dict[mac] = self.wheel
        elif t == "制动踏板":
            self.type_dict[mac] = self.brake
        elif t == "加速踏板":
            self.type_dict[mac] = self.acc
        elif t == "无":
            self.type_dict.pop(mac)
        
    @Slot()
    def on_data_receive(self, e):
        if e['address'] in self.type_dict:
            self.type_dict[e['address']].on_data_receive(e)
        