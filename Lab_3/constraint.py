from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
class Point:

    def __init__(self, **kwargs):
        for coord, value in kwargs.items():
            self.__setattr__(coord, value)
    
    def __add__(self, other):
        if len(vars(self).keys())!=len(vars(other).keys()):
            raise ArithmeticError("Addition not applicable!")
        coord_names = self.get_coord_names()
        self_coord, other_coord = self.get_np_array(), other.get_np_array()
        sub = self_coord + other_coord
        return Point(**dict(zip(coord_names, sub)))

    def __sub__(self, other):
        if len(vars(self).keys())!=len(vars(other).keys()):
            raise ArithmeticError("Substraction not applicable!")
        coord_names = self.get_coord_names()
        self_coord, other_coord = self.get_np_array(), other.get_np_array()
        add = self_coord - other_coord
        return Point(**dict(zip(coord_names, add)))

    def __rmul__(self, other:int):
        coord_names = self.get_coord_names()
        self_coord = self.get_np_array()
        mul = other*self_coord
        return Point(**dict(zip(coord_names, mul)))

    def __mul__(self, other):
        self_coord, other_coord = self.get_np_array(), other.get_np_array()
        return self_coord@other_coord

    def create_from_np_array(self, arr:np.array):
        keys = [f"x_{k+1}" for k in range(len(arr))]
        return Point(**dict(zip(keys, arr)))

    def get_np_array(self):
        return np.array([float(elem) for elem in vars(self).values()])
    
    def get_coord_names(self):
        return np.array([elem for elem in vars(self).keys()])

    def write_to_file(self, file, m = "a"):
        with open(file, mode = m) as f:
            for name in self.get_coord_names():
                f.write(f"{name} = {round(vars(self).get(name), 5)} ")
            f.write("\n")
class Constraint(ABC):

    @abstractmethod
    def find_projection(self, x:Point):
        pass   

class Sphere(Constraint):

    def __init__(self, center:Point, r:float):
        self.center = center
        self.r = r
    
    def find_projection(self, x:Point):
        coord_x_val, coord_x_key = x.get_np_array(), x.get_coord_names()
        coord_center_val = self.center.get_np_array()
        
        if len(coord_x_val) > len(coord_center_val):
            coord_x_val = coord_x_val[:len(coord_center_val)]
        elif len(coord_x_val) < len(coord_center_val):
            coord_x_val = np.append(coord_x_val, np.zeros(len(coord_center_val) - len(coord_x_val)))
        
        if (np.linalg.norm(coord_center_val - coord_x_val)<self.r):
            return x

        dist = np.linalg.norm(coord_center_val - coord_x_val)
        if dist == 0:
            return self.center
        else:
            proj_coord = coord_center_val + (coord_x_val - coord_center_val)/dist*self.r
        return Point(**dict(zip(coord_x_key, proj_coord)))

class Parallelepiped(Constraint):

    def __init__(self, coord_start:np.array, coord_end:np.array):
        if len(coord_start) != len(coord_end):
            raise ValueError("Dimensions of the start and the end are different")
        
        if sum(coord_start>coord_end):
            raise ValueError("Start point coordinates are greater than end point coordinates")
        
        self.coord_start = coord_start
        self.coord_end = coord_end

    def find_projection(self, x:Point):
        coord_x_val, coord_x_key = x.get_np_array(), x.get_coord_names()
        try:
            coord_x_val = np.append(coord_x_val, np.zeros(len(self.coord_start) - len(coord_x_val)))
        except ValueError:
            coord_x_val = coord_x_val[:len(self.coord_start)]

        proj_coord = np.array([self.coord_start[i] if coord_x_val[i]<self.coord_start[i] else 
                                self.coord_end[i] if coord_x_val[i]>self.coord_end[i] 
                                else coord_x_val[i] 
                                for i in range(len(coord_x_val))])
        
        return Point(**dict(zip(coord_x_key, proj_coord)))

class Ortant(Constraint):

    def __init__(self, n:int):
        self.n = n

    def find_projection(self, x:Point):
        coord_x_val, coord_x_key = x.get_np_array(), x.get_coord_names()

        try:
            coord_x_val = np.append(coord_x_val, np.zeros(self.n - len(coord_x_val)))
        except ValueError:
            coord_x_val = coord_x_val[:self.n]

        proj_coord = np.array([max(0, coord_x_val[i]) for i in range(self.n)])

        return Point(**dict(zip(coord_x_key, proj_coord)))

class Hyperplane(Constraint):

    def __init__(self, p:Point, b: float):
        if not sum(p.get_np_array() != 0):
            raise ValueError("Vector should not be equal to zero")
        self.p = p
        self.b = b
    
    def find_projection(self, x:Point):
        coord_x_val, coord_x_key = x.get_np_array(), x.get_coord_names()
        coord_p = self.p.get_np_array()

        try:
            coord_x_val = np.append(coord_x_val, np.zeros(len(coord_p) - len(coord_x_val)))
        except ValueError:
            coord_x_val = coord_x_val[:len(coord_p)]

        proj_coord = coord_x_val + (self.b - coord_p@coord_x_val)*coord_p/np.linalg.norm(coord_p)**2

        return Point(**dict(zip(coord_x_key, proj_coord)))

class Halfspace (Hyperplane):

    def find_projection(self, x:Point):
        coord_x_val, coord_x_key = x.get_np_array(), x.get_coord_names()
        coord_p = self.p.get_np_array()

        try:
            coord_x_val = np.append(coord_x_val, np.zeros(len(coord_p) - len(coord_x_val)))
        except ValueError:
            coord_x_val = coord_x_val[:len(coord_p)]

        proj_coord = coord_x_val + max(0, (self.b - coord_p@coord_x_val))*coord_p/np.linalg.norm(coord_p)**2

        return Point(**dict(zip(coord_x_key, proj_coord)))