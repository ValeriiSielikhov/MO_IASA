from constraint import *
import numpy as np

class OptimizationProblem(object):

    def __init__(self, func, constraint: Constraint):
        self.func = func
        self.constr = constraint
    
    def first_order_der(self, arg:str, h:float, **val:float):
        first_arr, second_arr = list(val.values()), list(val.values())
        for i, key in enumerate(val.keys()):
            if key == arg:
                first_arr[i]+=h
                second_arr[i]-=h
        return (self.func(*first_arr) - self.func(*second_arr))/(2*h)

    def find_grad(self, h:float, x:Point):
        list_of_args = x.get_coord_names()
        grads = np.array([self.first_order_der(arg, h, **dict(zip(list_of_args, x.get_np_array()))) for arg in list_of_args])
        return Point(**dict(zip(list_of_args, grads)))


    def solution(self,x_0:Point, method:str= "const", alpha:float = 1, eps:float = 0.001, file:str = "cache.txt"):
        if method not in ["const", "divide"]:
            raise ValueError("This method is not applicable")
        a = alpha
        h = self.find_grad(0.01, x_0)
        x = self.constr.find_projection(x_0 - alpha*h)
        if method == "divide":
            while self.func(*x.get_np_array()) > self.func(*x_0.get_np_array()):
                a*=0.5
                x = self.constr.find_projection(x_0 - a*h)
        diff = x - x_0
        dist = np.linalg.norm(diff.get_np_array())
        m = 1
        
        with open(file, "a") as f:
                f.write(f"Iteration:{m}\n")
                f.write(f"Alpha:{a}\n")
        
        x.write_to_file(file)

        
        while dist>=eps:
            a = alpha
            x_0 = x
            h = self.find_grad(0.01, x_0)
            x = self.constr.find_projection(x_0 - a*h)
            if method == "divide":
                
                while self.func(*x.get_np_array()) > self.func(*x_0.get_np_array()):
                    a*=0.5
                    x = self.constr.find_projection(x_0 - a*h)
            
            diff = x - x_0
            dist = np.linalg.norm(diff.get_np_array())
            m+=1
            with open(file, "a") as f:
                f.write(f"Iteration:{m}\n")
                f.write(f"Alpha:{a}\n")
            x.write_to_file(file)
        return x
