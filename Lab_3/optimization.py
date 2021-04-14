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

    def solution(self,x_0:Point, method:str= "const", alpha:float = 1, eps:float = 0.001):
        if method not in ["const", "divide"]:
            raise ValueError("This method is not applicable")
        h = self.find_grad(0.01, x_0)
        x_1 = x_0 - alpha*h
        x = self.constr.find_projection(x_1)
        diff = x - x_0
        dist = np.linalg.norm(diff.get_np_array())

        e = 0.25
        k,m = 1, 1
#abs(dist)>=eps and 
        #print(dist)
        while dist>=eps and m<=50:
            #print(vars(x), vars(x_0))
            x_0 = x
            h = self.find_grad(0.01, x_0)
            x = self.constr.find_projection(x_0 - alpha*h)
            #print(vars(x), vars(x_0))
            if method == "divide":
                norm_h = np.linalg.norm(h.get_np_array())
                #print(diff_func)
                #print(-alpha*e*np.linalg.norm(h.get_np_array())**2)
                while self.func(*x.get_np_array()) > self.func(*x_0.get_np_array())-alpha*e*(norm_h**2):
                    alpha/=2
                    x = self.constr.find_projection(x_0 - alpha*h)
            print(alpha)
            diff = x - x_0
            dist = np.linalg.norm(diff.get_np_array())
            m+=1
        return vars(x), np.linalg.norm(x.get_np_array()),self.func(*x_0.get_np_array()), m
        #self.func(*vars(self.constr.find_projection(x)).values()), sum(self.constr.find_projection(x).get_np_array()**2)
#print(vars(Point(x = 1, y = 2) - 5*Point(x = 1, y = 2)))
print(OptimizationProblem(lambda x, y, z: x+y+z, Sphere(Point(x = 0, y = 0, z = 0), r = 1)).solution(Point(x = 2.5, y = -0.5, z = 0), method = "divide"))