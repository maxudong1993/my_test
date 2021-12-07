import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __call__(self):
        print('name-{}'.format(self.name))
        print('age-{}'.format(self.age))

    def __str__(self):
        return 'name-{}; age-{}'.format(self.name, self.age)

    def __repr__(self):
        return 'Student: name-{} age-{}'.format(self.name, self.age)
    
    def my_student():
        return "*****"


@torch.jit.script
def foo(x, y):
    if x.max() > y.max():
        r = x
    else:
        r = y
    return r

def plt_save_gray(image_path):
    image = Image.open(image_path)
    plt.imsave('../pltsave.png',np.array(image),cmap = 'gray') #useless


def main():
    student1 = Student('eric', 19)
    # student1()
    # print(student1.__str__())
    # print(student1.__repr__())
    print(student1)
    print(dir(student1))
    # print(repr(student1))

if __name__ == '__main__':
    plt_save_gray('../dog1.jpeg')
    # main()
    # print(type(foo))  # torch.jit.ScriptFuncion
    # # See the compiled graph as Python code
    # print(foo.code)