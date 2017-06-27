import numpy as np
import math
import cv2

round_digits = 4


class Vector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        self.x += other.x
        self.y += other.y
        self.z += other.z

        self.round()

    def __sub__(self, other):
        self.x -= other.x
        self.y -= other.y
        self.z -= other.z

        self.round()

    def __div__(self, number):
        self.x /= number
        self.y /= number
        self.z /= number

        self.round()

    def __str__(self):
        return 'X = ' + self.x.__str__() + ', Y = ' + self.y.__str__() + ', Z = ' + self.z.__str__()

    def __unicode__(self):
        return self.x.__str__() + ',' + self.y.__str__() + ',' + self.z.__str__()

    @staticmethod
    def sub(v1, v2):
        x = v1.x - v2.x
        y = v1.y - v2.y
        z = v1.z - v2.z

        result = Vector(x, y, z)
        result.round()
        return result

    @staticmethod
    def add(v1, v2):
        x = v1.x + v2.x
        y = v1.y + v2.y
        z = v1.z + v2.z

        result = Vector(x=x, y=y, z=z)
        result.round()

        return result

    @staticmethod
    def mult(v1, number):
        x = v1.x * number
        y = v1.y * number
        z = v1.z * number
        result = Vector(x=x, y=y, z=z)
        result.round()

        return result

    def copy(self):
        return Vector(self.x, self.y, self.z)

    @staticmethod
    def distance(v1, v2):
        dx = (v1.x - v2.x) * (v1.x - v2.x)
        dy = (v1.y - v2.y) * (v1.y - v2.y)
        dz = (v1.z - v2.z) * (v1.z - v2.z)
        return math.sqrt(dx + dy + dz)

    @staticmethod
    def mid_point(v1, v2):
        x = int((v1.x + v2.x) / float(2))
        y = int((v1.y + v2.y) / float(2))
        z = int((v1.z + v2.z) / float(2))
        return Vector(x, y, z)

    def dot_product(self, other):
        x = self.x * other.x
        y = self.y * other.y
        z = self.z * other.z
        result = x + y + z
        return result

    def cross_product(self, other):
        x = self.y * other.z - self.z * other.y
        y = self.z * other.x - self.x * other.z
        z = self.x * other.y - self.y * other.x

        result = Vector(x=x, y=y, z=z)
        result.round()

        return result

    def length(self):
        length = math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
        return length

    def get_angle(self, other):
        if self.length() == 0 or other.length() == 0:
            return 0
        product = self.dot_product(other)
        cosine = float(product) / (self.length() * other.length())
        if cosine > 1:
            cosine = 1.0
        elif cosine < -1:
            cosine = -1.0
        angle = math.acos(cosine)
        angle = math.degrees(angle)
        return angle

    @staticmethod
    def get_angle2(v1, v2):
        if v1.length() == 0 or v2.length() == 0:
            return 0
        product = v1.dot_product(v2)
        cosine = product / (v1.length() * v2.length())
        if cosine > 1:
            cosine = 1.0
        elif cosine < -1:
            cosine = -1.0
        angle = math.acos(cosine)
        angle = math.degrees(angle)
        return angle

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def round(self):
        self.x = round(self.x, round_digits)
        self.y = round(self.y, round_digits)
        self.z = round(self.z, round_digits)

    def set(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def print_self(self):
        return self.__unicode__()

    def get_magnitude(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def get_normalization(self):
        magnitude = self.get_magnitude()
        return Vector(x=float(self.x) / magnitude, y=float(self.y) / magnitude, z=float(self.z) / magnitude)


class Region:
    def __init__(self, w, h, start_point, name=''):
        self.width = int(w)
        self.height = int(h)
        self.name=name
        self.top_left = start_point.copy()

        self.bottom_right = Vector(self.top_left.x + self.width, self.top_left.y + self.height, 0)
        self.center = None
        self.calculate_center()

    def reposition_around_center(self, center):
        x = 0
        y = 0

        if center.x > self.width / 2:
            x = center.x - self.width / 2

        if center.y > self.height / 2:
            y = center.y - self.height / 2

        self.top_left = Vector(x, y)
        self.set_end_point()
        self.calculate_center()

    def calculate_center(self):
        x = int(self.top_left.x + self.width / 2)
        y = int(self.top_left.y + self.height / 2)
        self.center = Vector(x, y, 0)

    def copy(self):
        return Region(self.width, self.height, self.top_left)

    def draw(self, image, color):
        pt1 = (int(self.top_left.x), int(self.top_left.y))
        pt2 = (int(self.bottom_right.x), int(self.bottom_right.y))
        cv2.rectangle(image, pt1, pt2, color, 2)
