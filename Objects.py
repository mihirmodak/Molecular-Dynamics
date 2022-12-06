import math
import numpy as np

class Vector:
    """A three-dimensional vector with Cartesian coordinates."""

    def __init__(self, x=0, y=0, z=0):

        self.x, self.y, self.z = x, y, z
        self.dim = [2 if self.z==0 else 3][0]

    def __str__(self):
        """Human-readable string representation of the vector."""
        return f'{self.x}i + {self.y}j + {self.z}k'

    def __repr__(self):
        """Unambiguous string representation of the vector."""
        return repr((self.x, self.y, self.z))

    def set_dim(self):
        return [2 if self.z == 0 else 3][0]

    def dot(self, other):
        """The scalar (dot) product of self and other. Both must be vectors."""

        if not isinstance(other, Vector):
            raise TypeError('Can only take dot product of two Vector objects')
        return self.x * other.x + self.y * other.y + self.z * other.z
    # Alias the __matmul__ method to dot so we can use a @ b as well as a.dot(b).
    __matmul__ = dot

    def __add__(self, other):
        """Vector addition."""
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        """Vector subtraction."""
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        """Multiplication of a vector by a scalar."""

        if isinstance(scalar, int) or isinstance(scalar, float):
            return Vector(self.x*scalar, self.y*scalar, self.z*scalar)
        raise TypeError('Can only multiply Vector by a scalar')

    def __rmul__(self, scalar):
        """Reflected multiplication so vector * scalar also works."""
        return self.__mul__(scalar)

    def __neg__(self):
        """Negation of the vector (invert through origin.)"""
        return Vector(-self.x, -self.y, -self.z)

    def __truediv__(self, scalar):
        """True division of the vector by a scalar."""
        return Vector(self.x / scalar, self.y / scalar, self.z / scalar)

    def __mod__(self, scalar):
        """One way to implement modulus operation: for each component."""
        return Vector(self.x % scalar, self.y % scalar, self.z % scalar)

    def __abs__(self):
        """Absolute value (magnitude) of the vector."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def distance_to(self, other):
        """The distance between vectors self and other."""
        return abs(self - other)

    def to_polar(self):
        """Return the vector's components in polar coordinates (r, theta)"""

        if self.z == 0:

            return self.__abs__(), math.degrees(math.atan2(self.y, self.x))
        else:
            raise TypeError("Can only covert 2D vectors to Polar form. Use to_cylindrical or to_spherical for 3D vectors.")

    def to_cylindrical(self):
        """Return vector's components in cylindrical coordinate form (rho, theta, z)"""
        if self.z != 0:
            rho, theta = Vector(self.x, self.y).to_polar()

            return rho, theta, self.z
        else:
            raise TypeError("Can only covert 3D vectors to Cylindrical form. Use to_polar for 2D vectors.")
    
    def to_spherical(self):
        """Return a sector's components in spherical coordinate form (rho, theta, phi)"""

        if self.z != 0:
            rho = self.__abs__()
            theta = math.atan2(self.y, self.x)

            r = Vector(self.x,self.y).__abs__()
            phi = math.atan2(r,self.z)

            return rho, math.degrees(theta), math.degrees(phi)
        else:
            raise TypeError("Can only covert 3D vectors to Cylindrical form. Use to_polar for 2D vectors.")

class Matrix:

    def __init__(self, arr=None, dim: list = None):

        if arr is not None:
            self.array = arr
        else:
            self.array = np.zeros(tuple(dim), dtype=int)

        if dim is not None:
            self.shape = dim
        else:
            self.shape = np.array(self.array).shape
    
    def __str__(self):
        """Human-readable string representation of the matrix."""

        temp = str(self.array)
        temp = temp.replace("\n", "\n       ")

        return f"Matrix({temp})"

    # def __repr__(self):
    #     return repr(self.array)
    __repr__ = __str__

    def __len__(self):
        return len(self.to_list())

    def __getitem__(self, x):
        return self.to_list()[x]

    def to_numpy(self):
        return np.array(self.array)

    def to_list(self):
        return self.array

    def __add__(self, other):
        """Matrix addition."""
        assert isinstance(other, Matrix)
        if self.shape == other.shape:
            return Matrix( array= self.to_numpy() + other.to_numpy() )

    def __sub__(self, other):
        assert isinstance(other, Matrix)
        """Vector subtraction."""
        return Matrix( array= self.to_numpy() - other.to_numpy() )

    def __mul__(self, other):
        """Multiplication of a vector by a scalar."""
        if len(self.shape) == 1:
            self.array = self.to_numpy()[np.newaxis]
        if len(other.shape) == 1:
            other.array = other.to_numpy()[np.newaxis]
        if isinstance(other, int) or isinstance(other, float):
            return Matrix( self.to_list() * other )
        else:
            return Matrix( np.matmul(self, other) )

    def __rmul__(self, other):
        """Reflected multiplication so vector * scalar also works."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """True division of the vector by a scalar."""
        if isinstance(other, int) or isinstance(other, float):
            return Matrix( self.to_list() / other )
        else:
            raise ArithmeticError("Cannot divide Matrices")

    def __mod__(self, other):
        """One way to implement modulus operation: for each component."""
        if isinstance(other, int) or isinstance(other, float):
            return Matrix( self.to_list() % other )
        else:
            raise ArithmeticError("Cannot divide Matrices")
    
    def __round__(self, decimals=0):
        if decimals == 0:
            return Matrix( self.to_numpy().round(decimals=decimals).astype('int') )
        else:
            return Matrix( self.to_numpy().round(decimals=decimals) )

    def determinant(self):
        return np.linalg.det(self)
    det = determinant # Allow for the use of m.determinant() and m.det()

    def transpose(self):
        if len(self.shape) == 1:
            temp = self.to_numpy()[np.newaxis]
            return Matrix(np.matrix.transpose(temp))
        else:
            return Matrix(np.matrix.transpose(self.to_numpy()))
    T = transpose # Allow for m.T() as well as m.transpose()