from typing import List, Optional, Union, Tuple
import numpy as np

class Rect:
    def __init__(self, *args, min_x: float = None, min_y: float = None, max_x: float = None, max_y: float = None):
        """
        Initialize Rect.
        
        Supports multiple initialization patterns:
        1. Rect(min_x, min_y, max_x, max_y) - 2D backward compatibility
        2. Rect(min_coords, max_coords) - N-dim (lists or arrays)
        3. Rect(min_x=..., min_y=..., ...) - Keyword arguments
        """
        if len(args) == 4:
            # 2D case: min_x, min_y, max_x, max_y
            self.min = np.array([args[0], args[1]], dtype=np.float32)
            self.max = np.array([args[2], args[3]], dtype=np.float32)
        elif len(args) == 2:
            # N-dim case: min_array, max_array
            self.min = np.array(args[0], dtype=np.float32)
            self.max = np.array(args[1], dtype=np.float32)
            if self.min.shape != self.max.shape:
                raise ValueError("Min and max coordinate arrays must have same shape")
        elif min_x is not None and min_y is not None and max_x is not None and max_y is not None:
             # KWArgs 2D
            self.min = np.array([min_x, min_y], dtype=np.float32)
            self.max = np.array([max_x, max_y], dtype=np.float32)
        else:
            raise ValueError("Invalid initialization arguments for Rect")
            
        self.dims = len(self.min)

    @property
    def min_x(self):
        return self.min[0]

    @property
    def min_y(self):
        return self.min[1]

    @property
    def max_x(self):
        return self.max[0]

    @property
    def max_y(self):
        return self.max[1]

    def __eq__(self, other):
        if isinstance(other, Rect):
            if self.dims != other.dims:
                return False
            return np.allclose(self.min, other.min) and np.allclose(self.max, other.max)
        return False

    def __repr__(self):
        if self.dims == 2:
            return f'Rect({self.min[0]}, {self.min[1]}, {self.max[0]}, {self.max[1]})'
        return f'Rect(d={self.dims}, min={self.min}, max={self.max})'

    def union(self, rect: 'Rect') -> 'Rect':
        if self.dims != rect.dims:
            raise ValueError(f"Dimension mismatch in union: {self.dims} vs {rect.dims}")
        
        new_min = np.minimum(self.min, rect.min)
        new_max = np.maximum(self.max, rect.max)
        return Rect(new_min, new_max)

    def intersection(self, rect: 'Rect') -> Optional['Rect']:
        if self.dims != rect.dims:
            raise ValueError(f"Dimension mismatch in intersection: {self.dims} vs {rect.dims}")
            
        inter_min = np.maximum(self.min, rect.min)
        inter_max = np.minimum(self.max, rect.max)
        
        if np.all(inter_min < inter_max):
            return Rect(inter_min, inter_max)
        return None

    def intersects(self, rect: 'Rect') -> bool:
        if self.dims != rect.dims:
            raise ValueError(f"Dimension mismatch in intersects: {self.dims} vs {rect.dims}")
            
        return np.all(self.max > rect.min) and np.all(self.min < rect.max)

    def get_intersection_area(self, rect: 'Rect') -> float:
        if self.dims != rect.dims:
            raise ValueError(f"Dimension mismatch in area: {self.dims} vs {rect.dims}")
            
        overlaps = np.maximum(0.0, np.minimum(self.max, rect.max) - np.maximum(self.min, rect.min))
        return np.prod(overlaps)

    @property
    def width(self):
        return self.max[0] - self.min[0]

    @property
    def height(self):
        return self.max[1] - self.min[1]

    def perimeter(self) -> float:
        # Actually surface area / perimeter sum of edge lengths?
        # Standard R-tree metric is margin (sum of edge lengths)
        return np.sum(self.max - self.min) * 2

    def area(self) -> float:
        return np.prod(self.max - self.min)

    def centroid(self) -> Union[Tuple[float, float], np.ndarray]:
        c = (self.min + self.max) / 2
        if self.dims == 2:
            return c[0], c[1]
        return c

    def min_dist_sq(self, point) -> float:
        """
        Calculates the squared minimum Euclidean distance from a point to this rectangle.
        Point can be an object with .x, .y or a list/array.
        """
        # Handle legacy point object if likely 2D
        if hasattr(point, 'x') and hasattr(point, 'y'):
            p_coords = np.array([point.x, point.y], dtype=np.float32)
        else:
            p_coords = np.array(point, dtype=np.float32)

        if len(p_coords) != self.dims:
             # Minimal support for mismatch if 2D check
             if self.dims == 2 and len(p_coords) >= 2:
                  p_coords = p_coords[:2]
             else:
                  # This might happen if point is higher dim?
                  pass

        dists = np.maximum(0, np.maximum(self.min - p_coords, p_coords - self.max))
        return np.sum(dists**2)


def union(rect1: Rect, rect2: Rect) -> Rect:
    if rect1 is None:
        return rect2
    if rect2 is None:
        return rect1
    return rect1.union(rect2)


def union_all(rects: List[Rect]) -> Rect:
    result = None
    for rect in rects:
        result = union(result, rect)
    return result

