from typing import Union, Tuple, List
import numpy as np
from functools import partial
from .rect import Rect
from .point import Point


Location = Union[
    Point,
    Rect,
    Tuple[float, float],
    Tuple[float, float, float, float],
    List[float]
]


def get_loc_intersection_fn(loc: Location):
    if isinstance(loc, Point):
        return partial(point_intersects_rect, loc)
    if isinstance(loc, Rect):
        return partial(rect_intersects_rect, loc)
    if isinstance(loc, (list, tuple)):
        # If it's 2 coords, it could be a 2D point.
        # If it's 4 coords, it could be a 2D rect [minx, miny, maxx, maxy].
        # However, for N-dim, we should probably treat it as a point if it doesn't match a rect pattern.
        # But for backward compatibility we keep the 2/4 checks.
        if len(loc) == 2:
            point = Point(loc[0], loc[1])
            return partial(point_intersects_rect, point)
        if len(loc) == 4:
            # Check if it's likely a 2D Rect
            rect = Rect(loc[0], loc[1], loc[2], loc[3])
            return partial(rect_intersects_rect, rect)
            
        # N-dim Point fallback
        return partial(point_intersects_rect, loc)
    raise TypeError(f"Invalid location type: {type(loc)}. Location must either be a Point, Rect, list or tuple.")


def point_intersects_rect(point: Union[Point, List[float], np.ndarray], rect: Rect):
    import numpy as np
    if isinstance(point, Point):
        p_coords = np.array([point.x, point.y], dtype=np.float32)
    else:
        p_coords = np.array(point, dtype=np.float32)
        
    if len(p_coords) != rect.dims:
        raise ValueError(f"Point dimension mismatch: {len(p_coords)} vs {rect.dims}")
        
    return np.all(rect.min <= p_coords) and np.all(p_coords <= rect.max)


def rect_intersects_rect(rect1: Rect, rect2: Rect):
    return rect1.intersects(rect2)
