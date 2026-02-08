
import numpy as np
from typing import List, Tuple, Callable, Any, Optional
from aria.types import Grid

COLOR_BLACK = 0
COLOR_BLUE = 1
COLOR_RED = 2
COLOR_GREEN = 3
COLOR_YELLOW = 4
COLOR_GRAY = 5
COLOR_MAGENTA = 6
COLOR_ORANGE = 7
COLOR_CYAN = 8
COLOR_MAROON = 9

def shift(grid: Grid, dx: int, dy: int, fill: int=COLOR_BLACK) -> Grid:
    """Shift grid content by (dx, dy)."""
    h, w = (grid.H, grid.W)
    new_data = np.full((h, w), fill, dtype=np.int8)
    src_y_start = max(0, -dy)
    src_y_end = min(h, h - dy)
    src_x_start = max(0, -dx)
    src_x_end = min(w, w - dx)
    dst_y_start = max(0, dy)
    dst_y_end = min(h, h + dy)
    dst_x_start = max(0, dx)
    dst_x_end = min(w, w + dx)
    if src_y_end > src_y_start and src_x_end > src_x_start:
        new_data[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = grid.data[src_y_start:src_y_end, src_x_start:src_x_end]
    return Grid(new_data)

def rotate_cw(grid: Grid) -> Grid:
    """Rotate 90 degrees clockwise."""
    return Grid(np.rot90(grid.data, k=-1))

def rotate_ccw(grid: Grid) -> Grid:
    """Rotate 90 degrees counter-clockwise."""
    return Grid(np.rot90(grid.data, k=1))

def flip_x(grid: Grid) -> Grid:
    """Flip horizontally."""
    return Grid(np.fliplr(grid.data))

def flip_y(grid: Grid) -> Grid:
    """Flip vertically."""
    return Grid(np.flipud(grid.data))

def color_replace(grid: Grid, old_c: int, new_c: int) -> Grid:
    """Replace all instances of old_c with new_c."""
    new_data = grid.data.copy()
    new_data[new_data == old_c] = new_c
    return Grid(new_data)

def isolate_color(grid: Grid, color: int) -> Grid:
    """Keep only pixels of 'color', make rest background(0)."""
    new_data = np.zeros_like(grid.data)
    mask = grid.data == color
    new_data[mask] = color
    return Grid(new_data)

def flood_fill(grid: Grid, start_x: int, start_y: int, new_c: int) -> Grid:
    """Standard BFS-based flood fill."""
    h, w = (grid.H, grid.W)
    if not (0 <= start_x < w and 0 <= start_y < h):
        return grid
    target_c = grid.data[start_y, start_x]
    if target_c == new_c:
        return grid
    new_data = grid.data.copy()
    queue = [(start_x, start_y)]
    visited = set()
    while queue:
        x, y = queue.pop(0)
        if (x, y) in visited:
            continue
        visited.add((x, y))
        new_data[y, x] = new_c
        for nx, ny in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
            if 0 <= nx < w and 0 <= ny < h:
                if new_data[ny, nx] == target_c:
                    queue.append((nx, ny))
    return Grid(new_data)

def bitwise_or(grid1: Grid, grid2: Grid) -> Grid:
    """Combine two grids. Non-zero pixels from grid2 overwrite grid1."""
    new_data = grid1.data.copy()
    h1, w1 = new_data.shape
    h2, w2 = grid2.data.shape
    h_min, w_min = (min(h1, h2), min(w1, w2))
    mask = grid2.data[:h_min, :w_min] != 0
    new_data[:h_min, :w_min][mask] = grid2.data[:h_min, :w_min][mask]
    return Grid(new_data)

def if_color(grid: Grid, color: int, then_f: Callable[[Grid], Grid], else_f: Callable[[Grid], Grid]) -> Grid:
    """Conditional masking."""
    then_grid = then_f(grid)
    else_grid = else_f(grid)
    mask = grid.data == color
    new_data = np.where(mask, then_grid.data, else_grid.data)
    return Grid(new_data)

def detect_objects(grid: Grid, bg_color: int=0) -> List[Grid]:
    return label_grid(flip_x(rotate_cw(grid)))

def label_grid(grid: Grid) -> List[Grid]:
    """High-performance connected component labeling."""
    from scipy.ndimage import label
    labeled, num_features = label(grid.data != 0)
    objs = []
    for i in range(1, num_features + 1):
        mask = labeled == i
        obj_data = np.zeros_like(grid.data)
        obj_data[mask] = grid.data[mask]
        objs.append(Grid(obj_data))
    return objs

def map_grid(grids: List[Grid], func: Callable[[Grid], Grid]) -> List[Grid]:
    return [func(g) for g in grids]

def fold_grid(grids: List[Grid], func: Callable[[Grid, Grid], Grid], initial: Grid) -> Grid:
    res = initial
    for g in grids:
        res = func(res, g)
    return res

def hconcat(left: Grid, right: Grid) -> Grid:
    """Horizontally concatenate with padding."""
    h_max = max(left.H, right.H)
    d1 = np.vstack([left.data, np.zeros((h_max - left.H, left.W), dtype=left.data.dtype)]) if left.H < h_max else left.data
    d2 = np.vstack([right.data, np.zeros((h_max - right.H, right.W), dtype=right.data.dtype)]) if right.H < h_max else right.data
    return Grid(np.hstack([d1, d2]))

def vconcat(top: Grid, bottom: Grid) -> Grid:
    """Vertically concatenate with padding."""
    w_max = max(top.W, bottom.W)
    d1 = np.hstack([top.data, np.zeros((top.H, w_max - top.W), dtype=top.data.dtype)]) if top.W < w_max else top.data
    d2 = np.hstack([bottom.data, np.zeros((bottom.H, w_max - bottom.W), dtype=bottom.data.dtype)]) if bottom.W < w_max else bottom.data
    return Grid(np.vstack([d1, d2]))

def filter_list(data: List[int], func: Callable[[int], bool]) -> List[int]:
    return [x for x in data if func(x)]

def is_even(x: int) -> bool:
    return x % 2 == 0

def is_positive(x: int) -> bool:
    return x > 0

def logical_and(a: bool, b: bool) -> bool:
    return a and b

def logical_or(a: bool, b: bool) -> bool:
    return a or b

# --- Meta-Evolution Primitives ---
def add(a: float, b: float) -> float: return a + b
def sub(a: float, b: float) -> float: return a - b
def mul(a: float, b: float) -> float: return a * b
def div(a: float, b: float) -> float: return a / b if b != 0 else 0.0