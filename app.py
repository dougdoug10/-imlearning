"""
Shape Equal Area Splitter
A program that splits any shape into N equal area parts
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Rectangle, Circle
from typing import List, Tuple, Union
import math


class Shape:
    """Base class for shapes"""
    def get_area(self) -> float:
        raise NotImplementedError

    def split(self, n: int) -> List['Shape']:
        raise NotImplementedError

    def plot(self, ax, color='blue', alpha=0.3, edgecolor='black'):
        raise NotImplementedError


class Polygon(Shape):
    """Polygon defined by vertices"""
    def __init__(self, vertices: List[Tuple[float, float]]):
        self.vertices = np.array(vertices)

    def get_area(self) -> float:
        """Calculate polygon area using shoelace formula"""
        x = self.vertices[:, 0]
        y = self.vertices[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def get_centroid(self) -> Tuple[float, float]:
        """Calculate polygon centroid"""
        x = self.vertices[:, 0]
        y = self.vertices[:, 1]
        area = self.get_area()
        cx = np.sum((x + np.roll(x, 1)) * (x * np.roll(y, 1) - np.roll(x, 1) * y)) / (6 * area)
        cy = np.sum((y + np.roll(y, 1)) * (x * np.roll(y, 1) - np.roll(x, 1) * y)) / (6 * area)
        return cx, cy

    def split_horizontal(self, n: int) -> List['Polygon']:
        """Split polygon into n horizontal strips of equal area"""
        min_y = np.min(self.vertices[:, 1])
        max_y = np.max(self.vertices[:, 1])
        total_area = self.get_area()
        target_area = total_area / n

        strips = []
        current_y = min_y

        for i in range(n):
            if i == n - 1:
                next_y = max_y
            else:
                next_y = self._find_split_y(current_y, max_y, (i + 1) * target_area, min_y)

            strip_vertices = self._get_strip_vertices(current_y, next_y)
            if len(strip_vertices) >= 3:
                strips.append(Polygon(strip_vertices))

            current_y = next_y

        return strips

    def _find_split_y(self, y_min: float, y_max: float, target_area: float, absolute_min_y: float) -> float:
        """Binary search to find y coordinate that gives target cumulative area"""
        tolerance = 1e-6
        max_iterations = 50

        for _ in range(max_iterations):
            y_mid = (y_min + y_max) / 2
            strip_vertices = self._get_strip_vertices(absolute_min_y, y_mid)

            if len(strip_vertices) >= 3:
                current_area = Polygon(strip_vertices).get_area()

                if abs(current_area - target_area) < tolerance:
                    return y_mid

                if current_area < target_area:
                    y_min = y_mid
                else:
                    y_max = y_mid
            else:
                y_min = y_mid

        return (y_min + y_max) / 2

    def _get_strip_vertices(self, y_min: float, y_max: float) -> List[Tuple[float, float]]:
        """Get vertices of horizontal strip between y_min and y_max"""
        vertices = []

        for i in range(len(self.vertices)):
            v1 = self.vertices[i]
            v2 = self.vertices[(i + 1) % len(self.vertices)]

            if y_min <= v1[1] <= y_max:
                vertices.append(tuple(v1))

            if (v1[1] - y_min) * (v2[1] - y_min) < 0:
                t = (y_min - v1[1]) / (v2[1] - v1[1])
                x = v1[0] + t * (v2[0] - v1[0])
                vertices.append((x, y_min))

            if (v1[1] - y_max) * (v2[1] - y_max) < 0:
                t = (y_max - v1[1]) / (v2[1] - v1[1])
                x = v1[0] + t * (v2[0] - v1[0])
                vertices.append((x, y_max))

        if not vertices:
            return []

        vertices = self._sort_vertices(vertices)
        return vertices

    def _sort_vertices(self, vertices: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Sort vertices in counter-clockwise order"""
        if len(vertices) < 3:
            return vertices

        center_x = sum(v[0] for v in vertices) / len(vertices)
        center_y = sum(v[1] for v in vertices) / len(vertices)

        def angle_from_center(v):
            return math.atan2(v[1] - center_y, v[0] - center_x)

        return sorted(vertices, key=angle_from_center)

    def split(self, n: int) -> List['Polygon']:
        """Split polygon into n equal area parts"""
        return self.split_horizontal(n)

    def plot(self, ax, color='blue', alpha=0.3, edgecolor='black'):
        patch = MplPolygon(self.vertices, facecolor=color, alpha=alpha, edgecolor=edgecolor, linewidth=2)
        ax.add_patch(patch)


class RectangleShape(Shape):
    """Rectangle shape"""
    def __init__(self, x: float, y: float, width: float, height: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def get_area(self) -> float:
        return self.width * self.height

    def split(self, n: int) -> List['RectangleShape']:
        """Split rectangle into n equal area parts (horizontal strips)"""
        strip_height = self.height / n
        strips = []

        for i in range(n):
            strips.append(RectangleShape(
                self.x,
                self.y + i * strip_height,
                self.width,
                strip_height
            ))

        return strips

    def plot(self, ax, color='blue', alpha=0.3, edgecolor='black'):
        rect = Rectangle((self.x, self.y), self.width, self.height,
                        facecolor=color, alpha=alpha, edgecolor=edgecolor, linewidth=2)
        ax.add_patch(rect)


class CircleShape(Shape):
    """Circle shape"""
    def __init__(self, center_x: float, center_y: float, radius: float):
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius

    def get_area(self) -> float:
        return math.pi * self.radius ** 2

    def split(self, n: int) -> List['CircleShape']:
        """Split circle into n concentric rings of equal area"""
        rings = []

        for i in range(n):
            outer_radius = self.radius * math.sqrt((i + 1) / n)
            inner_radius = self.radius * math.sqrt(i / n) if i > 0 else 0

            rings.append(CircleRing(self.center_x, self.center_y, inner_radius, outer_radius))

        return rings

    def plot(self, ax, color='blue', alpha=0.3, edgecolor='black'):
        circle = Circle((self.center_x, self.center_y), self.radius,
                       facecolor=color, alpha=alpha, edgecolor=edgecolor, linewidth=2)
        ax.add_patch(circle)


class CircleRing(Shape):
    """Ring (annulus) between two circles"""
    def __init__(self, center_x: float, center_y: float, inner_radius: float, outer_radius: float):
        self.center_x = center_x
        self.center_y = center_y
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius

    def get_area(self) -> float:
        return math.pi * (self.outer_radius ** 2 - self.inner_radius ** 2)

    def plot(self, ax, color='blue', alpha=0.3, edgecolor='black'):
        outer_circle = Circle((self.center_x, self.center_y), self.outer_radius,
                             facecolor=color, alpha=alpha, edgecolor=edgecolor, linewidth=2)
        ax.add_patch(outer_circle)

        if self.inner_radius > 0:
            inner_circle = Circle((self.center_x, self.center_y), self.inner_radius,
                                 facecolor='white', edgecolor=edgecolor, linewidth=2)
            ax.add_patch(inner_circle)


def split_shape(shape: Shape, n: int) -> List[Shape]:
    """Split any shape into n equal area parts"""
    if n <= 0:
        raise ValueError("Number of splits must be positive")
    if n == 1:
        return [shape]

    return shape.split(n)


def visualize_split(original_shape: Shape, split_shapes: List[Shape], title: str = "Shape Split"):
    """Visualize the original shape and its equal area splits"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot original shape
    ax1.set_aspect('equal')
    ax1.set_title(f'Original Shape\nTotal Area: {original_shape.get_area():.2f}', fontsize=12, fontweight='bold')
    original_shape.plot(ax1, color='skyblue', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    # Plot split shapes
    ax2.set_aspect('equal')
    ax2.set_title(f'Split into {len(split_shapes)} Equal Area Parts\nArea per part: {original_shape.get_area() / len(split_shapes):.2f}',
                  fontsize=12, fontweight='bold')

    colors = plt.cm.rainbow(np.linspace(0, 1, len(split_shapes)))

    for i, (split_shape, color) in enumerate(zip(split_shapes, colors)):
        split_shape.plot(ax2, color=color, alpha=0.6)
        area = split_shape.get_area()

    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def demo():
    """Demonstrate the shape splitting functionality"""

    print("=" * 60)
    print("SHAPE EQUAL AREA SPLITTER")
    print("=" * 60)

    # Example 1: Rectangle
    print("\n1. Splitting a Rectangle into 4 equal area parts...")
    rect = RectangleShape(0, 0, 10, 8)
    rect_splits = split_shape(rect, 4)
    print(f"   Original area: {rect.get_area():.2f}")
    print(f"   Area per part: {rect.get_area() / 4:.2f}")
    for i, part in enumerate(rect_splits):
        print(f"   Part {i+1} area: {part.get_area():.2f}")
    visualize_split(rect, rect_splits, "Rectangle Split into 4 Equal Parts")

    # Example 2: Circle
    print("\n2. Splitting a Circle into 5 equal area parts...")
    circle = CircleShape(0, 0, 5)
    circle_splits = split_shape(circle, 5)
    print(f"   Original area: {circle.get_area():.2f}")
    print(f"   Area per part: {circle.get_area() / 5:.2f}")
    for i, part in enumerate(circle_splits):
        print(f"   Part {i+1} area: {part.get_area():.2f}")
    visualize_split(circle, circle_splits, "Circle Split into 5 Equal Parts (Concentric Rings)")

    # Example 3: Triangle
    print("\n3. Splitting a Triangle into 3 equal area parts...")
    triangle = Polygon([(0, 0), (10, 0), (5, 8)])
    triangle_splits = split_shape(triangle, 3)
    print(f"   Original area: {triangle.get_area():.2f}")
    print(f"   Area per part: {triangle.get_area() / 3:.2f}")
    for i, part in enumerate(triangle_splits):
        print(f"   Part {i+1} area: {part.get_area():.2f}")
    visualize_split(triangle, triangle_splits, "Triangle Split into 3 Equal Parts")

    # Example 4: Pentagon
    print("\n4. Splitting a Pentagon into 6 equal area parts...")
    pentagon_vertices = []
    for i in range(5):
        angle = 2 * math.pi * i / 5 - math.pi / 2
        x = 5 * math.cos(angle)
        y = 5 * math.sin(angle)
        pentagon_vertices.append((x, y))

    pentagon = Polygon(pentagon_vertices)
    pentagon_splits = split_shape(pentagon, 6)
    print(f"   Original area: {pentagon.get_area():.2f}")
    print(f"   Area per part: {pentagon.get_area() / 6:.2f}")
    for i, part in enumerate(pentagon_splits):
        print(f"   Part {i+1} area: {part.get_area():.2f}")
    visualize_split(pentagon, pentagon_splits, "Pentagon Split into 6 Equal Parts")

    # Example 5: Custom polygon (L-shape)
    print("\n5. Splitting an L-shaped Polygon into 4 equal area parts...")
    l_shape = Polygon([(0, 0), (6, 0), (6, 3), (3, 3), (3, 6), (0, 6)])
    l_splits = split_shape(l_shape, 4)
    print(f"   Original area: {l_shape.get_area():.2f}")
    print(f"   Area per part: {l_shape.get_area() / 4:.2f}")
    for i, part in enumerate(l_splits):
        print(f"   Part {i+1} area: {part.get_area():.2f}")
    visualize_split(l_shape, l_splits, "L-Shape Split into 4 Equal Parts")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
