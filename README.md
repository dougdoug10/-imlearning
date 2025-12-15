# Shape Equal Area Splitter

A Python program that can split any shape into any number of equal area parts.

## Features

- **Multiple Shape Types Supported:**
  - Rectangles
  - Circles (split into concentric rings)
  - Arbitrary Polygons (triangles, pentagons, L-shapes, etc.)

- **Accurate Area Calculations:**
  - Uses the Shoelace formula for polygon areas
  - Binary search algorithm for precise equal-area splitting

- **Visual Output:**
  - Side-by-side comparison of original and split shapes
  - Color-coded parts for easy distinction
  - Area verification for each split

## Requirements

```bash
pip install numpy matplotlib
```

## Usage

### Run the Demo

```bash
python app.py
```

The demo will show 5 examples:
1. Rectangle split into 4 parts
2. Circle split into 5 parts
3. Triangle split into 3 parts
4. Pentagon split into 6 parts
5. L-shape split into 4 parts

### Use in Your Code

```python
from app import Polygon, RectangleShape, CircleShape, split_shape, visualize_split

# Create a triangle
triangle = Polygon([(0, 0), (10, 0), (5, 8)])

# Split into 3 equal area parts
parts = split_shape(triangle, 3)

# Visualize the result
visualize_split(triangle, parts, "My Triangle Split")

# Check areas
print(f"Original area: {triangle.get_area()}")
for i, part in enumerate(parts):
    print(f"Part {i+1} area: {part.get_area()}")
```

### Creating Custom Shapes

**Rectangle:**
```python
rect = RectangleShape(x=0, y=0, width=10, height=5)
rect_parts = split_shape(rect, 4)
```

**Circle:**
```python
circle = CircleShape(center_x=0, center_y=0, radius=5)
circle_parts = split_shape(circle, 3)
```

**Custom Polygon:**
```python
# Define vertices in counter-clockwise or clockwise order
vertices = [(0, 0), (4, 0), (4, 3), (2, 3), (2, 5), (0, 5)]
polygon = Polygon(vertices)
polygon_parts = split_shape(polygon, 5)
```

## How It Works

### For Rectangles
Simple horizontal strips of equal height.

### For Circles
Concentric rings with radii calculated using: `r_i = R * sqrt(i/n)`
where R is the original radius and n is the number of splits.

### For Arbitrary Polygons
1. Calculate total area using Shoelace formula
2. Use binary search to find horizontal split lines
3. Each split creates equal areas with precision tolerance of 1e-6

## Algorithm Details

**Polygon Splitting:**
- Uses horizontal slicing method
- Binary search finds y-coordinates that create equal cumulative areas
- Handles complex shapes like L-shapes and concave polygons
- Vertex sorting ensures proper polygon formation

**Area Calculation:**
- Shoelace formula: `A = 0.5 * |Σ(x_i * y_{i+1} - x_{i+1} * y_i)|`
- Accurate for any simple polygon

## Examples Output

Each example shows:
- Original shape with total area
- Split visualization with color-coded parts
- Individual area verification for each part

## Limitations

- Polygon splitting uses horizontal slicing (vertical slicing can be added)
- Complex self-intersecting polygons are not supported
- Visualization requires display capability (won't work in headless environments without backend configuration)

## Future Enhancements

- Radial splitting for polygons
- Vertical splitting option
- 3D shape support
- Export split coordinates to file
- Interactive GUI for custom shape input
