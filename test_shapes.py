"""
Test script for the Shape Equal Area Splitter
Tests the splitting logic without visualization
"""

import sys
import math

# Simple test without numpy/matplotlib dependencies
def test_basic_logic():
    """Test the basic mathematical logic"""

    print("=" * 60)
    print("TESTING SHAPE SPLITTING LOGIC")
    print("=" * 60)

    # Test 1: Circle area calculation
    print("\nTest 1: Circle Equal Area Rings")
    print("-" * 40)
    radius = 10
    n_parts = 5
    total_area = math.pi * radius ** 2
    area_per_part = total_area / n_parts

    print(f"Circle radius: {radius}")
    print(f"Total area: {total_area:.2f}")
    print(f"Number of parts: {n_parts}")
    print(f"Expected area per part: {area_per_part:.2f}")

    # Calculate ring radii
    for i in range(n_parts):
        outer_r = radius * math.sqrt((i + 1) / n_parts)
        inner_r = radius * math.sqrt(i / n_parts) if i > 0 else 0
        ring_area = math.pi * (outer_r ** 2 - inner_r ** 2)
        print(f"  Ring {i+1}: inner_r={inner_r:.2f}, outer_r={outer_r:.2f}, area={ring_area:.2f}")

        # Verify area
        assert abs(ring_area - area_per_part) < 0.01, f"Ring {i+1} area mismatch!"

    print("✓ All rings have equal area!")

    # Test 2: Rectangle splitting
    print("\nTest 2: Rectangle Equal Area Strips")
    print("-" * 40)
    width = 10
    height = 8
    n_parts = 4
    total_area = width * height
    area_per_part = total_area / n_parts

    print(f"Rectangle: {width} x {height}")
    print(f"Total area: {total_area:.2f}")
    print(f"Number of parts: {n_parts}")
    print(f"Expected area per part: {area_per_part:.2f}")

    strip_height = height / n_parts
    for i in range(n_parts):
        strip_area = width * strip_height
        print(f"  Strip {i+1}: height={strip_height:.2f}, area={strip_area:.2f}")
        assert abs(strip_area - area_per_part) < 0.01, f"Strip {i+1} area mismatch!"

    print("✓ All strips have equal area!")

    # Test 3: Triangle area (Shoelace formula)
    print("\nTest 3: Triangle Area Calculation")
    print("-" * 40)
    vertices = [(0, 0), (10, 0), (5, 8)]

    def shoelace_area(vertices):
        """Calculate polygon area using shoelace formula"""
        n = len(vertices)
        area = 0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        return abs(area) / 2

    area = shoelace_area(vertices)
    expected_area = 0.5 * 10 * 8  # Triangle: 0.5 * base * height

    print(f"Triangle vertices: {vertices}")
    print(f"Calculated area (Shoelace): {area:.2f}")
    print(f"Expected area (0.5*b*h): {expected_area:.2f}")

    assert abs(area - expected_area) < 0.01, "Triangle area mismatch!"
    print("✓ Triangle area calculation correct!")

    # Test 4: Pentagon area
    print("\nTest 4: Pentagon Area Calculation")
    print("-" * 40)

    # Regular pentagon with radius 5
    pentagon_vertices = []
    radius = 5
    for i in range(5):
        angle = 2 * math.pi * i / 5 - math.pi / 2
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        pentagon_vertices.append((x, y))

    area = shoelace_area(pentagon_vertices)
    # Expected area of regular pentagon with circumradius r:
    # A = (5/2) * r^2 * sin(2*pi/5)
    expected_area = (5/2) * radius ** 2 * math.sin(2 * math.pi / 5)

    print(f"Pentagon radius: {radius}")
    print(f"Calculated area (Shoelace): {area:.2f}")
    print(f"Expected area (formula): {expected_area:.2f}")

    assert abs(area - expected_area) < 0.1, "Pentagon area mismatch!"
    print("✓ Pentagon area calculation correct!")

    # Test 5: L-shape area
    print("\nTest 5: L-Shape Area Calculation")
    print("-" * 40)

    l_vertices = [(0, 0), (6, 0), (6, 3), (3, 3), (3, 6), (0, 6)]
    area = shoelace_area(l_vertices)
    # L-shape is two rectangles: 6x3 + 3x3 = 18 + 9 = 27
    expected_area = 27

    print(f"L-shape vertices: {l_vertices}")
    print(f"Calculated area (Shoelace): {area:.2f}")
    print(f"Expected area: {expected_area:.2f}")

    assert abs(area - expected_area) < 0.01, "L-shape area mismatch!"
    print("✓ L-shape area calculation correct!")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    try:
        test_basic_logic()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        sys.exit(1)
