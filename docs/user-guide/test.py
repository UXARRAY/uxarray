import numpy as np

def latlon_to_cartesian(lat, lon):
    """
    Convert latitude and longitude to Cartesian coordinates on the unit sphere.
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.array([x, y, z])

def spherical_triangle_area(vertices):
    """
    Calculate the area of a spherical triangle using the cross product method.
    """
    # Convert vertices to numpy arrays if they aren't already
    v1, v2, v3 = [np.array(v) for v in vertices]
    
    # Calculate the angles using cross products
    cross12 = np.cross(v1, v2)
    cross23 = np.cross(v2, v3)
    cross31 = np.cross(v3, v1)
    
    # Normalize the cross products
    n12 = cross12 / np.linalg.norm(cross12)
    n23 = cross23 / np.linalg.norm(cross23)
    n31 = cross31 / np.linalg.norm(cross31)
    
    # Calculate the angles
    angle1 = np.arccos(np.clip(np.dot(n12, -n31), -1.0, 1.0))
    angle2 = np.arccos(np.clip(np.dot(n23, -n12), -1.0, 1.0))
    angle3 = np.arccos(np.clip(np.dot(n31, -n23), -1.0, 1.0))
    
    # Calculate the spherical excess
    area = angle1 + angle2 + angle3 - np.pi
    
    return area

# Define the triangle vertices
node_lat = np.array([25.0, 48.0, 48.0])  # Latitudes in degrees
node_lon = np.array([-125.0, -68.0, -125.0])  # Longitudes in degrees

# Convert to Cartesian coordinates
vertices = [latlon_to_cartesian(lat, lon) for lat, lon in zip(node_lat, node_lon)]

# Calculate the area
area = spherical_triangle_area(vertices)

print(f"Area of the spherical triangle: {area:.6f} steradians")
print(f"Area in square degrees: {np.degrees(area):.6f}")