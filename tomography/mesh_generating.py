import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def maximal_coordinates(ve,delta_x):
    return -200/ve + delta_x, + 200/ve + delta_x, -200, 200
 
def calculate_mask_and_extrema(delta_x, ve, *points):
    xa, xb , ya, yb = maximal_coordinates(ve,delta_x)

    # ya, yb = ve * (t1 - delta_x), ve * (t2 - delta_x)

    masks, ix, iy, t_intersections = [], [], [], []
    for i in range(0, len(points), 2):
        mask, ix_, iy_ = line_segment_intersection(xa, ya, xb, yb, points[i], points[i+1], points[(i+2)%len(points)], points[(i+3)%len(points)])
        masks.append(mask)
        ix.append(ix_)
        iy.append(iy_)
        t_intersections.append((ix_ - xa)/(xb - xa))

    # Create masked intersection arrays with NaNs where the mask is False
    masked_t = [np.where(mask, t, np.nan) for mask, t in zip(masks, t_intersections)]

    # Stack the masked intersection arrays along a new axis
    stacked_intersections = np.stack(masked_t, axis=-1)

    # Find the minimum and maximum values among the masked intersection arrays, ignoring NaNs
    t_min_values = np.nanmin(stacked_intersections, axis=-1)
    t_max_values = np.nanmax(stacked_intersections, axis=-1)

    t_min_values = t_min_values[~np.isnan(t_min_values)]
    t_max_values = t_max_values[~np.isnan(t_max_values)]

    mask = np.any(masks, axis=0)
    
    return mask, t_max_values, t_min_values


def line_segment_intersection(xa, ya, xb, yb, x1, y1, x2, y2):
    # Calculate the differences
    dx1 = xb - xa
    dy1 = yb - ya
    dx2 = x2 - x1
    dy2 = y2 - y1

    # Calculate the denominator and numerator
    denom = dx1 * dy2 - dy1 * dx2
    numer = (x1 - xa) * dy2 - (y1 - ya) * dx2

    # Calculate the parameter along the first line segment
    t1 = numer / denom

    # Calculate the parameter along the second line segment
    t2 = ((x1 - xa) * dy1 - (y1 - ya) * dx1) / denom

    # Initialize the arrays for the intersection points
    ix = np.empty_like(x1)
    iy = np.empty_like(y1)

    # Find the intersection points
    mask = (denom != 0) & (t1 >= 0) & (t1 <= 1) & (t2 >= 0) & (t2 <= 1)
    ix[mask] = xa + t1[mask] * dx1
    iy[mask] = ya + t1[mask] * dy1

    return mask, ix, iy

def generate_grid(x_min, x_max, y_min, y_max, nx, ny):
    """Generate grid points and triangulate the grid."""
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)
    triangulation = tri.Triangulation(X.flatten(), Y.flatten())
    return triangulation

def calculate_intersections(delta_x, ve, triangles, x_coords, y_coords):
    """Calculate intersections and mask for triangles."""
    x1 = x_coords[triangles[:, 0]]
    y1 = y_coords[triangles[:, 0]]
    x2 = x_coords[triangles[:, 1]]
    y2 = y_coords[triangles[:, 1]]
    x3 = x_coords[triangles[:, 2]]
    y3 = y_coords[triangles[:, 2]]
    
    # Placeholder for calculate_mask_and_extrema function
    mask, t_max_values, t_min_values = calculate_mask_and_extrema(delta_x, ve, x1, y1, x2, y2, x3, y3)
    return mask, t_min_values, t_max_values


def main():
    # Define the rectangular domain and number of points
    x_min, x_max = -3, 3
    y_min, y_max = -150, 150
    nx, ny = 40, 20

    # Generate grid and triangulate
    triangulation = generate_grid(x_min, x_max, y_min, y_max, nx, ny)
    
    # Define parameters
    ve = 0.7 * 3e2
    ve2 = 0.4 * 3e2
    delta_x = 0
    # xa, xb = 0, 5
    xa, xb , ya, yb = maximal_coordinates(ve,delta_x)
    # Calculate intersections and mask
    mask, t_min_values, t_max_values = calculate_intersections(delta_x, ve, triangulation.triangles, triangulation.x, triangulation.y)

    def gaussian_spot(x , y, y0, x0, sigma):
        return np.exp(-((x-x0)**2 / (2 * (0.5 * sigma)**2) + (y-y0)**2 / (2 * (50*sigma)**2) )) * np.sin(5 * x)
    
    fxy = gaussian_spot(triangulation.x, triangulation.y , 0, 0, 1)
    # Adjust mask and determine line segment
    mask[np.where(mask)[0][15]] = False
    # x_start = xa + t_min_values[15] * (xb - xa)
    # x_end = xa + t_max_values[15] * (xb - xa)
    x_start = xa + min(t_min_values[:]) * (xb-xa)
    x_end = xa + max(t_max_values[:]) * (xb-xa)
    x_line2 = np.linspace(x_start, x_end, 5)
    y_line2 = ve * (x_line2 - delta_x)

    # Plot the results
    x_coords = triangulation.x
    y_coords = triangulation.y
    

    x_line = np.linspace(np.min(y_coords) / ve + delta_x, np.max(y_coords) / ve + delta_x, 500)
    y_line = ve * (x_line - delta_x)

    x_line3 = np.linspace(np.min(y_coords) / ve2 + delta_x, np.max(y_coords) / ve2 + delta_x, 500)
    y_line3 = ve2 * (x_line3 - delta_x)

    plt.figure(figsize=(10, 6), facecolor='white')
    plt.tripcolor(triangulation, fxy, shading='flat', edgecolors='k', linewidths=0.2, cmap='coolwarm')  # Use 'binary' colormap
    # plt.triplot(triangulation, 'ko-', lw=0.5, color='black')  # Add this line to plot the mesh lines
    plt.title('Finite Element Style Triangular Grid on a Rectangular Domain', fontsize=15)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.colorbar(label='Result', shrink=0.7)
    plt.plot(x_line, y_line, 'g-', linewidth=2, label=f'y = {ve}(x - {delta_x})')
    plt.plot(x_line3, y_line3, 'g-', linewidth=2, label=f'y = {ve2}(x - {delta_x})')
    plt.plot(x_line2, y_line2, 'r', linewidth=2)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    print("")

if __name__ == "__main__":
    main()
    print("")
