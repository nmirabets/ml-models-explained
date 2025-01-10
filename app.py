import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Configure Matplotlib font settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Page config
st.set_page_config(page_title="KMeans Clustering", page_icon="📊", layout="wide")

# Initialize session state for step tracking
if 'kmeans_step' not in st.session_state:
    st.session_state.kmeans_step = 0

# Generate simple sample data
simple_points = np.array([
    [1, 1],   # Point A
    [2, 2],   # Point B
    [5, 1],   # Point C
    [6, 2],   # Point D
    [3, 5],   # Point E
    [4, 6]    # Point F
])
point_names = ['A', 'B', 'C', 'D', 'E', 'F']
df = pd.DataFrame(simple_points, columns=['X', 'Y'])
df['Point'] = point_names

# Initial centers (fixed positions)
initial_centers = np.array([
    [5, 3],    # Center 1 (Red)
    [1, 4],    # Center 2 (Blue)
    [6, 6]     # Center 3 (Green)
])

# Let's calculate the first assignments and new centers
# Points: A(1,1), B(2,2), C(5,1), D(6,2), E(3,5), F(4,6)

# After calculating distances, points will be assigned as:
# C1 (5,3): Points C(5,1), D(6,2)
# C2 (1,4): Points A(1,1), B(2,2), E(3,5)
# C3 (6,6): Point F(4,6)

# Centers after first assignment (iteration 1)
centers_iter1 = np.array([
    [5.5, 1.5],  # Center 1 - average of C(5,1), D(6,2)
    [2.0, 2.67], # Center 2 - average of A(1,1), B(2,2), E(3,5)
    [4.0, 6.0]   # Center 3 - position of F(4,6)
])

# Final centers (iteration 2)
final_centers = np.array([
    [5.5, 1.5],  # Center 1 - average of C(5,1), D(6,2)
    [1.5, 1.5],  # Center 2 - average of A(1,1), B(2,2)
    [3.5, 5.5]   # Center 3 - average of E(3,5), F(4,6)
])

# Steps content
steps = [
    {
        "title": "1. Introduction",
        "text": """Let's learn KMeans clustering using a simple example with 6 points in 2D space. Each point has an X and Y 
coordinate, like on a map.

| Point | X | Y |
|-------|---|---|
| A | 1 | 1 |
| B | 2 | 2 |
| C | 5 | 1 |
| D | 6 | 2 |
| E | 3 | 5 |
| F | 4 | 6 |

We want to group similar points together into 3 clusters.""",
        "n_clusters": 0,
        "show_centers": False,
        "show_assignments": False,
        "show_math": False
    },
    {
        "title": "2. Initialize Centers",
        "text": """We start by placing 3 center points at **random** positions:
```
- Center C1 (Red ×): (5, 3)
- Center C2 (Blue ×): (1, 4)
- Center C3 (Green ×): (6, 6)
```

These centers will serve as the initial representatives for our clusters.
Each point will be assigned to its nearest center.""",
        "n_clusters": 3,
        "show_centers": True,
        "show_assignments": True,
        "show_math": False,
        "centers": initial_centers,
        "iteration": 0,
        "assigned_points": []
    },
    {
        "title": "3. Assign Point A",
        "text": """Let's start with Point A (1, 1). We'll calculate its distance to each center:
```
To Center C1 (5, 3): 
√[(1-5)² + (1-3)²] = √(16 + 4) = √20 = 4.47

To Center C2 (1, 4): 
√[(1-1)² + (1-4)²] = √(0 + 9) = √9 = 3.00

To Center C3 (6, 6): 
√[(1-6)² + (1-6)²] = √(25 + 25) = √50 = 7.07
```
Point A is closest to Center C2 (Blue) with distance 3.00.""",
        "n_clusters": 3,
        "show_centers": True,
        "show_assignments": True,
        "highlight_point": 0,
        "show_math": True,
        "show_distances": True,
        "centers": initial_centers,
        "iteration": 0,
        "assigned_points": [0]
    },
    {
        "title": "4. Assign Point B",
        "text": """Now let's calculate distances for Point B (2, 2):
```
To Center C1 (5, 3): 
√[(2-5)² + (2-3)²] = √(9 + 1) = √10 = 3.16

To Center C2 (1, 4): 
√[(2-1)² + (2-4)²] = √(1 + 4) = √5 = 2.24

To Center C3 (6, 6): 
√[(2-6)² + (2-6)²] = √(16 + 16) = √32 = 5.66
```
Point B is closest to Center C2 (Blue) with distance 2.24.""",
        "n_clusters": 3,
        "show_centers": True,
        "show_assignments": True,
        "highlight_point": 1,
        "show_math": True,
        "show_distances": True,
        "centers": initial_centers,
        "iteration": 0,
        "assigned_points": [0, 1]
    },
    {
        "title": "5. Assign Point C",
        "text": """Let's calculate distances for Point C (5, 1):
```
To Center C1 (5, 3): 
√[(5-5)² + (1-3)²] = √(0 + 4) = √4 = 2.00

To Center C2 (1, 4): 
√[(5-1)² + (1-4)²] = √(16 + 9) = √25 = 5.00

To Center C3 (6, 6): 
√[(5-6)² + (1-6)²] = √(1 + 25) = √26 = 5.10
```
Point C is closest to Center C1 (Red) with distance 2.00.""",
        "n_clusters": 3,
        "show_centers": True,
        "show_assignments": True,
        "highlight_point": 2,
        "show_math": True,
        "show_distances": True,
        "centers": initial_centers,
        "iteration": 0,
        "assigned_points": [0, 1, 2]
    },
    {
        "title": "6. Assign Point D",
        "text": """Let's calculate distances for Point D (6, 2):
```
To Center C1 (5, 3): 
√[(6-5)² + (2-3)²] = √(1 + 1) = √2 = 1.41

To Center C2 (1, 4): 
√[(6-1)² + (2-4)²] = √(25 + 4) = √29 = 5.39

To Center C3 (6, 6): 
√[(6-6)² + (2-6)²] = √(0 + 16) = √16 = 4.00
```
Point D is closest to Center C1 (Red) with distance 1.41.""",
        "n_clusters": 3,
        "show_centers": True,
        "show_assignments": True,
        "highlight_point": 3,
        "show_math": True,
        "show_distances": True,
        "centers": initial_centers,
        "iteration": 0,
        "assigned_points": [0, 1, 2, 3]
    },
    {
        "title": "7. Assign Point E",
        "text": """Let's calculate distances for Point E (3, 5):
```
To Center C1 (5, 3): 
√[(3-5)² + (5-3)²] = √(4 + 4) = √8 = 2.83

To Center C2 (1, 4): 
√[(3-1)² + (5-4)²] = √(4 + 1) = √5 = 2.24

To Center C3 (6, 6): 
√[(3-6)² + (5-6)²] = √(9 + 1) = √10 = 3.16
```
Point E is closest to Center C2 (Blue) with distance 2.24.""",
        "n_clusters": 3,
        "show_centers": True,
        "show_assignments": True,
        "highlight_point": 4,
        "show_math": True,
        "show_distances": True,
        "centers": initial_centers,
        "iteration": 0,
        "assigned_points": [0, 1, 2, 3, 4]
    },
    {
        "title": "8. Assign Point F",
        "text": """Finally, let's calculate distances for Point F (4, 6):
```
To Center C1 (5, 3): 
√[(4-5)² + (6-3)²] = √(1 + 9) = √10 = 3.16

To Center C2 (1, 4): 
√[(4-1)² + (6-4)²] = √(9 + 4) = √13 = 3.61

To Center C3 (6, 6): 
√[(4-6)² + (6-6)²] = √(4 + 0) = √4 = 2.00
```
Point F is closest to Center C3 (Green) with distance 2.00.""",
        "n_clusters": 3,
        "show_centers": True,
        "show_assignments": True,
        "highlight_point": 5,
        "show_math": True,
        "show_distances": True,
        "centers": initial_centers,
        "iteration": 0,
        "assigned_points": [0, 1, 2, 3, 4, 5]
    },
    {
        "title": "9. First Assignment Complete",
        "text": """After assigning all points to their nearest centers, here's the result:
```
Cluster 1 (Red - C1):
- Point C (5, 1) - distance: 2.00
- Point D (6, 2) - distance: 1.41

Cluster 2 (Blue - C2):
- Point A (1, 1) - distance: 3.00
- Point B (2, 2) - distance: 2.24
- Point E (3, 5) - distance: 2.24

Cluster 3 (Green - C3):
- Point F (4, 6) - distance: 2.00
```
Now we can recalculate the center positions based on these assignments.""",
        "n_clusters": 3,
        "show_centers": True,
        "show_assignments": True,
        "show_math": False,
        "centers": initial_centers,
        "iteration": 0,
        "assigned_points": [0, 1, 2, 3, 4, 5]
    },
    {
        "title": "10. Recalculate Centers",
        "text": """Let's recalculate each center position as the average of its assigned points:
```
Center C1 (Red):
Points: C(5,1), D(6,2)
New X = (5 + 6)/2 = 5.5
New Y = (1 + 2)/2 = 1.5
→ New position: (5.5, 1.5)

Center C2 (Blue):
Points: A(1,1), B(2,2), E(3,5)
New X = (1 + 2 + 3)/3 = 2.0
New Y = (1 + 2 + 5)/3 = 2.67
→ New position: (2.0, 2.67)

Center C3 (Green):
Point: F(4,6)
→ New position: (4.0, 6.0)
```
The star markers (★) show the new center positions calculated by averaging their assigned points.
The dashed lines show which points are being averaged together.""",
        "n_clusters": 3,
        "show_centers": True,
        "show_assignments": True,
        "show_math": True,
        "centers": initial_centers,
        "show_new_centers": True,
        "new_centers": centers_iter1,
        "show_movement": True,
        "old_centers": initial_centers,
        "iteration": 1,
        "assigned_points": [0, 1, 2, 3, 4, 5],
        "show_averaging": True
    },
    {
        "title": "11. Second Assignment - Point A",
        "text": """Now with the new center positions, let's reassign all points, starting with Point A (1, 1):
```
To Center C1 (5.5, 1.5): 
√[(1-5.5)² + (1-1.5)²] = √(20.25 + 0.25) = √20.5 = 4.53

To Center C2 (2.0, 2.67): 
√[(1-2)² + (1-2.67)²] = √(1 + 2.79) = √3.79 = 1.95

To Center C3 (4.0, 6.0): 
√[(1-4)² + (1-6)²] = √(9 + 25) = √34 = 5.83
```
Point A stays with Center C2 (Blue) with distance 1.95.""",
        "n_clusters": 3,
        "show_centers": True,
        "show_assignments": True,
        "highlight_point": 0,
        "show_math": True,
        "show_distances": True,
        "centers": centers_iter1,
        "iteration": 1,
        "assigned_points": [0]
    },
    {
        "title": "12. Second Assignment - Point B",
        "text": """Let's check Point B (2, 2):
```
To Center C1 (5.5, 1.5): 
√[(2-5.5)² + (2-1.5)²] = √(12.25 + 0.25) = √12.5 = 3.54

To Center C2 (2.0, 2.67): 
√[(2-2)² + (2-2.67)²] = √(0 + 0.45) = √0.45 = 0.67

To Center C3 (4.0, 6.0): 
√[(2-4)² + (2-6)²] = √(4 + 16) = √20 = 4.47
```
Point B stays with Center C2 (Blue) with distance 0.67.""",
        "n_clusters": 3,
        "show_centers": True,
        "show_assignments": True,
        "highlight_point": 1,
        "show_math": True,
        "show_distances": True,
        "centers": centers_iter1,
        "iteration": 1,
        "assigned_points": [0, 1]
    },
    {
        "title": "13. Second Assignment - Point C",
        "text": """Now for Point C (5, 1):
```
To Center C1 (5.5, 1.5): 
√[(5-5.5)² + (1-1.5)²] = √(0.25 + 0.25) = √0.5 = 0.71

To Center C2 (2.0, 2.67): 
√[(5-2)² + (1-2.67)²] = √(9 + 2.79) = √11.79 = 3.43

To Center C3 (4.0, 6.0): 
√[(5-4)² + (1-6)²] = √(1 + 25) = √26 = 5.10
```
Point C stays with Center C1 (Red) with distance 0.71.""",
        "n_clusters": 3,
        "show_centers": True,
        "show_assignments": True,
        "highlight_point": 2,
        "show_math": True,
        "show_distances": True,
        "centers": centers_iter1,
        "iteration": 1,
        "assigned_points": [0, 1, 2]
    },
    {
        "title": "14. Second Assignment - Point D",
        "text": """Checking Point D (6, 2):
```
To Center C1 (5.5, 1.5): 
√[(6-5.5)² + (2-1.5)²] = √(0.25 + 0.25) = √0.5 = 0.71

To Center C2 (2.0, 2.67): 
√[(6-2)² + (2-2.67)²] = √(16 + 0.45) = √16.45 = 4.06

To Center C3 (4.0, 6.0): 
√[(6-4)² + (2-6)²] = √(4 + 16) = √20 = 4.47
```
Point D stays with Center C1 (Red) with distance 0.71.""",
        "n_clusters": 3,
        "show_centers": True,
        "show_assignments": True,
        "highlight_point": 3,
        "show_math": True,
        "show_distances": True,
        "centers": centers_iter1,
        "iteration": 1,
        "assigned_points": [0, 1, 2, 3]
    },
    {
        "title": "15. Second Assignment - Point E",
        "text": """For Point E (3, 5):
```
To Center C1 (5.5, 1.5): 
√[(3-5.5)² + (5-1.5)²] = √(6.25 + 12.25) = √18.5 = 4.30

To Center C2 (2.0, 2.67): 
√[(3-2)² + (5-2.67)²] = √(1 + 5.43) = √6.43 = 2.54

To Center C3 (4.0, 6.0): 
√[(3-4)² + (5-6)²] = √(1 + 1) = √2 = 1.41
```
Point E moves to Center C3 (Green) with distance 1.41.""",
        "n_clusters": 3,
        "show_centers": True,
        "show_assignments": True,
        "highlight_point": 4,
        "show_math": True,
        "show_distances": True,
        "centers": centers_iter1,
        "iteration": 1,
        "assigned_points": [0, 1, 2, 3, 4]
    },
    {
        "title": "16. Second Assignment - Point F",
        "text": """Finally, for Point F (4, 6):
```
To Center C1 (5.5, 1.5): 
√[(4-5.5)² + (6-1.5)²] = √(2.25 + 20.25) = √22.5 = 4.74

To Center C2 (2.0, 2.67): 
√[(4-2)² + (6-2.67)²] = √(4 + 11.11) = √15.11 = 3.89

To Center C3 (4.0, 6.0): 
√[(4-4)² + (6-6)²] = √(0 + 0) = √0 = 0.00
```
Point F stays with Center C3 (Green) with distance 0.00.""",
        "n_clusters": 3,
        "show_centers": True,
        "show_assignments": True,
        "highlight_point": 5,
        "show_math": True,
        "show_distances": True,
        "centers": centers_iter1,
        "iteration": 1,
        "assigned_points": [0, 1, 2, 3, 4, 5]
    },
    {
        "title": "17. Final Centers",
        "text": """After the second assignment, we have some changes:
```
Center C1 (Red):
- Points C(5,1), D(6,2) remain
- Final position (5.5, 1.5)

Center C2 (Blue):
- Points A(1,1), B(2,2) remain
- Point E moved to C3
- Final position (1.5, 1.5)

Center C3 (Green):
- Point F(4,6) remains
- Gained point E(3,5)
- Final position (3.5, 5.5)
```
Since points are still changing clusters, we would continue iterating. However, for this example, we'll stop here to keep it simple.""",
        "n_clusters": 3,
        "show_centers": True,
        "show_assignments": True,
        "show_math": False,
        "centers": centers_iter1,
        "iteration": 2,
        "assigned_points": [0, 1, 2, 3, 4, 5]
    },
    {
        "title": "18. Final Recalculation",
        "text": """Let's recalculate the centers one final time with the new assignments:
```
Center C1 (Red):
Points: C(5,1), D(6,2)
New X = (5 + 6)/2 = 5.5
New Y = (1 + 2)/2 = 1.5
→ New position: (5.5, 1.5)

Center C2 (Blue):
Points: A(1,1), B(2,2)
New X = (1 + 2)/2 = 1.5
New Y = (1 + 2)/2 = 1.5
→ New position: (1.5, 1.5)

Center C3 (Green):
Points: E(3,5), F(4,6)
New X = (3 + 4)/2 = 3.5
New Y = (5 + 6)/2 = 5.5
→ New position: (3.5, 5.5)
```
The star markers (★) show the final center positions.
The dashed lines show how points in each cluster are averaged together.""",
        "n_clusters": 3,
        "show_centers": True,
        "show_assignments": True,
        "show_math": True,
        "centers": final_centers,
        "show_movement": True,
        "old_centers": centers_iter1,
        "iteration": 2,
        "assigned_points": [0, 1, 2, 3, 4, 5],
        "show_averaging": True
    },
    {
        "title": "19. Convergence Reached",
        "text": """Let's verify if we need another iteration:

1. Current Cluster Assignments:
```
Center C1 (Red, at 5.5, 1.5):
- Point C (5, 1) - distance: 0.71
- Point D (6, 2) - distance: 0.71

Center C2 (Blue, at 1.5, 1.5):
- Point A (1, 1) - distance: 0.71
- Point B (2, 2) - distance: 0.71

Center C3 (Green, at 3.5, 5.5):
- Point E (3, 5) - distance: 0.71
- Point F (4, 6) - distance: 0.71
```

2. If we were to calculate distances again:
- Each point is already closest to its current center
- No point would change clusters
- Therefore, centers would stay in the same positions

We have reached convergence because:
1. Points are well-grouped with similar points
2. Centers are at the exact average position of their clusters
3. Another iteration would not change anything

This is our final clustering result! 🎉""",
        "n_clusters": 3,
        "show_centers": True,
        "show_assignments": True,
        "show_math": False,
        "centers": final_centers,
        "iteration": 3,
        "assigned_points": [0, 1, 2, 3, 4, 5],
        "show_averaging": True
    }
]

# Title
st.title("KMeans Clustering Explained")

# Add a divider
st.markdown("---")

# Create three columns
col2, col3 = st.columns([2, 3])

# Current step content
current_step = steps[st.session_state.kmeans_step]

with st.sidebar:
    st.write("")
    # Create a list of steps with the current one highlighted
    for i, step in enumerate(steps):
        if i == st.session_state.kmeans_step:
            st.markdown(f"**→ :blue[{step['title']}]**")
        else:
            st.markdown(f"  {step['title']}")
    
    st.markdown("---")

with col2:

    # Navigation buttons
    scol1, scol2, scol3 = st.columns([2, 2, 2])
    with scol1:
        # Navigation buttons
        if st.button("⬅️ Previous Step") and st.session_state.kmeans_step > 0:
            st.session_state.kmeans_step -= 1
            st.rerun()
    with scol2:
        st.write(f"Step {st.session_state.kmeans_step + 1} of {len(steps)}")
    with scol3:
        if st.button("Next Step ➡➡️") and st.session_state.kmeans_step < len(steps) - 1:
            st.session_state.kmeans_step += 1
            st.rerun()

    # Display step title and explanation
    st.subheader(current_step["title"])
    st.divider()
    st.write(current_step["text"])
    
    # Show mathematical explanation if needed
    if current_step.get("show_math", False):
        st.markdown("""
        ### 📐 Distance Formula
        To find the closest center to a point, we use the Euclidean distance formula:
        ```
        Distance = √[(x₁ - x₂)² + (y₁ - y₂)²]
        ```
        where (x₁, y₁) is the point and (x₂, y₂) is the center.
        """)

with col3:
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))

    if current_step["n_clusters"] > 0:
        # Use custom centers if provided, otherwise use KMeans
        if "centers" in current_step:
            centers = current_step["centers"]
            # Manually calculate labels based on distances
            labels = np.zeros(len(simple_points), dtype=int)
            for i, point in enumerate(simple_points):
                distances = np.sqrt(((point - centers) ** 2).sum(axis=1))
                labels[i] = np.argmin(distances)
        else:
            kmeans = KMeans(n_clusters=current_step["n_clusters"], random_state=42)
            kmeans.fit(simple_points)
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_

        if current_step["show_assignments"]:
            # Only color points that have been assigned so far
            colors = ['gray'] * len(simple_points)
            if "assigned_points" in current_step:
                for i, label in enumerate(labels):
                    if i in current_step["assigned_points"]:
                        colors[i] = ['red', 'blue', 'green'][label]
            
            scatter = ax.scatter(simple_points[:, 0], simple_points[:, 1], 
                               c=colors, alpha=0.8, s=150)
        else:
            scatter = ax.scatter(simple_points[:, 0], simple_points[:, 1], 
                               c='gray', alpha=0.8, s=150)

        if current_step["show_centers"]:
            # Plot centers with numbers
            centers_scatter = ax.scatter(centers[:, 0], centers[:, 1],
                                      c=['red', 'blue', 'green'], marker='x', s=200,
                                      linewidths=3, label='Current Centers')
            
            # Add center labels
            for i, (x, y) in enumerate(centers):
                ax.annotate(f'C{i+1}', (x, y),
                          xytext=(-20, -20), textcoords='offset points',
                          color=['red', 'blue', 'green'][i],
                          fontweight='bold')
            
            # Show distance lines if calculating for a specific point
            if "highlight_point" in current_step and current_step.get("show_distances", False):
                point = simple_points[current_step["highlight_point"]]
                for i, center in enumerate(centers):
                    # Draw line from point to center
                    line = ax.plot([point[0], center[0]], [point[1], center[1]], 
                                 c=['red', 'blue', 'green'][i], 
                                 linestyle='--', alpha=0.5)
                    # Add distance label
                    mid_x = (point[0] + center[0]) / 2
                    mid_y = (point[1] + center[1]) / 2
                    distance = np.sqrt(((point - center) ** 2).sum())
                    ax.annotate(f'{distance:.2f}', (mid_x, mid_y),
                              xytext=(5, 5), textcoords='offset points',
                              color=['red', 'blue', 'green'][i])

            # Show movement from old centers if requested
            if current_step.get("show_movement", False) and "old_centers" in current_step:
                old_centers = current_step["old_centers"]
                old_centers_scatter = ax.scatter(old_centers[:, 0], old_centers[:, 1],
                                               c=['red', 'blue', 'green'], marker='x', s=200,
                                               alpha=0.3, linewidths=3)
                
                # Add labels to old centers
                for i, (x, y) in enumerate(old_centers):
                    ax.annotate(f'C{i+1}', (x, y),
                              xytext=(-20, -20), textcoords='offset points',
                              color=['red', 'blue', 'green'][i],
                              fontweight='bold', alpha=0.3)
                
                # Draw arrows showing movement
                if current_step.get("show_new_centers", False) and "new_centers" in current_step:
                    new_centers = current_step["new_centers"]
                    for i in range(len(centers)):
                        ax.arrow(old_centers[i, 0], old_centers[i, 1],
                                new_centers[i, 0] - old_centers[i, 0],
                                new_centers[i, 1] - old_centers[i, 1],
                                head_width=0.1, head_length=0.1, fc='k', ec='k', alpha=0.3)
                    # Plot new centers with star markers
                    new_centers_scatter = ax.scatter(new_centers[:, 0], new_centers[:, 1],
                                                   c=['red', 'blue', 'green'], marker='*', s=300,
                                                   alpha=0.8, linewidths=3, label='New Centers')
                    # Add labels to new centers
                    for i, (x, y) in enumerate(new_centers):
                        ax.annotate(f'C{i+1}', (x, y),
                                  xytext=(10, 10), textcoords='offset points',
                                  color=['red', 'blue', 'green'][i],
                                  fontweight='bold')
                else:
                    for i in range(len(centers)):
                        ax.arrow(old_centers[i, 0], old_centers[i, 1],
                                centers[i, 0] - old_centers[i, 0],
                                centers[i, 1] - old_centers[i, 1],
                                head_width=0.1, head_length=0.1, fc='k', ec='k', alpha=0.3)
            
            ax.legend()

        # Highlight current point being processed
        if "highlight_point" in current_step:
            highlight = ax.scatter(simple_points[current_step["highlight_point"], 0],
                                 simple_points[current_step["highlight_point"], 1],
                                 c='yellow', s=300, alpha=0.3)
    else:
        scatter = ax.scatter(simple_points[:, 0], simple_points[:, 1],
                           c='gray', alpha=0.8, s=150)

    # Add point labels
    for i, txt in enumerate(point_names):
        ax.annotate(txt, (simple_points[i, 0], simple_points[i, 1]),
                   xytext=(5, 5), textcoords='offset points')

    # Add iteration number if available
    if "iteration" in current_step:
        ax.set_title(f"KMeans Clustering - Iteration {current_step['iteration']}")
    else:
        ax.set_title("KMeans Clustering Visualization")
    
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 7)
    st.pyplot(fig)


# Add visualization for center recalculation
if current_step.get("show_averaging", False):
    # Draw lines connecting points in each cluster and show averaging
    if current_step["iteration"] == 1:  # First recalculation
        cluster_points = [
            [(5,1), (6,2)],  # Red cluster - C, D
            [(1,1), (2,2), (3,5)],  # Blue cluster - A, B, E
            [(4,6)]  # Green cluster - F
        ]
    else:  # Final recalculation
        cluster_points = [
            [(5,1), (6,2)],  # Red cluster - C, D
            [(1,1), (2,2)],  # Blue cluster - A, B
            [(3,5), (4,6)]  # Green cluster - E, F
        ]
    
    for i, points in enumerate(cluster_points):
        color = ['red', 'blue', 'green'][i]
        points = np.array(points)
        
        # Draw lines connecting points in the cluster in a star pattern
        if len(points) > 1:
            # Calculate centroid
            centroid = points.mean(axis=0)
            # Draw lines from centroid to each point
            for point in points:
                ax.plot([centroid[0], point[0]], 
                       [centroid[1], point[1]], 
                       c=color, linestyle='--', alpha=0.4, linewidth=2)
            
            # Add centroid marker and label
            ax.scatter(centroid[0], centroid[1], 
                      c=color, marker='*', s=200, alpha=0.8,
                      label=f'C{i+1} avg')
            # Add calculation text
            if len(points) == 2:
                ax.annotate(f'({points[:,0].mean():.1f}, {points[:,1].mean():.1f})',
                          (centroid[0], centroid[1]),
                          xytext=(10, 10), textcoords='offset points',
                          color=color, fontweight='bold',
                          bbox=dict(facecolor='white', edgecolor=color, alpha=0.7))
            elif len(points) == 3:
                ax.annotate(f'({points[:,0].mean():.1f}, {points[:,1].mean():.1f})',
                          (centroid[0], centroid[1]),
                          xytext=(10, 10), textcoords='offset points',
                          color=color, fontweight='bold',
                          bbox=dict(facecolor='white', edgecolor=color, alpha=0.7))
        else:
            # For single points, just mark them as center
            ax.scatter(points[0,0], points[0,1], 
                      c=color, marker='*', s=200, alpha=0.8,
                      label=f'C{i+1} avg')
            ax.annotate(f'({points[0,0]:.1f}, {points[0,1]:.1f})',
                       (points[0,0], points[0,1]),
                       xytext=(10, 10), textcoords='offset points',
                       color=color, fontweight='bold',
                       bbox=dict(facecolor='white', edgecolor=color, alpha=0.7))