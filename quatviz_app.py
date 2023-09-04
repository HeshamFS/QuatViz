# streamlit_app.py
import streamlit as st
import numpy as np
from quaternion import Quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_single_quaternion_orientation(ax, q):
    # Scale down the cube
    scale = 0.3
    points = np.array([
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5]
    ]) * scale

    R = q.to_rotation_matrix()
    points = np.dot(points, R.T)

    Z = [
        [points[j] for j in [0, 1, 5, 4]],
        [points[j] for j in [7, 6, 2, 3]],
        [points[j] for j in [0, 3, 7, 4]],
        [points[j] for j in [1, 2, 6, 5]],
        [points[j] for j in [0, 1, 2, 3]],
        [points[j] for j in [4, 5, 6, 7]]
    ]
    
    ax.add_collection3d(Poly3DCollection(Z, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add reference arrows
    ax.quiver(0, 0, 0, 1, 0, 0, color="r")
    ax.quiver(0, 0, 0, 0, 1, 0, color="g")
    ax.quiver(0, 0, 0, 0, 0, 1, color="b")

def plot_three_quaternions(q1, q2, result):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), dpi=80, subplot_kw={'projection':'3d'})
    
    plot_single_quaternion_orientation(ax1, q1)
    ax1.set_title('Quaternion 1 Orientation')
    
    plot_single_quaternion_orientation(ax2, q2)
    ax2.set_title('Quaternion 2 Orientation')
    
    plot_single_quaternion_orientation(ax3, result)
    ax3.set_title('Result Orientation')

    return fig

def main():
    st.title("Quaternion Mathematics Toolbox")

    # Get user input
    q1_input = st.text_input("Quaternion 1 (w, x, y, z):", "1, 0, 0, 0")
    q2_input = st.text_input("Quaternion 2 (w, x, y, z):", "1, 0, 0, 0")

    operations = ["Add", "Subtract", "Multiply"]
    operation = st.selectbox("Select Operation", operations)

    if st.button("Calculate"):
        try:
            q1_str = q1_input.split(',')
            q1 = Quaternion(*map(float, q1_str))
            quaternion_to_plot = q1  # Default value

            if operation in ["Add", "Subtract", "Multiply"]:
                q2_str = q2_input.split(',')
                q2 = Quaternion(*map(float, q2_str))
                if operation == "Add":
                    result = q1 + q2
                elif operation == "Subtract":
                    result = q1 - q2
                elif operation == "Multiply":
                    result = q1 * q2
                quaternion_to_plot = result  # Update quaternion to plot

            # Display results
            if isinstance(result, Quaternion):
                st.write(f"Result: {result}")

                # Display 3D visualization
                fig = plot_three_quaternions(q1, q2, result)
                st.pyplot(fig)

            else:
                st.write(f"Result: {result:.4f}")

        except ValueError:
            st.error("Please provide valid quaternion values.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
