import os
from pickle import dump
from typing import List, Tuple, TypedDict, Union

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from tqdm import tqdm

plt.rcParams["agg.path.chunksize"] = 100000


@njit
def transient(proximity: int, additive=300_000, scale=100_000) -> Tuple[int, int]:
    """
    Calculates the number of integration steps given the proximity to
    a bifurcation
    :param proximity: proximity to a bifurcation point between 1 and 10
    :param additive: additive term for the calculation
    :param scale: scale term for the calculation
    :returns n: number of integration steps
             t: number of transient steps to drop before considering a valid path
    :raise ValueError if proximity is not in range
    """
    if not (1 <= proximity <= 10):
        raise ValueError("Proximity out of range")

    n = additive + proximity * scale
    t = round(n * 0.4)

    return n, t


@njit
def f(a: float, b: float, x: float) -> float:
    """
    Piecewise linear function
    :param a: function variable
    :param b: function variable
    :param x: function variable
    :return: evaluated function
    """
    return b * x + 0.5 * (a - b) * (abs(x + 1) - abs(x - 1))


@njit
def dx_dt(a: float, b: float, k: float, α: float, x: float, y: float) -> float:
    """
    Derivative of the x coordinate wrt time
    :param a: function variable
    :param b: function variable
    :param k: function variable
    :param α: function variable
    :param x: function variable
    :param y: function variable
    :return: derivative evaluation
    """
    return k * α * (y - x - f(a, b, x))


@njit
def dy_dt(k: float, x: float, y: float, z: float) -> float:
    """
    Derivative of the y coordinate wrt time
    :param k: function variable
    :param x: function variable
    :param y: function variable
    :param z: function variable
    :return: derivative evaluation
    """
    return k * (x - y + z)


@njit
def dz_dt(k: float, β: float, γ: float, y: float, z: float) -> float:
    """
    Derivative of the z coordinate wrt time
    :param k: function variable
    :param β: function variable
    :param γ: function variable
    :param y: function variable
    :param z: function variable
    :return: derivative evaluation
    """
    return -k * (β * y + γ * z)


@njit
def rk_solver(
    a: float,
    b: float,
    k: float,
    α: float,
    β: float,
    γ: float,
    x: float,
    y: float,
    z: float,
    n_steps: int,
    transient_drop: int,
    h=0.001,
) -> np.ndarray:
    trajectory = np.empty((n_steps - transient_drop, 3), dtype=np.float32)

    idx = 0
    for t in range(1, n_steps + 1):
        k1 = dx_dt(a, b, k, α, x, y)
        l1 = dy_dt(k, x, y, z)
        m1 = dz_dt(k, β, γ, y, z)

        k2 = dx_dt(a, b, k, α, x + 0.5 * h * k1, y + 0.5 * h * l1)
        l2 = dy_dt(k, x + 0.5 * h * k1, y + 0.5 * h * l1, z + 0.5 * h * m1)
        m2 = dz_dt(k, β, γ, y + 0.5 * h * l1, z + 0.5 * h * m1)

        k3 = dx_dt(a, b, k, α, x + 0.5 * h * k2, y + 0.5 * h * l2)
        l3 = dy_dt(k, x + 0.5 * h * k2, y + 0.5 * h * l2, z + 0.5 * h * m2)
        m3 = dz_dt(k, β, γ, y + 0.5 * h * l2, z + 0.5 * h * m2)

        k4 = dx_dt(a, b, k, α, x + h * k3, y + h * l3)
        l4 = dy_dt(k, x + h * k3, y + h * l3, z + h * m3)
        m4 = dz_dt(k, β, γ, y + h * l3, z + h * m3)

        px = x + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4) * h
        py = y + 1 / 6 * (l1 + 2 * l2 + 2 * l3 + l4) * h
        pz = z + 1 / 6 * (m1 + 2 * m2 + 2 * m3 + m4) * h

        if t > transient_drop:
            trajectory[idx, 0] = px
            trajectory[idx, 1] = py
            trajectory[idx, 2] = pz
            idx += 1

        x, y, z = px, py, pz

    return trajectory


class Diagram(TypedDict):
    bifurcation: str
    alpha: float
    beta: float
    gamma: float
    a: float
    b: float
    k: float
    initial_condition: List[float]
    axis: str


class AxisRange(TypedDict):
    start: float
    end: float
    proximity: int


def save_data(
    base_path: str,
    diagram_data: Union[Diagram, None],
    label: str,
    points: np.ndarray,
    num: int,
    save_points=True,
    save_diagram=True,
    save_image=True,
    num_images=1,
):
    name = f"Chua_{diagram_data['bifurcation']}_{label}_{num}"

    if save_points:
        if not os.path.isdir(f"{base_path}/arrays"):
            os.makedirs(f"{base_path}/arrays")
        # Save numpy array
        np.save(f"{base_path}/arrays/{name}.npy", points)

    if save_diagram:
        if not os.path.isdir(f"{base_path}/data"):
            os.makedirs(f"{base_path}/data")
        # Save data
        with open(f"{base_path}/data/{name}.dat", "wb") as file:
            dump(diagram_data, file)

    if save_image:
        if not os.path.isdir(f"{base_path}/images"):
            os.makedirs(f"{base_path}/images")
        # Save image
        for n in range(num_images):
            elevation, azimuth = np.random.uniform(0, 180), np.random.uniform(0, 360)
            ax = plt.axes(projection="3d")

            ax.plot3D(points[:, 0], points[:, 1], points[:, 2], "black", linewidth=0.5)
            ax.set_axis_off()
            ax.view_init(elev=elevation, azim=azimuth)
            plt.savefig(f"{base_path}/images/{name}_{n}.png", bbox_inches="tight")
            plt.close()


def chua_integrator(
    diagram_data: Diagram,
    n_attractors: int,
    axis_ranges: List[AxisRange],
    label: str,
    base_path="",
    save_points=True,
    save_diagram=True,
    save_image=True,
    num_images=1,
):
    """
    :param diagram_data: dictionary with control parameters similar to:
                         "bifurcation": "B",
                         "alpha": -1.5590535687,
                         "beta": 0.0156453845,
                         "gamma": 999999999999,
                         "a": -0.2438532907,
                         "b": -0.0425189943,
                         "k": -1,
                         "initial_condition": [0.00, 0.00, 0.20],
                         "axis": "gamma"
                        where "axis" indicate which axis we have to vary,
                        and that varying axis is marked with 999999999999
    :param n_attractors: number of attractors to generate for each range
    :param axis_ranges: a list of ranges, each range is a dictionary like:
                         "start": -0.07, "end": -0.04, "proximity": 6
    :param label: caotico o regular
    :param base_path: path where the file will be saved
    :param save_points: whether to save or not the points
    :param save_diagram: whether to save or not the diagram data
    :param save_image: whether to save or not the images
    :return:
    """
    num = 1
    axis_name = diagram_data["axis"]
    for axis_range in axis_ranges:

        # calculates step size to calculate values of the varying parameter between the range
        step = (axis_range["end"] - axis_range["start"]) / n_attractors

        for i in tqdm(range(n_attractors)):
            # set value to variable axis parameter
            diagram_data[axis_name] = axis_range["start"] + i * step

            init_cond = diagram_data["initial_condition"]

            n_steps, transient_drop = transient(axis_range["proximity"])

            points = rk_solver(
                diagram_data["a"],
                diagram_data["b"],
                diagram_data["k"],
                diagram_data["alpha"],
                diagram_data["beta"],
                diagram_data["gamma"],
                init_cond[0],
                init_cond[1],
                init_cond[2],
                n_steps,
                transient_drop,
            )

            save_data(
                base_path,
                diagram_data,
                label,
                points,
                num,
                save_points,
                save_diagram,
                save_image,
                num_images,
            )

            num += 1


def plot(
    α: float,
    β: float,
    γ: float,
    a: float,
    b: float,
    k: float,
    init_cond: List[float],
    row: int,
    col: int,
    base_path=None,
):  # base_path controla si se guarda la imagen o no
    n_steps, transient_drop = transient(1)

    points = rk_solver(
        a,
        b,
        k,
        α,
        β,
        γ,
        init_cond[0],
        init_cond[1],
        init_cond[2],
        n_steps,
        transient_drop,
    )

    elevation = np.random.randint(0, 180)
    azimuth = np.random.randint(0, 360)

    ax = plt.axes(projection="3d")
    ax.set_axis_off()
    ax.view_init(elev=elevation, azim=azimuth)
    ax.plot3D(points[:, 0], points[:, 1], points[:, 2], "black", linewidth=0.5)

    if base_path:
        if not os.path.isdir(base_path):
            os.makedirs(base_path)
        plt.savefig(f"{base_path}/{row}_{col}.png", bbox_inches="tight")
        plt.close()
    else:
        plt.show()
