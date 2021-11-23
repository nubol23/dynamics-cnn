import argparse
from json import load

import numpy as np

from functions import chua_integrator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_images", type=int, default=1, help="number of images of the same attractor"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="global random generator seed"
    )

    num_images = parser.parse_args().n_images
    rng = np.random.default_rng(parser.parse_args().seed)

    with open("data.json") as f:
        data = load(f)

    # Iterating over the json data
    for k in data:
        diagram = data[k]["diagram"]
        n_attractors = data[k]["n_attractors"]
        regular_axis = data[k]["eje_regular"]
        chaotic_axis = data[k]["eje_caotico"]
        print(f"{k=}")
        chua_integrator(
            rng,
            diagram,
            n_attractors,
            regular_axis,
            "regular",
            base_path="Plots/regular",
            save_image=True,
            num_images=num_images,
            save_diagram=False,
            save_points=False,
        )
        chua_integrator(
            rng,
            diagram,
            n_attractors,
            chaotic_axis,
            "caotico",
            base_path="Plots/caotico",
            save_image=True,
            num_images=num_images,
            save_diagram=False,
            save_points=False,
        )


if __name__ == "__main__":
    main()
