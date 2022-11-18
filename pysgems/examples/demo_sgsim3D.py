#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os
from os.path import join as join_path

import numpy as np
from matplotlib import pyplot as plt

from pysgems.algo.sgalgo import XML
from pysgems.dis.sgdis import Discretize, blocks_from_rc
from pysgems.io.sgio import PointSet, datread
from pysgems.plot.sgplots import Plots
from pysgems.sgems import sg


def main():
    # %% Initiate sgems pjt
    cwd = os.getcwd()  # Working directory
    rdir = join_path(cwd, "results", "demo_sgsim3D")  # Results directory
    pjt = sg.Sgems(project_name="sgsim3D_test", project_wd=cwd, res_dir=rdir, verbose=True)

    # %% Load hard data point set
    data_dir = join_path(cwd, "datasets", "demo_sgsim3D")
    dataset = "sgsim_hard_data3D.eas"
    file_path = join_path(data_dir, dataset)

    hd = PointSet(project=pjt, pointset_path=file_path)
    hd.export_01()  # Exports in binary

    # %% Generate grid. Grid dimensions can automatically be generated based on the data points
    # unless specified otherwise, but cell dimensions dx, dy, (dz) must be specified
    dis = Discretize(project=pjt, dx=5, dy=5, dz=5, xo=-100, yo=-100, zo=-100, x_lim=500, y_lim=500, z_lim=200)

    # %% Display point coordinates and grid
    pl = Plots(project=pjt)
    pl.plot_coordinates()

    # %% Which feature are available
    print(pjt.point_set.columns)

    # %% Load your algorithm xml file in the 'algorithms' folder.
    algo_dir = join_path(os.path.dirname(cwd), "algorithms")
    al = XML(project=pjt, algo_dir=algo_dir)
    al.xml_reader("sgsim")

    # %% Show xml structure tree
    al.show_tree()

    # %% Modify xml below:
    # By default, the feature grid name of feature X is called 'X_grid'.
    # sgems.xml_update(path, attribute, new value)
    al.xml_update("Assign_Hard_Data", "value", "1")
    al.xml_update("Hard_Data", new_attribute_dict={"grid": "hd_grid", "property": "hd"})

    # %% Write python script
    pjt.write_command()

    # %% Run sgems
    pjt.run()
    # Plot 2D results
    name_kriging = "results"
    result_file_kriging = join_path(rdir, f"{name_kriging}.grid")
    save = "sgsim3D"
    # pl.plot_2d(name_kriging, res_file=result_file_kriging, save=save)

    matrix = datread(result_file_kriging, start=3)
    matrix = np.where(matrix == -9966699, np.nan, matrix)
    lay = 0
    matrix = matrix.reshape((dis.nlay, dis.nrow, dis.ncol))[lay]
    extent = (
        dis.xo,
        dis.x_lim,
        dis.yo,
        dis.y_lim,
    )
    plt.imshow(matrix, cmap="coolwarm", extent=extent)
    plt.plot(
        hd.raw_data[:, 0],
        hd.raw_data[:, 1],
        "k+",
        markersize=1,
        alpha=0.7,
    )
    plt.colorbar()
    # if save:
    #     plt.savefig(jp(res_dir, name), bbox_inches="tight", dpi=300)

    plt.show()

    rows = dis.along_r
    cols = dis.along_c
    lays = dis.along_l

    brc = blocks_from_rc(rows=rows, columns=cols, layers=lays, xo=dis.xo, yo=dis.yo, zo=dis.zo)

    blocks = [b for _, _, b in brc]
    # brc is a generator object
    # to get the item, use list comprehension, or "next"

if __name__ == "__main__":
    main()
