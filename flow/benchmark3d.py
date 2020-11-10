from typing import Literal

import porepy as pp
import numpy as np

import porepy_mesh_factory as pmf

from . import helpers

__all__ = ["case2"]


def case2(
    grid_ref: Literal[0, 1, 2],
    method: Literal["mpfa", "tpfa", "mvem", "rt0"],
    blocking: bool = False,
):

    if blocking:
        raise NotImplementedError("Have only implemented the all-conductive case")

    # helper functions for setting parameters
    def low_zones(g):
        if g.dim < 3:
            return np.zeros(g.num_cells, dtype=np.bool)

        zone_0 = np.logical_and(g.cell_centers[0, :] > 0.5, g.cell_centers[1, :] < 0.5)

        zone_1 = np.logical_and.reduce(
            tuple(
                [
                    g.cell_centers[0, :] > 0.75,
                    g.cell_centers[1, :] > 0.5,
                    g.cell_centers[1, :] < 0.75,
                    g.cell_centers[2, :] > 0.5,
                ]
            )
        )

        zone_2 = np.logical_and.reduce(
            tuple(
                [
                    g.cell_centers[0, :] > 0.625,
                    g.cell_centers[0, :] < 0.75,
                    g.cell_centers[1, :] > 0.5,
                    g.cell_centers[1, :] < 0.625,
                    g.cell_centers[2, :] > 0.5,
                    g.cell_centers[2, :] < 0.75,
                ]
            )
        )

        return np.logical_or.reduce(tuple([zone_0, zone_1, zone_2]))

    def set_parameters(gb):

        data = {"km": 1, "km_low": 1e-1, "kf": 1e4, "aperture": 1e-4}

        tol = 1e-8

        for g, d in gb:
            d["is_tangential"] = True
            d["low_zones"] = low_zones(g)
            d["Aavatsmark_transmissibilities"] = True

            unity = np.ones(g.num_cells)
            empty = np.empty(0)

            if g.dim == 2:
                d["frac_num"] = g.frac_num * unity
            else:
                d["frac_num"] = -1 * unity

            # set the permeability
            if g.dim == 3:
                kxx = data["km"] * unity
                kxx[d["low_zones"]] = data["km_low"]
                perm = pp.SecondOrderTensor(kxx=kxx)

            elif g.dim == 2:
                kxx = data["kf"] * unity
                perm = pp.SecondOrderTensor(kxx=kxx)
            else:  # dim == 1
                kxx = data["kf"] * unity
                perm = pp.SecondOrderTensor(kxx=kxx)

            # Assign apertures
            aperture = np.power(data["aperture"], 3 - g.dim)

            # Boundaries
            b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            bc_val = np.zeros(g.num_faces)

            if b_faces.size != 0:

                b_face_centers = g.face_centers[:, b_faces]
                b_inflow = np.logical_and.reduce(
                    tuple(b_face_centers[i, :] < 0.25 - tol for i in range(3))
                )
                b_outflow = np.logical_and.reduce(
                    tuple(b_face_centers[i, :] > 0.875 + tol for i in range(3))
                )

                labels = np.array(["neu"] * b_faces.size)
                labels[b_outflow] = "dir"
                bc = pp.BoundaryCondition(g, b_faces, labels)

                f_faces = b_faces[b_inflow]
                bc_val[f_faces] = -aperture * g.face_areas[f_faces]
                bc_val[b_faces[b_outflow]] = 1

            else:
                bc = pp.BoundaryCondition(g, empty, empty)

            specified_parameters_f = {
                "second_order_tensor": perm,
                "aperture": aperture * unity,
                "bc": bc,
                "bc_values": bc_val,
            }
            pp.initialize_default_data(g, d, "flow", specified_parameters_f)

        # Assign coupling permeability, the aperture is read from the lower dimensional grid
        for _, d in gb.edges():
            mg = d["mortar_grid"]
            kn = 2 * data["kf"] * np.ones(mg.num_cells) / data["aperture"]
            d[pp.PARAMETERS] = pp.Parameters(
                mg, ["flow", "transport"], [{"normal_diffusivity": kn}, {}]
            )
            d[pp.DISCRETIZATION_MATRICES] = {"flow": {}, "transport": {}}

    ######
    # problem construction starts here

    gb = pmf.main.generate("flow_benchmark_3d_case_2", refinement=grid_ref)
    set_parameters(gb)

    if method == "tpfa":
        discr = pp.Tpfa("flow")
    elif method == "mpfa":
        discr = pp.Mpfa("flow")
    elif method == "mvem":
        discr = pp.MVEM("flow")
    elif method == "rt0":
        discr = pp.RT0("flow")
    else:
        raise ValueError(f"Unknown method {method}")

    assembler, _ = helpers.setup_flow_assembler(gb, discr)
    return gb, assembler
