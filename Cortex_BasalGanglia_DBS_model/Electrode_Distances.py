# -*- coding: utf-8 -*-
"""
Created on Wed April 03 14:27:26 2019

Description: Functions for calculating cell distances to electrode. Distance is
             calculated similar to space.distance, but distance is now between
             a cell position and a point for an electrode rather than between
             two cell positions

Edits:
    10-01-18: Created electrode_distance function

@author: John Fleming, john.fleming@ucdconnect.ie

"""

# There must be some Python package out there that provides most of this stuff.
# Distance computations are provided by scipy.spatial, but scipy is a fairly
# heavy dependency.

import numpy as np
import logging

logger = logging.getLogger("PyNN")


def distance_to_electrode(src_electrode, tgt_cell, mask=None):
    """
    Return the Euclidian distance from a point source electrode to a cell.
    `mask` allows only certain dimensions to be considered, e.g.::
            * to ignore the z-dimension, use `mask=array([0,1])`
            * to ignore y, `mask=array([0,2])`
            * to just consider z-distance, `mask=array([2])`
    'src_electrode' is the electrode positon in xyz co-ordinates.
    'tgt_cell' is the cell that the distance will be calculated to.
    """
    d = src_electrode - tgt_cell.position

    if mask is not None:
        d = d[mask]
    return np.sqrt(np.dot(d, d))


def distances_to_electrode(src_electrode, tgt_pop, coordinate_mask=None):
    """
    Return an array of the Euclidian distances from a point source
    electrode to a population of cells.
    `coordinate_mask` allows only certain dimensions to be considered, e.g.::
            * to ignore the z-dimension, use `coordinate_mask=array([0,1])`
            * to ignore y, `coordinate_mask=array([0,2])`
            * to just consider z-distance, `coordinate_mask=array([2])`
    'src_electrode' is the electrode positon in xyz co-ordinates.
    'tgt_pop' is the target population of cells.
    """

    cell_electrode_distances = np.zeros((tgt_pop.local_size, 1))
    cell_electrode_distances.flatten()

    for ii, tgt_cell in enumerate(tgt_pop):
        cell_electrode_distances[ii] = distance_to_electrode(
            src_electrode, tgt_cell, mask=coordinate_mask
        )

    return cell_electrode_distances


def collateral_distances_to_electrode(src_electrode, tgt_pop, L, nseg):
    """
    Return an nd-array of the Euclidian distances from a point source
    electrode to a population of cells. Each row corresponds to a collateral
    from the cortical population. Each column corresponds to the segments of
    the collateral, with 0 being the furthest segment from the 2d plane the
    cells are distributed in and 1 being in the plane.
    'src_electrode' is the electrode positon in xyz co-ordinates.
    'tgt_pop' is the target population of cells.
    'L' is the length of the cortical collateral
    'nseg' is the number of segments in a collateral
    'segment_electrode_distances' is the distance from the centre of each
    collateral segment to the stimulating electrode. Each row corresponds to
    a collateral of a single cortical cell. Each column corresponds to a
    segment of the collateral.
    """
    segment_electrode_distances = np.zeros((tgt_pop.local_size, nseg))

    segment_centres = np.arange(0, nseg + 3 - 1) * (1 / nseg)
    segment_centres = segment_centres - (1 / (2 * nseg))
    segment_centres[0] = 0
    segment_centres[-1] = 1
    segment_centres = segment_centres[1 : len(segment_centres) - 1]

    z_coordinate = L * segment_centres - L / 2
    # print(z_coordinate)

    for ii, tgt_cell in enumerate(tgt_pop):
        for seg in np.arange(nseg):
            tgt_cell.position = np.array(
                [tgt_cell.position[0], tgt_cell.position[1], z_coordinate[seg]]
            )
            segment_electrode_distances[ii][seg] = distance_to_electrode(
                src_electrode, tgt_cell
            )

    return segment_electrode_distances
