SOFA
====

The SOFA layer is the structured file interface for SOFA data. It loads SOFA
files into ``SOFA`` objects backed by ``netCDF4.Dataset`` and exposes dimensions,
global attributes, variable attributes, and variables through wrapper
collections.

Use this layer when you need direct access to the SOFA file structure.

.. toctree::
   :maxdepth: 2

   loading
   object
   validation
   conventions
