# Input and Output {#input-output}

## Visualization and checkpoints

\note Putting the string 'VTK' here in case you are looking for it via full-text search. We are using XDMF instead.

The \ref terra::io::XDMFOutput class implements a combined format for storing simulation data that can be visualized in Paraview,
and can also be loaded back into the simulation (i.e., it serves as a checkpoint).

Refer to the \ref terra::io::XDMFOutput class documentation for more details. It is quite exhaustively documented.

## Tabular data

Have a look at the \ref terra::util::Table class for writing all sorts of tabular data. Also, just for writing to
the console this class can be useful. Consider this for writing to CSV or JSON files.

## Radial profiles

Have a look at the functions \ref terra::shell::radial_profiles() and \ref terra::shell::radial_profiles_to_table() that
compute radial profiles of the shell (min/max/avg) on the device and write them to a table (\ref terra::util::Table) if
desired. This way if can easily be written to console, JSON, or CSV files.
