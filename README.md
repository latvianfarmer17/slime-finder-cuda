# Slime Finder: CUDA
This is a program which utilises the CUDA API for **NVIDIA** GPUs to heavily improve the searching capabilities of slime chunks in Minecraft.

This program has two modes of searching for slime chunks:
- Pattern - *look for a specific pattern of slime chunks in some region.*
- Frequency - *look for a specific frequency of slime chunks in some region.*

The *Pattern* mode will look for a specific pattern that you have provided in some rectangular region within a specified seed range.

The *Frequency* mode will only count the slime chunks found within a sub set of some rectangular region within a specified seed range.

## Usage
The general form for using the program is:

`SlimeFinder.exe <mode=pattern> <start-seed> <end-seed> <rx> <rz> <rw> <rh> <pattern>`
`SlimeFinder.exe <mode=frequency> <start-seed> <end-seed> <rx> <rz> <rw> <rh> <frequency.pw.ph>`

`<mode>` can either be `pattern` or `frequency`
`<start-seed>` is the first seed to be checked
`<end-seed>` is the last seed to be checked
`<rx>` is the x-coordinate of the region
`<rz>` is the z-coordinate of the region
`<rw>` is the width of the region
`<rh>` is the height of the region

`<pattern>` is entered by listing the rows of a 2D matrix which are delimited (separated) by `.` where `1` is a slime chunk and `0` is not a slime chunk. All the lengths of the rows should be consistent with the first row's length.

Some valid pattern entries:
- `100.010.001`
- `0000.1100.1111`
- `00.11.11.00`

Some invalid pattern entries:
- `111.00.101`
- `0000.111.00`
- `1.000.111`

`<frequency.pw.ph>` is entered by writing the desired frequency, the pattern width and height to check. If I wanted to check a 3x3 pattern which has a frequency of 5 or more I would write `5.3.3` or if I wanted to check a 2x6 pattern with a frequency of 3 or more it would be `3.2.6`

## Examples
If I want to look for a pattern checking the seeds from 0 to 1,000,000 in a region from (-2, -2) to (2, 2) and the said pattern is a "cross" i.e. `101.010.101` I would write:

`SlimeFinder.exe pattern 0 1000000 -2 -2 4 4 101.010.101`

If I wanted to look for an area containing 5 or more chunks with dimensions 3x3 from seeds -1,000,000 to -123,456 in a region from (-3, -2) to (0, 0) I would write:

`SlimeFinder.exe frequency -1000000 -123456 -3 -2 3 2 5.3.3`
