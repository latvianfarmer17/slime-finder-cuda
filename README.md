# Slime Finder: CUDA
This is a program which utilises the CUDA API for **NVIDIA** GPUs to heavily improve the searching capabilities of slime chunks in Minecraft.

When a *region* is referenced, this will indicate the entire range of co-ordinates that will be checked for a pattern/frequency of slime chunks. When a *sub-region* is referenced, it will talk about an area which is **within** the region.

This program has two modes of searching for slime chunks:
- Pattern - *look for a specific pattern of slime chunks in a region.*
- Frequency - *look for a specific frequency of slime chunks in a region.*

The *Pattern* mode will look for a specific pattern that you have provided in some rectangular region within a specified seed range.

The *Frequency* mode will only count the slime chunks found within a sub-region of a rectangular region within a specified seed range.

## Usage
The general form for using the program is:

`SlimeFinder.exe <pattern> <start-seed> <end-seed> <rx> <rz> <rw> <rh> <pattern>`

`SlimeFinder.exe <frequency> <start-seed> <end-seed> <rx> <rz> <rw> <rh> <frequency.srw.srh>`

`<mode>` can either be `pattern` or `frequency`
`<start-seed>` is the first seed to be checked
`<end-seed>` is the last seed to be checked
`<rx>` is the x-coordinate for the top-left corner of the region
`<rz>` is z-coordinate for the top-left corner of the region
`<rw>` is the width of the region
`<rh>` is the height of the region

`<pattern>` is entered by listing the rows of a 2D matrix which are delimited (separated) by `.` where `0` is not a slime chunk, `1` is a slime chunk and `2` is neither (can be a slime chunk or not, it does not matter). All the lengths of the rows should be consistent with the first row's length.

Some valid pattern entries:
- `100.010.201`
- `0220.1100.1111`
- `00.11.11.20`

Some invalid pattern entries:
- `121.00.101`
- `0000.121.00`
- `1.022.111`

`<frequency.srw.srh>` is entered by writing the desired frequency, the sub-region width and height to count the number of slime chunks. If I wanted to check a 3x3 sub-region which has a frequency of 5 or more I would write `5.3.3` or if I wanted to check a 2x6 sub-region with a frequency of 3 or more it would be `3.2.6`

## Examples
If I want to look for a pattern checking the seeds from 0 to 1,000,000 in a region from (-2, -2) to (2, 2) and the said pattern is a "cross" i.e. `101.010.101` I would write:

`SlimeFinder.exe pattern 0 1000000 -2 -2 4 4 101.010.101`

If I wanted to look for a sub-region containing 5 or more chunks with dimensions 3x3 from seeds -1,000,000 to -123,456 in a region from (-3, -2) to (0, 0) I would write:

`SlimeFinder.exe frequency -1000000 -123456 -3 -2 3 2 5.3.3`

Here are some actual results from the program:
`SlimeFinder.exe pattern -2000000000 -1000000000 -50 -50 100 100 12121.21212.12121.21212.12121`

`(+) Found seed -> -1511919784 at (15, -7) / (240, -112)`
`(+) Found seed -> -1258237182 at (5, 7) / (80, 112)`

[![](https://i.imgur.com/XNJPZxC.png)](https://i.imgur.com/XNJPZxC.png)

`SlimeFinder.exe pattern 0 10000000 -5 -5 10 10 111.101.111`

`(+) Found seed -> 550986 at (-4, -3) / (-64, -48)`
`(+) Found seed -> 4951472 at (0, -3) / (0, -48)`

[![](https://i.imgur.com/pLfEbwI.png)](https://i.imgur.com/pLfEbwI.png)

`SlimeFinder.exe pattern 0 1000000 -50 -50 100 100 121.212.101`

`(+) Found seed -> 623 at (-3, -2) / (-48, -32)`
`(+) Found seed -> 944 at (1, -1) / (16, -16)`

[![](https://i.imgur.com/7c0TTHl.png)](https://i.imgur.com/7c0TTHl.png)
