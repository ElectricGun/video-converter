# video-converter
[![Github All Releases](https://img.shields.io/github/downloads/electricgun/video-converter/total.svg)]()
Converts a video to a bunch of values for the Mindustry mod [electricgun/video-to-logic](https://github.com/ElectricGun/video-to-logic "video-to-logic")

## How to use:
0. [Download the files](https://github.com/ElectricGun/video-converter/releases/latest) <br> <br>
### Linux
1. Install python3 if you haven't already <br> <br>
2. *cd* to the directory of the script <br> <br>
3. Run *setup.sh* to install modules. <br> <br>
4. Run *python3 video-converter.py "path/to/video"* (args) <br> <br>
5. Run *python3 video-converter.py -h* for list of args <br> <br>
Example: *python3 video-converter.py "~/user/videos/among-us.mp4" -m "raw" --size "176x176"*
### Windows
1. Download and install python3 from python.org, make sure to add it to environment variables. <br> <br>
2. *cd* to the directory of the script <br> <br>
3. Run *setup-windows.bat* to install modules. <br> <br>
4. Run *python video-converter.py "path/to/video"* (args) <br> <br>
5. Run *python video-converter.py -h* for list of args <br> <br>
Example: *python video-converter.py "D:\videos\among-us.mp4" -m "sorter" --size "88x88"*
## Arguments:
### Required
<pre>
-m, --mode      -        Available modes:
                             'sorter' - converts video into a sequence of indices of Mindustry resource colours
                             'raw'    - converts video into a sequence of raw colours
</pre>
### Optional
<pre>
-o, --output    -      Output destination (Default ./output)
  -s, --step      -      The number of frames to skip every step. This causes choppiness but cuts down the size of the output (Default 1)
-l, --length    -      Length of output in seconds. Ignore this arg to convert the entire video
  -i, --integrity -      Integrity of the output. Set to 1 for no compression, set to a low value for maximum compression. Only relevant on "raw" mode (Default 0.99)
--key-interval  -      Creates an uncompressed frame for every n frames. Set to 0 for maximum space reduction (Default 30)
  --size          -      Output aspect ratio. Overrides scale percentage factor. Overrides --scale. Example: '88x88'
--scale         -      Scale percentage. Scales the overall size of the media. Doesn't work with --size (Default 100)
  --batchSize     -      Maximum array length per file. Set to high for massive but fewer files, or low for many but smaller ones (Default 500000)
--offset        -      Length offset in seconds. For example, an offset of 1 will only output from the 30th frame (Default 0)
  --cpu-cores     -      Amount of cpu cores to use, in case multithreading doesn't work
</pre>




