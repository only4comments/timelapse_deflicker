Disclaimer:
I know literally nothing about Python. All the programming was done by Claude based on my requirements and it appears to be generally working as expected.
The software is provided under the MIT license "as-is", no warranty and no liability at my end. 

---

This is a tool to be used for deflickering an image sequence within a timelapse. 


Main features: 
- support of JPEG or TIFF (8 or 16-bit) as input and output (as well as conversion between the formats)
- GUI indicating the original luminance (blue), as well as curves showing the proposed adjustments (green) and the final smoothened luminance curve based on the configured rolling-average (orange). 
- 2-pass deflicker (analysis, application) based on a user-configurable rolling-average luminance

Just choose the source folder with an image sequence (assuming the frame numbers are exactly following each other), choose the destination folder, run Pass 1, adjust the rolling-average as needed, refresh the graph and export. 
The tool only assumes local paths (no UNC). 

Warning:
The tool is made to provide a decent and scalable performance and by default will attempt to fully utilize all your available CPU threads. 
One should expect that 1 used thread = 1 GB of memory. So if you have too many CPU cores and low amount of available memory, you'd better reduce the amount of workers accordingly! 
