''' When processing one file at a time'''
from ij import IJ
from ij.io import FileSaver
from os import path

imp = IJ.getImage()
print(imp)
print("title:", imp.title)

IJ.run("Auto Threshold", "method=Huang white");
IJ.run("Find Edges");
IJ.run("Close-");
IJ.run("Fill Holes");
IJ.run("Remove Outliers...", "radius=3 threshold=50 which=Bright");

fs = FileSaver(imp)

# A known folder to store the image at:
folder = "<target folder>"

# Test if the folder exists before attempting to save the image:
if path.exists(folder) and path.isdir(folder):
  print("folder exists:", folder)
  filepath = path.join(folder, imp.title) # Operating System-specific
  if path.exists(filepath):
    print("File exists! Not saving the image, would overwrite a file!")
  elif fs.saveAsTiff(filepath):
    print("File saved successfully at ", filepath)
else:
  print("Folder does not exist or it's not a folder!")


'''When processing an entire directory of files together'''
import os
from ij import IJ, ImagePlus
from ij.gui import GenericDialog
from loci.plugins import BF

def run():
  srcDir = IJ.getDirectory("Input_directory")
  if not srcDir:
    return
  dstDir = IJ.getDirectory("Output_directory")
  if not dstDir:
    return
  gd = GenericDialog("Process Folder")
  gd.addStringField("File_extension", ".tif")
  gd.addStringField("File_name_contains", "")
  gd.addCheckbox("Keep directory structure when saving", True)
  gd.showDialog()
  if gd.wasCanceled():
    return
  ext = gd.getNextString()
  containString = gd.getNextString()
  keepDirectories = gd.getNextBoolean()
  for root, directories, filenames in os.walk(srcDir):
    for filename in filenames:
      # Check for file extension
      if not filename.endswith(ext):
        continue
      # Check for file name pattern
      if containString not in filename:
        continue
      process(srcDir, dstDir, root, filename, keepDirectories)

def process(srcDir, dstDir, currentDir, fileName, keepDirectories):
  print("Processing:")

  # Opening the image
  print("Open image file", fileName)

  print("Open image path", os.path.join(srcDir,fileName))
  imps = BF.openImagePlus(os.path.join(srcDir, fileName))

  for imp in imps:
    imp.show()
  # Processing commands
  IJ.run("Auto Threshold", "method=Huang white")
  IJ.run("Find Edges")
  IJ.run("Close-")
  IJ.run("Fill Holes")

  # IJ.run("Auto Threshold", "method=Huang2") #run for manual masks
  IJ.run("Remove Outliers...", "radius=3 threshold=50 which=Bright")

  # Saving the image
  saveDir = currentDir.replace(srcDir, dstDir) if keepDirectories else dstDir
  if not os.path.exists(saveDir):
    os.makedirs(saveDir)
  print("Saving to", saveDir)

  # Test if the folder exists before attempting to save the image:
  if os.path.exists(saveDir) and os.path.isdir(saveDir):
    print("folder exists:", saveDir)
    filepath = os.path.join(saveDir, fileName) # Operating System-specific
    if os.path.exists(filepath):
      print("File exists! Not saving the image, would overwrite a file!")
      imp.close()
    else:
      IJ.saveAs(imp, "TIFF", os.path.join(saveDir, fileName))
      imp.close()
  else:
    print("Folder does not exist or it's not a folder!")
    imp.close()

run()
