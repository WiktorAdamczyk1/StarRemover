# StarRemover

This is a tool for removing or reducing stars from images of deep sky objects, allowing the user to customize parameters by combining machine learning with image reproduction techniques. An [implementation of the Mask R-CNN architecture](https://github.com/matterport/Mask_RCNN) by [Matterport, Inc](https://matterport.com/) has been used for this purpose. The resulting model performs segmentation by creating individual masks for each detected object. Using the [implementation of the biharmonic masking algorithm](https://scikit-image.org/docs/stable/auto_examples/filters/plot_inpaint.html), it has been possible to remove the objects detected in the image with varying success.

Weights can be downloaded [here](https://mega.nz/file/T2IwASgK#C-25pOzrVrCxmlsnUaLp2vok52EFZ3MnuOcreIk9BgQ)

## Examples

The following example showcases an image after being processed twice:
1. Removal of 90% of the smallest stars.
2. Removal of 94% of the smallest stars.

<div align="center">
  <img src="https://github.com/WiktorAdamczyk1/StarRemover/blob/main/Examples/Comparison_Ori_1.2.jpg" width="706" height="520"><br><br>
</div>

## Setup

1. Download the [exe](https://mega.nz/file/nnYRnYKC#Xgh6CyHDhxYpVuYOu00lnAC_ysFmtciqzhZywVYbEu4) (made with [Pyinstaller](https://pyinstaller.org/en/stable/)).
2. Download the [weights](https://mega.nz/file/T2IwASgK#C-25pOzrVrCxmlsnUaLp2vok52EFZ3MnuOcreIk9BgQ).
3. Put both files into the same folder.

## Usage

1. Load a suitable image.
2. Set desired parameters e.g. amount of processed stars, desired output size of chosen stars.
3. Process the image.
4. Save the image.

## Functionalities

* Loading images in many common file formats.
* Star detection on images of any size.
* Reduction and removal of stars.

## Summary

This way of star reduction definitely has potential, however other products such as [StarXTerminator](https://www.rc-astro.com/resources/StarXTerminator/) and [StarNet](https://github.com/nekitmm/starnet) achieve better results with GAN Network based solutions. The biggest con of those products is that they process the whole image without allowing the user for any customisation. This is usually offset by providing the user with a mask of detected stars that can be used to manually tweak and fix the received result in some graphics editing software. An improved version of this project could substitute mentioned products and get rid of the necessary post processing step in 3rd party graphics editors. 
