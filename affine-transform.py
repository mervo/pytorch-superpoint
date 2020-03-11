#!/usr/bin/env python

import cv2
import numpy as np

if __name__ == '__main__':
    # Read source image.
    im_src = cv2.imread(
        '/data/projects/pixelplus/pytorch-superpoint/exper_dir/magicpoint_synth_homoAdapt_coco/predictions/val/box_front.png',
        cv2.IMREAD_UNCHANGED)
    rows, cols, ch = im_src.shape

    # Read destination image.
    im_dst = cv2.imread(
        '/data/projects/pixelplus/pytorch-superpoint/exper_dir/magicpoint_synth_homoAdapt_coco/predictions/val/box_top.png')

    pts_src = np.float32([[330, 412], [629, 409], [628, 582]])
    pts_dst = np.float32([[333, 464], [627, 467], [611, 589]])
    print(pts_src.shape)

    M = cv2.getAffineTransform(pts_src, pts_dst)
    im_out = cv2.warpAffine(im_src, M, (cols, rows), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)
    print(im_dst.shape[1], im_dst.shape[0])

    # Display images
    cv2.imshow("Source Image", im_src)
    cv2.imshow("Destination Image", im_dst)
    cv2.imshow("Warped Source Image", im_out)
    # cv2.imwrite(f'./{limit}_box.png', im_out)

    cv2.waitKey(0)
