#!/usr/bin/env python

import cv2
import numpy as np

if __name__ == '__main__':
    # Read source image.
    # im_src = cv2.imread('/data/projects/pixelplus/pytorch-superpoint/data_dir/COCO/val2014/cruise_front.png', cv2.IMREAD_UNCHANGED)
    # pts_src = np.load('/data/projects/pixelplus/pytorch-superpoint/exper_dir/magicpoint_synth_homoAdapt_coco/predictions/val/cruise_front.npz')
    im_src = cv2.imread('/data/projects/pixelplus/pytorch-superpoint/exper_dir/magicpoint_synth_homoAdapt_coco/predictions/val/box_front.png',
                        cv2.IMREAD_UNCHANGED)
    pts_src = np.load(
        '/data/projects/pixelplus/pytorch-superpoint/exper_dir/magicpoint_synth_homoAdapt_coco/predictions/val/box_front.npz')

    # Read destination image.
    im_dst = cv2.imread('/data/projects/pixelplus/pytorch-superpoint/exper_dir/magicpoint_synth_homoAdapt_coco/predictions/val/box_front2.png')
    pts_dst = np.load('/data/projects/pixelplus/pytorch-superpoint/exper_dir/magicpoint_synth_homoAdapt_coco/predictions/val/box_front2.npz')

    pts_src = pts_src['pts']
    pts_dst = pts_dst['pts']
    print(pts_dst.shape)

    for limit in range(10,120,10):
        # Calculate Homography
        # limit=70
        '''
        TODO
        pts = pts[:top_k, :]
        print("topK filter: ", pts.shape)        
        '''
        print(limit)
        print(pts_src[:2])
        # h, status = cv2.findHomography(pts_src[:limit, :], pts_dst[:limit, :])
        M = cv2.getAffineTransform(pts_src[:3], pts_dst[:3])
        # print(h)

        # Warp source image to destination based on homography
        # im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))
        im_out = cv2.warpAffine(im_src, M, (im_dst.shape[1], im_dst.shape[0]), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
        print(im_dst.shape[1], im_dst.shape[0])

        # Display images
        # cv2.imshow("Source Image", im_src)
        # cv2.imshow("Destination Image", im_dst)
        cv2.imshow("Warped Source Image", im_out)
        cv2.imwrite(f'./{limit}_box.png', im_out)

        cv2.waitKey(0)
