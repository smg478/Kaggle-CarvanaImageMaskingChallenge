import cv2

img_file = '/home/galib/kaggle/car_segment/carvana-keras/input/train/c3dafdb02e7f_04.jpg'
image = cv2.imread(img_file)
image = cv2.resize(image,(960,960))
retina = cv2.bioinspired.createRetina((960
                                       , 960),)

retina.setupOPLandIPLParvoChannel(
    1 , #para_dict["color_mode"],
    1 , #para_dict["normalise_output_parvo"],
    0.89 , #para_dict["photoreceptors_local_adaptation_sensitivity"],
    0.9 , #para_dict["photoreceptors_temporal_constant"],
    0.53 , #para_dict["photoreceptors_spatial_constant"],
    0.3 , #para_dict["horizontal_cells_gain"],
    0.5 , #para_dict["hcells_temporal_constant"],
    7 , #para_dict["hcells_spatial_constant"],
    0.89 , #para_dict["ganglion_cells_sensitivity"]
)

retina.setupIPLMagnoChannel(
    1 , #para_dict["normalise_output_magno"],
    0 , #para_dict["parasol_cells_beta"],
    0 , #para_dict["parasol_cells_tau"],
    7 , #para_dict["parasol_cells_k"],
    2 , #para_dict["amacrin_cells_temporal_cut_frequency"],
    0.95 , #para_dict["v0_compression_parameter"],
    0 ,    #para_dict["local_adapt_integration_tau"],
    7 ,    #para_dict["local_adapt_integration_k"]
)

for i in range(20):
   print (i)
   retina.run(image)


retinaOut_parvo = retina.getParvo()
cv2.imshow('image', image)
cv2.imshow('retinaOut_parvo',  retinaOut_parvo)
cv2.waitKey(0)