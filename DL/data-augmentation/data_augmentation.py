# coding=utf-8

import keras.preprocessing.image as image

if __name__ == '__main__':
    datagen = image.ImageDataGenerator(
        rotation_range=40,  #rotation_range：整数，数据提升时图片随机转动的角度
        width_shift_range=0.2,  #width_shift_range：浮点数，图片宽度的某个比例，数据提升时图片水平偏移的幅度`
        height_shift_range=0.2, #height_shift_range：浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度
        rescale=None,   #rescale: 重放缩因子,默认为None. 如果为None或0则不进行放缩,否则会将该数值乘到数据上(在应用其他变换之前)
        shear_range=0.2,    #shear_range：浮点数，剪切强度（逆时针方向的剪切变换角度）
        zoom_range=0.2, #zoom_range：浮点数或形如[lower,upper]的列表，随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
        fill_mode='reflect',    #fill_mode：‘constant’‘nearest’，‘reflect’或‘wrap’之一，当进行变换时超出边界的点将根据本参数给定的方法进行处理
        # cval=5    #cval：浮点数或整数，当fill_mode=constant时，指定要向超出边界的点填充的值
        # channel_shift_range #channel_shift_range: Float. Range for random channel shifts.
        horizontal_flip=True,   #horizontal_flip：布尔值，进行随机水平翻转
        vertical_flip=True, #vertical_flip：布尔值，进行随机竖直翻转
    )
    img = image.load_img('test.jpg')
    x = image.img_to_array(img) # 把PIL图像转换成一个numpy数组，形状为(3, 150, 150)
    x = x.reshape((1,) + x.shape)   # 这是一个numpy数组，形状为 (1, 3, 150, 150)

    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='preview',
                              save_prefix='pig', save_format='jpg'):
        i += 1
        if i>50:
            break

    print("finish data augmentation!")


