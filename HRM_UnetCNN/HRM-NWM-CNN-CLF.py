class NormUNet:
    @staticmethod
    def build(input_shape = (128, 128, 3),
                  num_classes = 1):
        inputs = Input(shape=input_shape, name='input_image')
        # Block 1
        down1 = Conv2D(64, (3, 3), padding='same', name='down1_conv1')((inputs))
        down1 = BatchNormalization()(down1)
        down1 = Activation('relu')(down1)
        down1 = Conv2D(64, (3, 3), padding='same', name='down1_conv2')(down1)
        down1 = BatchNormalization()(down1)
        down1 = Activation('relu')(down1)
        down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
        # Block 2
        down2 = Conv2D(128, (3, 3), padding='same', name='down2_conv1')(down1_pool)
        down2 = BatchNormalization()(down2)
        down2 = Activation('relu')(down2)
        down2 = Conv2D(128, (3, 3), padding='same', name='down2_conv2')(down2)
        down2 = BatchNormalization()(down2)
        down2 = Activation('relu')(down2)
        down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
        # Block 3
        down3 = Conv2D(256, (3, 3), padding='same', name='down3_conv1')(down2_pool)
        down3 = BatchNormalization()(down3)
        down3 = Activation('relu')(down3)
        down3 = Conv2D(256, (3, 3), padding='same', name='down3_conv2')(down3)
        down3 = BatchNormalization()(down3)
        down3 = Activation('relu')(down3)
        down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
        # Block 4
        down4 = Conv2D(512, (3, 3), padding='same', name='down4_conv1')(down3_pool)
        down4 = BatchNormalization()(down4)
        down4 = Activation('relu')(down4)
        down4 = Conv2D(512, (3, 3), padding='same', name='down4_conv2')(down4)
        down4 = BatchNormalization()(down4)
        down4 = Activation('relu')(down4)
        down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
        # center
        center = Conv2D(1024, (3, 3), padding='same', name='center_conv1')(down4_pool)
        center = BatchNormalization()(center)
        center = Activation('relu')(center)
        center = Conv2D(1024, (3, 3), padding='same', name='center_conv2')(center)
        center = BatchNormalization()(center)
        center = Activation('relu')(center)
        # up 4
        up4 = UpSampling2D((2, 2))(center)
        up4 = concatenate([down4, up4], axis=3)
        up4 = Conv2D(512, (3, 3), padding='same', name='up4_conv1')(up4)
        up4 = BatchNormalization()(up4)
        up4 = Activation('relu')(up4)
        up4 = Conv2D(512, (3, 3), padding='same', name='up4_conv3')(up4)
        up4 = BatchNormalization()(up4)
        up4 = Activation('relu')(up4)
        # up 3
        up3 = UpSampling2D((2, 2))(up4)
        up3 = concatenate([down3, up3], axis=3)
        up3 = Conv2D(256, (3, 3), padding='same', name='up3_conv1')(up3)
        up3 = BatchNormalization()(up3)
        up3 = Activation('relu')(up3)
        up3 = Conv2D(256, (3, 3), padding='same', name='up3_conv3')(up3)
        up3 = BatchNormalization()(up3)
        up3 = Activation('relu')(up3)
        # up 2
        up2 = UpSampling2D((2, 2))(up3)
        up2 = concatenate([down2, up2], axis=3)
        up2 = Conv2D(128, (3, 3), padding='same', name='up2_conv1')(up2)
        up2 = BatchNormalization()(up2)
        up2 = Activation('relu')(up2)
        up2 = Conv2D(128, (3, 3), padding='same', name='up2_conv3')(up2)
        up2 = BatchNormalization()(up2)
        up2 = Activation('relu')(up2)
        # up 1
        up1 = UpSampling2D((2, 2))(up2)
        up1 = concatenate([down1, up1], axis=3)
        up1 = Conv2D(64, (3, 3), padding='same', name='up1_conv1')(up1)
        up1 = BatchNormalization()(up1)
        up1 = Activation('relu')(up1)
        up1 = Conv2D(64, (3, 3), padding='same', name='up1_conv3')(up1)
        up1 = BatchNormalization()(up1)
        up1 = Activation('relu')(up1)

        # output
        classify = Conv2D(num_classes, (1, 1), activation='sigmoid', name='classify')(up1)
        model = Model(inputs=inputs, outputs=classify, name='norm_unet')
        # model.summary()
        return model

class SmallerVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=inputShape))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))        
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        if classes > 2:
            model.add(Dense(1024, activation='relu'))
            model.add(Dense(classes, activation='softmax'))
        else:         
            model.add(Dense(1024, activation='relu'))
            model.add(Dense(classes, activation='sigmoid'))            
        return model