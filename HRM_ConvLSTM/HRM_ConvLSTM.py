SEED, WF_SIZE, HRM_SEQ_SHAPE = 42, 32, (16, 32, 32, 3)         

def generate_sequence(img_path):
    image = cv2.imread(img_path)
    sequence = []
    for i in range(HRM_SEQ_SHAPE[0]):
        frame = image[:, i * WF_SIZE:(i + 1) * WF_SIZE]
        frame = cv2.resize(frame,(HRM_SEQ_SHAPE[2], HRM_SEQ_SHAPE[1]),interpolation=cv2.INTER_CUBIC)
        frame = img_to_array(frame)
        sequence.append(frame)
    return np.array(sequence) / 255.

class Conv2DLSTMNet:
    @staticmethod
    def build(n_frames, width, height, depth, classes):
        model = Sequential()
        inputShape = (n_frames, height, width, depth)
       

        model.add(ConvLSTM2D(8, kernel_size=(3, 3),
                           input_shape=inputShape, padding='same', return_sequences=True))
        model.add(ConvLSTM2D(16, kernel_size=(3, 3), padding='same', return_sequences=True))
        model.add(ConvLSTM2D(32, kernel_size=(3, 3), padding='same', return_sequences=True))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(classes, activation='softmax'))

        return model

class Conv3DLSTMNet:            #*****
    @staticmethod
    def build(n_frames, width, height, depth, classes):
        model = Sequential()
        inputShape = (n_frames, height, width, depth)

        # # kernel_size=(1, 3, 3) || Training:  0.1686 0.1388 0.9395 0.0552 || Validation: 0.2947 0.0511 0.9073 0.0109
        model.add(BatchNormalization(input_shape=inputShape))
        model.add(Conv3D(5, kernel_size=(1, 5, 5), padding='same', activation='relu'))
        model.add(Conv3D(5, kernel_size=(1, 3, 3), padding='same', activation='relu'))
        model.add(Conv3D(5, kernel_size=(3, 3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Bidirectional(ConvLSTM2D(5, kernel_size=(3, 3), padding='same', return_sequences=True)))
        model.add(Bidirectional(ConvLSTM2D(5, kernel_size=(3, 3), padding='same', return_sequences=True)))
        model.add(Conv3D(5, kernel_size=(1, 3, 3), padding='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(classes, activation='softmax'))

        # model.summary()
        return model


