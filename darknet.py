import numpy as np
import tensorflow as tf


'''
Run if  Tf version >=2
del tf    ## delete existing TensorFlow instance (version>=2)
import tensorflow.compat.v1 as tf  ## Enable Backward compatibility for v1 code
'''



_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1
_ANCHORS = [(10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326)]
_MODEL_SIZE = (416, 416)



def batch_norm(inputs, training, data_format):
    return tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
        scale=True, training=training)

def fixed_padding(inputs, kernel_size, data_format):
    pad_total = kernel_size - 1
    pad_start = pad_total // 2
    pad_end = pad_total - pad_start

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_start, pad_end],
                                        [pad_start, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_start, pad_end],
                                        [pad_start, pad_end], [0, 0]])
    return padded_inputs

def conv_padding(inputs, filters, kernel_size, data_format, strides=1):
    
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size,
                            strides=strides, padding=('SAME' if strides == 1 else 'VALID'),
                            use_bias=False, data_format=data_format)

def residual_block(inputs, filters, training, data_format, strides=1):
    
    skip_connection = inputs
    inputs = conv_padding(inputs, filters=filters, kernel_size=1, strides=strides,data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
    inputs = conv_padding(inputs, filters=2 * filters, kernel_size=3, strides=strides,data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
    inputs += skip_connection

    return inputs

def dNetArch(inputs, training, data_format):
    inputs = conv_padding(inputs, filters=32, kernel_size=3,
                                  data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
    inputs = conv_padding(inputs, filters=64, kernel_size=3,
                                  strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs = residual_block(inputs, filters=32, training=training,
                                      data_format=data_format)

    inputs = conv_padding(inputs, filters=128, kernel_size=3,
                                  strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    for _ in range(2):
        inputs = residual_block(inputs, filters=64,
                                          training=training,
                                          data_format=data_format)

    inputs = conv_padding(inputs, filters=256, kernel_size=3,
                                  strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    for _ in range(8):
        inputs = residual_block(inputs, filters=128,
                                          training=training,
                                          data_format=data_format)

    r1 = inputs

    inputs = conv_padding(inputs, filters=512, kernel_size=3,
                                  strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    for _ in range(8):
        inputs = residual_block(inputs, filters=256,
                                          training=training,
                                          data_format=data_format)

    r2 = inputs

    inputs = conv_padding(inputs, filters=1024, kernel_size=3,
                                  strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    for _ in range(4):
        inputs = residual_block(inputs, filters=512,
                                          training=training,
                                          data_format=data_format)

    return r1, r2, inputs


def conv_ext(inputs, filters, training, data_format):
    inputs = conv_padding(inputs, filters=filters, kernel_size=1,
                                  data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs = conv_padding(inputs, filters=2 * filters, kernel_size=3,
                                  data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs = conv_padding(inputs, filters=filters, kernel_size=1,
                                  data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs = conv_padding(inputs, filters=2 * filters, kernel_size=3,
                                  data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs = conv_padding(inputs, filters=filters, kernel_size=1,
                                  data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    r = inputs

    inputs = conv_padding(inputs, filters=2 * filters, kernel_size=3,
                                  data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    return r, inputs

def mouth(inputs, n_classes, anchor_boxes, Img_size, data_format):
    '''
    here we have number of anchor boxes
    '''
    n_anchor_boxes = len(anchor_boxes)
    
    '''
    Make a convolution layer with no. of filters as no. of anchor boxes times (classes+5)
    You can find the theory on yolo research paper.
    '''
    
    inputs = tf.layers.conv2d(inputs, filters=n_anchor_boxes * (5 + n_classes),
                              kernel_size=1, strides=1, use_bias=True,
                              data_format=data_format)
    
    shape = inputs.get_shape().as_list()
    
    '''
    Now we decide to make a grid overlay for our image, So we will find the shape of grid as follows.
    '''
    
    grid_shape = shape[2:4] if data_format == 'channels_first' else shape[1:3]
    
    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])
    
    '''
    Now I reformat the shape as follows would be a better practice and further more implementations 
    according to research paper.
    '''
    
    inputs = tf.reshape(inputs, [-1, n_anchor_boxes * grid_shape[0] * grid_shape[1],
                                 5 + n_classes])

    strides = (Img_size[0] // grid_shape[0], Img_size[1] // grid_shape[1])

    box_c, box_shapes, confidence, classes = tf.split(inputs, [2, 2, 1, n_classes], axis=-1)
    
    '''
    Here I start building the mesh grid as per our algorithm.
    '''

    x = tf.range(grid_shape[0], dtype=tf.float32)
    y = tf.range(grid_shape[1], dtype=tf.float32)
    x_off, y_off = tf.meshgrid(x, y)
    x_off = tf.reshape(x_off, (-1, 1))
    y_off = tf.reshape(y_off, (-1, 1))
    x_y_off = tf.concat([x_off, y_off], axis=-1)
    x_y_off = tf.tile(x_y_off, [1, n_anchor_boxes])
    x_y_off = tf.reshape(x_y_off, [1, -1, 2])
    box_c = tf.nn.sigmoid(box_c)
    box_c = (box_c + x_y_off) * strides

    anchor_boxes = tf.tile(anchor_boxes, [grid_shape[0] * grid_shape[1], 1])
    box_shapes = tf.exp(box_shapes) * tf.to_float(anchor_boxes)

    confidence = tf.nn.sigmoid(confidence)

    classes = tf.nn.sigmoid(classes)

    inputs = tf.concat([box_c, box_shapes,
                        confidence, classes], axis=-1)

    return inputs

def Upsample(inputs, outputShape, data_format):
    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])
        H = outputShape[3]
        W = outputShape[2]
    else:
        H = outputShape[2]
        W = outputShape[1]

    inputs = tf.image.resize_nearest_neighbor(inputs, (H, W))

    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    return inputs


def make_boxes(inputs):
    c_x, c_y, W, H, confidence, classes = \
        tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)

    TL_x = c_x - W / 2
    TL_y = c_y - H / 2
    BR_x = c_x + W / 2
    BR_y = c_y + H / 2

    boxes = tf.concat([TL_x, TL_y,
                       BR_x, BR_y,
                       confidence, classes], axis=-1)

    return boxes

def NMS(inputs, n_classes, max_output_size, iou_threshold,
                        confidence_threshold):
    """

    Args:
        inputs: Tensor input.
        n_classes: Number of classes.
        max_output_size: Max number of boxes to be selected for each class.
        iou_threshold: Threshold for the IOU.
        confidence_threshold: Threshold for the confidence score.
    Returns:
        A list containing class-to-boxes dictionaries
            for each sample in the batch.
    """
    
    
    batch = tf.unstack(inputs)
    boxes_dicts = []
    for boxes in batch:
        boxes = tf.boolean_mask(boxes, boxes[:, 4] > confidence_threshold)
        classes = tf.argmax(boxes[:, 5:], axis=-1)
        classes = tf.expand_dims(tf.to_float(classes), axis=-1)
        boxes = tf.concat([boxes[:, :5], classes], axis=-1)

        boxes_dict = dict()
        for cls in range(n_classes):
            mask = tf.equal(boxes[:, 5], cls)
            mask_shape = mask.get_shape()
            if mask_shape.ndims != 0:
                class_boxes = tf.boolean_mask(boxes, mask)
                boxes_coords, boxes_conf_scores, _ = tf.split(class_boxes,
                                                              [4, 1, -1],
                                                              axis=-1)
                boxes_conf_scores = tf.reshape(boxes_conf_scores, [-1])
                indices = tf.image.non_max_suppression(boxes_coords,
                                                       boxes_conf_scores,
                                                       max_output_size,
                                                       iou_threshold)
                class_boxes = tf.gather(class_boxes, indices)
                boxes_dict[cls] = class_boxes[:, :5]

        boxes_dicts.append(boxes_dict)

    return boxes_dicts

class Model:

    def __init__(self, n_classes, input_size, max_output_size, iou_threshold,
                 confidence_threshold, data_format=None):
        """
        Args:
            n_classes: Number of class labels.
            input_size: The input size of the model.
            max_output_size: Max number of boxes to be selected for each class.
            iou_threshold: Threshold for the IOU.
            confidence_threshold: Threshold for the confidence score.
            data_format: The input format.
        """
        if not data_format:
            if tf.test.is_built_with_cuda():
                data_format = 'channels_first'
            else:
                data_format = 'channels_last'

        self.n_classes = n_classes
        self.input_size = input_size
        self.max_output_size = max_output_size
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.data_format = data_format

    def __call__(self, inputs, training):
        """
        Add operations to detect boxes for a batch of input images.

        Args:
            inputs: A Tensor representing a batch of input images.
            training: A boolean, whether to use in training or inference mode.

        Returns:
            A list containing class-to-boxes dictionaries
                for each sample in the batch.
        """
        
        with tf.variable_scope('yolo_v3_model'):
            if self.data_format == 'channels_first':
                inputs = tf.transpose(inputs, [0, 3, 1, 2])

            inputs = inputs / 255

            r1, r2, inputs = dNetArch(inputs, training=training,
                                               data_format=self.data_format)

            r, inputs = conv_ext(
                inputs, filters=512, training=training,
                data_format=self.data_format)
            detect1 = mouth(inputs, n_classes=self.n_classes,
                                 anchor_boxes=_ANCHORS[6:9],
                                 Img_size=self.input_size,
                                 data_format=self.data_format)

            inputs = conv_padding(r, filters=256, kernel_size=1,
                                          data_format=self.data_format)
            inputs = batch_norm(inputs, training=training,
                                data_format=self.data_format)
            inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
            Upsample_size = r2.get_shape().as_list()
            inputs = Upsample(inputs, outputShape=Upsample_size,
                              data_format=self.data_format)
            axis = 1 if self.data_format == 'channels_first' else 3
            inputs = tf.concat([inputs, r2], axis=axis)
            r, inputs = conv_ext(
                inputs, filters=256, training=training,
                data_format=self.data_format)
            detect2 = mouth(inputs, n_classes=self.n_classes,
                                 anchor_boxes=_ANCHORS[3:6],
                                 Img_size=self.input_size,
                                 data_format=self.data_format)

            inputs = conv_padding(r, filters=128, kernel_size=1,
                                          data_format=self.data_format)
            inputs = batch_norm(inputs, training=training,
                                data_format=self.data_format)
            inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
            Upsample_size = r1.get_shape().as_list()
            inputs = Upsample(inputs, outputShape=Upsample_size,
                              data_format=self.data_format)
            inputs = tf.concat([inputs, r1], axis=axis)
            r, inputs = conv_ext(
                inputs, filters=128, training=training,
                data_format=self.data_format)
            detect3 = mouth(inputs, n_classes=self.n_classes,
                                 anchor_boxes=_ANCHORS[0:3],
                                 Img_size=self.input_size,
                                 data_format=self.data_format)

            inputs = tf.concat([detect1, detect2, detect3], axis=1)

            inputs = make_boxes(inputs)

            boxes_dicts = NMS(
                inputs, n_classes=self.n_classes,
                max_output_size=self.max_output_size,
                iou_threshold=self.iou_threshold,
                confidence_threshold=self.confidence_threshold)

            return boxes_dicts

def load_model_weights(vars, file_name):
    
    with open(file_name, "rb") as f:
        np.fromfile(f, dtype=np.int32, count=5)
        weights = np.fromfile(f, dtype=np.float32)

        assign_ops = []
        ptr = 0

        
        # Each convolution layer has batch normalization.
        for i in range(52):
            conv_var = vars[5 * i]
            gamma, beta, mean, variance = vars[5 * i + 1:5 * i + 5]
            batch_norm_vars = [beta, gamma, mean, variance]

            for var in batch_norm_vars:
                shape = var.shape.as_list()
                num_params = np.prod(shape)
                var_weights = weights[ptr:ptr + num_params].reshape(shape)
                ptr += num_params
                assign_ops.append(tf.assign(var, var_weights))

            shape = conv_var.shape.as_list()
            num_params = np.prod(shape)
            var_weights = weights[ptr:ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1]))
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(tf.assign(conv_var, var_weights))

       
        # 7th, 15th and 23rd convolution layer has biases and no batch norm.
        ranges = [range(0, 6), range(6, 13), range(13, 20)]
        unnormalized = [6, 13, 20]
        for j in range(3):
            for i in ranges[j]:
                current = 52 * 5 + 5 * i + j * 2
                conv_var = vars[current]
                gamma, beta, mean, variance =  \
                    vars[current + 1:current + 5]
                batch_norm_vars = [beta, gamma, mean, variance]

                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(tf.assign(var, var_weights))

                shape = conv_var.shape.as_list()
                num_params = np.prod(shape)
                var_weights = weights[ptr:ptr + num_params].reshape(
                    (shape[3], shape[2], shape[0], shape[1]))
                var_weights = np.transpose(var_weights, (2, 3, 1, 0))
                ptr += num_params
                assign_ops.append(tf.assign(conv_var, var_weights))

            bias = vars[52 * 5 + unnormalized[j] * 5 + j * 2 + 1]
            shape = bias.shape.as_list()
            num_params = np.prod(shape)
            var_weights = weights[ptr:ptr + num_params].reshape(shape)
            ptr += num_params
            assign_ops.append(tf.assign(bias, var_weights))

            conv_var = vars[52 * 5 + unnormalized[j] * 5 + j * 2]
            shape = conv_var.shape.as_list()
            num_params = np.prod(shape)
            var_weights = weights[ptr:ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1]))
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(tf.assign(conv_var, var_weights))

    return assign_ops
