import tensorflow as tf

class ModelWrapper():
    def __init__(self, model, HEIGHT, WIDTH, nb_classes, threshold = 0.25):
        self.model = model
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH
        self.nb_classes = nb_classes
        self.threshold = threshold
    '''
    def call(self, x):
        return self.__call__(x)
    '''
    def __call__(self, x):
        #tf.print("batch shape", x.shape)

        #preds = self.model(self.model.preprocess_input(x))
        preds = self.model(x)
        preds = self.model.decode_predictions(preds)

        # print(preds.shape)
        y = self.get_boxes_constant_size_map(preds)
        tf.print("printing output------------")
        tf.print(tf.shape(y))
        #tf.print(y[0])
        return y

    def get_boxes(self, preds):
        batch_results = []
        for pred in preds:
            #tf.print(pred)
            #tf.print(pred[0])
            #tf.print(pred[0].shape)
            #tf.print([self.WIDTH, self.HEIGHT, self.WIDTH, self.HEIGHT])
            boxes = pred[0] * tf.constant([self.WIDTH, self.HEIGHT, self.WIDTH, self.HEIGHT], dtype=tf.float32)
            results = tf.concat([boxes, tf.expand_dims(pred[2], axis=-1),
                       tf.cast(tf.one_hot(tf.cast(pred[1], tf.int32), self.nb_classes, 1), tf.float32)], axis=-1)


            batch_results.append(results)
        return batch_results

    def get_boxes_constant_size_map(self, preds):

        def pred_loop(pred):
            #tf.print(pred)
            #tf.print(pred[0])
            #tf.print(pred[0].shape)

            #pred = pred[pred[..., 5] > self.threshold]
            boxes = pred[..., :4] * tf.constant([self.WIDTH, self.HEIGHT, self.WIDTH, self.HEIGHT], dtype=tf.float32)
            scores = pred[..., 5]
            cls = pred[..., 4]
            one_hot_cls = tf.cast(tf.one_hot(tf.cast(cls, tf.int32), self.nb_classes, 1), tf.float32)

            results = tf.concat([boxes, tf.expand_dims(scores, axis=-1),
                       one_hot_cls], axis=-1)

            #tf.stack(boxes, scores, )
            #print(results.shape)
            #batch_results.append(results)
            return results
        batch_results = tf.map_fn(pred_loop, preds, infer_shape=False)
        return batch_results



    @property
    def layers(self):
        return self.model.layers

    @property
    def input(self):
        return self.model.input

    @property
    def output(self):
        return self.model.output

    @staticmethod
    def from_config(conf_wrapper):
        model_wrapper = conf_wrapper["model_wrapper"]
        config = conf_wrapper["config"]
        keras_model = model_wrapper.model.__class__.from_config(config)
        keras_model.decode_predictions = model_wrapper.model.decode_predictions
        return ModelWrapper(keras_model, model_wrapper.HEIGHT, model_wrapper.WIDTH,
                            model_wrapper.nb_classes, model_wrapper.threshold)

    def get_config(self):
        return {"config": self.model.get_config(), "model_wrapper": self}

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        return self.model.set_weights(weights)