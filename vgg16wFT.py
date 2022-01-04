import tensorflow as tf
from models.ClassifyRunnerModel import ClassifyRunnerModel
import multiprocessing
import time


class ClassifyRunner(ClassifyRunnerModel):

    def train_model(self, cfg, generator_train, generator_valid, key="model", checkpoint=False):
        conv = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=None, pooling=None, input_shape=cfg.img_input_shape)

        # Freeze all the layers
        for layer in conv.layers[:]:
            layer.trainable = False

        new_inputs = tf.keras.layers.Input(shape=cfg.img_input_shape)
        x = conv(new_inputs)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        new_outputs = tf.keras.layers.Dense(cfg.number_of_classes, activation='softmax')(x)

        model = tf.keras.Model(new_inputs, new_outputs)

        model.summary()
        # prints a layer-by-layer summary of the network
        print("Model Inputs: {ips}".format(ips=(model.inputs)))
        print("Model Outputs: {ops}".format(ops=(model.outputs)))

        # Compile new model
        model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lr1),
            metrics=['acc']
        )

        start_time = time.time()
        history = model.fit(
            generator_train,
            epochs=cfg.epochs1,
            validation_steps=generator_valid.samples / generator_valid.batch_size,
            steps_per_epoch=generator_train.samples / generator_train.batch_size,
            validation_data=generator_valid,
            workers=multiprocessing.cpu_count(),
            verbose=1
        )

        print("Finished Training:", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

        # Fine Tuning
        # -----------

        for layer in conv.layers[:]:
            layer.trainable = True

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=cfg.lr2),
                      loss='categorical_crossentropy',
                      metrics=['acc'])

        early_stopping = tf.keras.callbacks.EarlyStopping()

        history = model.fit(
            generator_train,
            epochs=cfg.epochs2,
            validation_steps=generator_valid.samples / generator_valid.batch_size,
            steps_per_epoch=generator_train.samples / generator_train.batch_size,
            validation_data=generator_valid,
            workers=multiprocessing.cpu_count(),
            verbose=1,
            callbacks=[early_stopping, self.get_checkpoint_callback(cfg.checkpoint_destination, key)] if checkpoint is True else [early_stopping],
        )

        print("Finished fine tuning:", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

        self.model = model
        self.history = history

        return model, history
