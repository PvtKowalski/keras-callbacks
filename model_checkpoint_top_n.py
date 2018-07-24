__author__ = 'Pavel Kavaliou'

import heapq
import os
import warnings

from keras.callbacks import Callback


class ModelCheckpointTopN(Callback):
    """Save n models with best scores. This callback is designed to
    fill the gap between saving only the best one and saving all.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    Note: this callback always adds suffix `{epoch:d}`
    to distinguish between previous checkpoints. So for example:
    your filepath `dir/model.h5` becomes `dir/model_{epoch:d}.h5`

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        n_models: int, number of models to save.
        mode: one of {auto, min, max}.
            For `val_acc`, this should be `max`,
            for `val_loss` this should be `min`, etc.
            In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 n_models=5, save_weights_only=False, mode='auto', period=1):
        super(ModelCheckpointTopN, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        head, tail = os.path.split(filepath)
        model_name, ext = os.path.splitext(tail)
        self.filepath = os.path.join(head, model_name+'_{epoch:d}'+ext)
        self.n_models = n_models
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.tracker = []

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % mode, RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.coef = -1.0
        elif mode == 'max':
            self.coef = 1.0
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.coef = 1.0
            else:
                self.coef = -1.0

    def __show_tracked(self):
        return ', '.join(["%0.5f" % (score*self.coef) for score in sorted(list(zip(*self.tracker))[0])])

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch+1, **logs)
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can save best model only with %s available, '
                              'skipping.' % self.monitor, RuntimeWarning)
            else:
                if len(self.tracker) >= self.n_models:
                    discard = heapq.heappushpop(self.tracker, (current*self.coef, epoch+1, filepath))
                    if discard != (current*self.coef, epoch+1, filepath):
                        if self.verbose > 0:
                            print('\nEpoch %05d: Checkpoint from epoch %05d with %s of %0.5f is discarded.'
                                  ' Model with new score of %0.5f is saved to %s. Deleting model %s.'
                                  ' Scores of saved checkpoints: %s.\n'
                                  % (epoch + 1, discard[1], self.monitor, discard[0]*self.coef,
                                     current, filepath, discard[2], self.__show_tracked()))
                        os.remove(discard[2])
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: Model did not improve. Current %s is %0.5f.'
                                  ' Scores of saved checkpoints: %s.\n'
                                  % (epoch + 1, self.monitor, current, self.__show_tracked()))
                else:
                    heapq.heappush(self.tracker, (current*self.coef, epoch+1, filepath))
                    if self.verbose > 0:
                        print('\nEpoch %05d: Saving new checkpoint with %s of %0.5f.'
                              ' Saving model to %s'
                              ' Scores of saved checkpoints: %s.\n'
                              % (epoch + 1, self.monitor, current, filepath, self.__show_tracked()))
                    if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                    else:
                        self.model.save(filepath, overwrite=True)
