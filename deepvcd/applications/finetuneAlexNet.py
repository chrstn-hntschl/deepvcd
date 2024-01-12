# -*- coding: utf-8 -*-
import logging
import pathlib
import functools
import json
from typing import List
 
import numpy as np
from argparse import ArgumentError

import tensorflow as tf
from keras.models import Model, Sequential, clone_model
from keras.layers import Dense
from keras.initializers import Constant, RandomNormal
from keras import regularizers
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import SGD
from keras import losses
from keras.metrics import AUC

from sklearn.metrics import average_precision_score

from deepvcd.models.alexnet.AlexNet  import AlexNetFlat, preprocess_train, preprocess_predict
from deepvcd.models.utils import get_layer_index
from deepvcd.dataset.descriptor import YAMLLoader, DirectoryLoader, DatasetDescriptor, get_cross_val_folds
from deepvcd.helpers.image import read_image, one_hot
from deepvcd.callbacks import GetLRSchedule, GetBest

log = logging.getLogger(__name__)


BATCH_SIZE = 256  # Krizhevsky2012: 128
IMG_SIZE = 227
AUTOTUNE = tf.data.AUTOTUNE

# chop-off layer and number of units in retrain layer have been optimized in test_ft_strategies - should actually be also part of the model selection process
CHOP_OFF_LAYER = "dense_3"
NUM_UNITS_EXT_LAYER = 128
# learning rates for retraining and fine-tuning stage
INITIAL_LR_RETRAIN = 0.01
LR_FINETUNE = 0.001

class TransferLearner(object):
    def __init__(self, pretrained_model:tf.keras.models.Model, chop_off_layer:str, num_ext_units:int, num_classes:int) -> None:
        self.num_classes = num_classes
        self.retrain_layer_prefix = "rt_"
        self.model = TransferLearner.__setup_model(base_model=pretrained_model,
                                       chop_off_layer=chop_off_layer,
                                       num_ext_units=num_ext_units,
                                       num_classes=num_classes,
                                       retrain_layer_prefix=self.retrain_layer_prefix)

    @staticmethod
    def __setup_model(base_model:Model, chop_off_layer:str, num_ext_units:int, num_classes:int, retrain_layer_prefix:str) -> Model:
        top_model = Sequential()
        top_model.add(Dense(num_ext_units, activation='relu', name=retrain_layer_prefix + 'ext_dense',
                            kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                            # Krizhevsky2012: 0.01, Caffe?: 0.005
                            bias_initializer=Constant(value=1.0),  # Krizhevsky2012: 1.0, Caffe: 0.1
                            kernel_regularizer=regularizers.l2(0.0005),
                            bias_regularizer=regularizers.l2(0.0005)))
        top_model.add(Dense(num_classes, activation='softmax', name=retrain_layer_prefix + 'predictions',
                            kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                            # Krizhevsky2012: 0.01, Caffe?: 0.1
                            bias_initializer=Constant(value=0.0),
                            kernel_regularizer=regularizers.l2(0.0005),
                            bias_regularizer=regularizers.l2(0.0005)))
        
        # get index of layer to be cut off the original model
        chop_off_index = get_layer_index(model=base_model, layer_name=chop_off_layer)
        # the new output is the output of the layer before chop_off_layer:
        x = base_model.layers[chop_off_index-1].output

        # add the new layers from top_model:
        for layer in top_model.layers:
            x = layer(x)

        return Model(inputs=base_model.input, outputs=x, name=base_model.name+"_ft")

    def retrain(self, train_ds, val_ds, lr:float, epochs:int, metrics:list=None, callbacks:list=list()):
        # 1.) retrain only newly added layers from scratch:
        log.debug("Retraining all layers with layer name equals '{rt_prefix}*'".format(rt_prefix=self.retrain_layer_prefix))
        # set all layers except new layers to not trainable:
        for layer in self.model.layers:
            if layer.name.startswith(self.retrain_layer_prefix):
                layer.trainable = True
            else:
                layer.trainable = False

        optimizer = SGD(learning_rate=lr, momentum=0.9, decay=0.0, nesterov=False)
        self.model.compile(optimizer=optimizer,
                    loss=losses.categorical_crossentropy,
                    metrics=metrics,
                    loss_weights=None,
                    sample_weight_mode=None,
                    weighted_metrics=None,
                    target_tensors=None
                    )
        history = self.model.fit(x=train_ds,
                        epochs=epochs,
                        verbose=2,
                        callbacks=callbacks,
                        validation_data=val_ds,
                        class_weight=None,
                        )
        
        #val_loss, val_metric, val_mAP = self.model.evaluate(val_ds)
        #log.info(f"Validation set results after retraining: val_categorical_accuracy={val_metric:.4f} (val_loss={val_loss:.4f})")
        #scores, gt = predict(model=model, testset_iter=val_ds, preprocessing_func='default')
        return history

    def finetune(self, train_ds, val_ds, lr:float, epochs:int, finetuning_layer=None, metrics:list=None, callbacks:list=list()):
        optimizer = SGD(learning_rate=lr, momentum=0.9, decay=0.0, nesterov=False)
        self.model.compile(optimizer=optimizer,
                    loss=losses.categorical_crossentropy,
                    metrics=metrics,
                    loss_weights=None,
                    sample_weight_mode=None,
                    weighted_metrics=None,
                    target_tensors=None
                    )
       
        log.debug(f"Fine-tuning all layers with small learning rate (lr={lr}")
        if finetuning_layer is None:
            set_trainable = True
        else:
            set_trainable = False

        for layer in self.model.layers:
            if layer.name == finetuning_layer:
                set_trainable = True
            if set_trainable:
                log.debug("setting layer '{lname}' to trainable.".format(lname=layer.name))
                layer.trainable = True
            else:
                layer.trainable = False
        
        history = self.model.fit(x=train_ds,
                            epochs=epochs,
                            verbose=2,
                            callbacks=callbacks,
                            validation_data=val_ds,
                            class_weight=None
                            )
        
        return history

    def predict(self, ds):
        scores = self.model.predict(x=ds, verbose=2)

        gt = list()
        for _, labels in ds:
            gt.append(labels.numpy())
        gt = np.vstack(gt)

        return scores, gt


def prepare_dataset(deepvcd_ds:DatasetDescriptor, seed=None):
    num_classes = len(deepvcd_ds.get_labels())

    #FIXME: needs to be loaded from somewhere rather than hard-coded
    mean = [123.0767, 115.56045, 101.68434]
    log.debug("Pre-computed mean: {mean}".format(mean=mean))

    train_ds,_ = deepvcd_ds.get_tfdataset(subset="train", shuffle_files=True, seed=seed)
    train_ds = train_ds.shuffle(train_ds.cardinality(), seed=seed, reshuffle_each_iteration=True)
    train_ds = train_ds.map(lambda x, y: (read_image(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(lambda x, y: (preprocess_train(x, mean=mean), y), num_parallel_calls=tf.data.AUTOTUNE)
    # convert class labels into one-hot encodings
    train_ds = train_ds.map(lambda x,y: one_hot(x,y,num_classes))
    # Batching the dataset
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    val_test_ds = {"val": None, "test": None}
    val_ds = None
    for subset in ["val", "test"]:
        if len(deepvcd_ds._get_subset_images(subset=subset)):
            val_test_ds[subset],_ = deepvcd_ds.get_tfdataset(subset=subset, shuffle_files=False)
            val_test_ds[subset] = val_test_ds[subset].map(lambda x, y: (read_image(x), y), num_parallel_calls=tf.data.AUTOTUNE)
            val_test_ds[subset] = val_test_ds[subset].map(lambda x, y: (preprocess_predict(x, mean=mean, crop_region="center", horizontal_flip=False), y), num_parallel_calls=tf.data.AUTOTUNE)
            # convert class labels into one-hot encodings
            val_test_ds[subset] = val_test_ds[subset].map(lambda x,y: one_hot(x,y,num_classes))
            # Batching the dataset
            val_test_ds[subset] = val_test_ds[subset].batch(BATCH_SIZE)
            val_test_ds[subset] = val_test_ds[subset].prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_test_ds["val"], val_test_ds["test"]


def optimize_lr_schedule(learner:TransferLearner, train_ds, val_ds, finetuning_layer:str, max_epochs_rt:int=90, max_epochs_ft:int=90, eval_metrics=["CategoricalAccuracy"]):
    # prepare calbacks:
    callbacks =list()
    reduce_lr_cb = ReduceLROnPlateau(monitor="val_categorical_accuracy",
                                    mode="max",
                                    factor=0.1,  # divide by 10
                                    patience=5,
                                    cooldown=5,
                                    min_delta=0.0,
                                    min_lr=0.00001,  # we reduce lr at max three times
                                    verbose=1)
    callbacks.append(reduce_lr_cb)

    # stop training if no more improvement
    early_stop_cb = EarlyStopping(monitor="val_categorical_accuracy",
                                mode="max",
                                min_delta=0.0,
                                patience=11,
                                verbose=1,
                                baseline=None,
                                restore_best_weights=True)  # after training, use model weights of
                                                            # best performing epoch
    callbacks.append(early_stop_cb)

    get_best_cb = GetBest(monitor="val_categorical_accuracy",
                        verbose=1,
                        mode="max",
                        period=1,
                        baseline=None)
    callbacks.append(get_best_cb)

    get_lr_cb = GetLRSchedule()
    callbacks.append(get_lr_cb)

    # retrain
    learner.retrain(train_ds=train_ds, 
                    val_ds=val_ds, 
                    lr=INITIAL_LR_RETRAIN, 
                    epochs=max_epochs_rt, 
                    metrics=eval_metrics, 
                    callbacks=callbacks)

    # evaluate retraining stage
    scores, gt = learner.predict(ds=val_ds)
    retrain_mAP = average_precision_score(y_score=scores, y_true=gt)

    # get learning rate schedule
    retrain_lr_schedule = get_lr_cb.lr_schedule[:get_best_cb.best_epoch - 1]
    
    callbacks =list()
    early_stop_cb = EarlyStopping(monitor="val_categorical_accuracy",
                                mode="max",
                                min_delta=0.0,
                                patience=11,
                                verbose=1,
                                baseline=None, #val_metric_rt,
                                restore_best_weights=True  # after training, use model weights of
                                                            # best performing epoch
                                )
    callbacks.append(early_stop_cb)

    get_best_cb = GetBest(monitor="val_categorical_accuracy",
                        verbose=1,
                        mode="max",
                        period=1,
                        baseline=None, #val_metric_rt
                        )
    callbacks.append(get_best_cb)

    get_lr_cb = GetLRSchedule()
    callbacks.append(get_lr_cb)

    # fine-tune
    learner.finetune(train_ds=train_ds, 
                        val_ds=val_ds, 
                        lr=LR_FINETUNE, 
                        epochs=max_epochs_ft, 
                        finetuning_layer=finetuning_layer, 
                        metrics=eval_metrics, 
                        callbacks=callbacks)
    
    # evaluate fine-tuning stage
    scores, gt = learner.predict(ds=val_ds)
    finetune_mAP = average_precision_score(y_score=scores, y_true=gt)

    # get learning rate schedule
    finetune_lr_schedule = get_lr_cb.lr_schedule[:get_best_cb.best_epoch - 1]

    return {
        "retrain" : {
            "mAP": retrain_mAP,
            "lr_schedule": retrain_lr_schedule
        },
        "finetune" : {
            "mAP": finetune_mAP,
            "lr_schedule": finetune_lr_schedule
        }
    }


def hyperparameter_optimization(deepvcd_ds:DatasetDescriptor, norm:str="lrn", model_weights:str="imagenet", num_xval_folds=5, start_ft_layer:List[str]=None, seed:int=None):
    if seed is None:
        from datetime import datetime
        seed = datetime.now().microsecond
    
    max_epochs_rt=90
    max_epochs_ft=90
    eval_metrics=["CategoricalAccuracy"]  # AUC(curve="PR", multi_label=True, name="mAP")
    
    num_classes = len(deepvcd_ds.get_labels())

    # grid search over all layers to start fine-tuning from - assumption: the smaller the trainig dataset, the better it is to start fine-tuning at a later stage:
    # HINT: optimizing learing rate schedule for retraining stage is actually independent from optimizing the fine-tuning layer. 
    # However, in order to fine-tune, we need a model, that has not been trained on any samples that were already used during retraining - hence we fine-tune per fold.
    grid_search_results = dict()
    grid = start_ft_layer if start_ft_layer is not None else ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'dense_1', 'dense_2']

    for start_ft_layer in grid:
        log.info(f"Optimizing for fine-tuning layer '{start_ft_layer}'")
        if len(deepvcd_ds.get_val_images()):
            log.info("Optimizing on provided validation data.")
            train_ds, val_ds, _ = prepare_dataset(deepvcd_ds=deepvcd_ds, seed=seed)

            # load basemodel and setup transfer learning
            basemodel = AlexNetFlat(include_top=True, input_shape=(227, 227, 3), norm=norm, weights=model_weights, classes=1000)
            learner = TransferLearner(pretrained_model=basemodel, chop_off_layer=CHOP_OFF_LAYER, num_ext_units=NUM_UNITS_EXT_LAYER, num_classes=num_classes)

            grid_search_results[start_ft_layer] = optimize_lr_schedule(learner=learner,
                                 train_ds=train_ds,
                                 val_ds=val_ds,
                                 finetuning_layer=start_ft_layer,
                                 max_epochs_rt=max_epochs_rt,
                                 max_epochs_ft=max_epochs_ft,
                                 eval_metrics=eval_metrics
                                 )
        else:
            fold_results = dict()
            fold_descriptors = get_cross_val_folds(ds_descriptor=deepvcd_ds, n_folds=num_xval_folds, seed=seed)
            log.info("Optimizing using cross validation.")
            for fold,ds in enumerate(fold_descriptors):
                log.info(f"Retrain on fold {fold+1}/{len(fold_descriptors)}")
                fold_train_ds, fold_val_ds, _ = prepare_dataset(deepvcd_ds=ds, seed=seed)

                # load basemodel and setup transfer learning
                basemodel = AlexNetFlat(include_top=True, input_shape=(227, 227, 3), norm=norm, weights=model_weights, classes=1000)
                learner = TransferLearner(pretrained_model=basemodel, chop_off_layer=CHOP_OFF_LAYER, num_ext_units=NUM_UNITS_EXT_LAYER, num_classes=num_classes)
                    
                fold_results[fold] = optimize_lr_schedule(learner=learner,
                                 train_ds=fold_train_ds,
                                 val_ds=fold_val_ds,
                                 finetuning_layer=start_ft_layer,
                                 max_epochs_rt=max_epochs_rt,
                                 max_epochs_ft=max_epochs_ft,
                                 eval_metrics=eval_metrics
                                 )

            log.debug(f"Cross validation results ({start_ft_layer}):")
            log.debug(json.dumps(fold_results, indent=2))

            for fold in range(num_xval_folds):
                retrain_lr_schedule_cnts = dict()
                for lr in fold_results[fold]["retrain"]["lr_schedule"]:
                    retrain_lr_schedule_cnts.setdefault(lr, 0)
                    retrain_lr_schedule_cnts[lr]+=1
                fold_results[fold]["retrain"]["lr_schedule"]=retrain_lr_schedule_cnts

                finetune_lr_schedule_cnts = dict()
                for lr in fold_results[fold]["finetune"]["lr_schedule"]:
                    finetune_lr_schedule_cnts.setdefault(lr, 0)
                    finetune_lr_schedule_cnts[lr]+=1
                fold_results[fold]["finetune"]["lr_schedule"]=finetune_lr_schedule_cnts
            
            # get the reverse sorted list of unique learning rate values applied in all folds (e.g. [0.1, 0.01, 0.001])
            lrs = sorted(max([fold_results[i]["retrain"]["lr_schedule"].keys() for i in fold_results], key=len), reverse=True)
            # get the average number of epochs each learning rate was applied for each fold:
            epochs = [round(e) for e in np.mean([[fold_result["retrain"]["lr_schedule"].get(lr, 0) for lr in lrs] for fold_result in fold_results.values()], axis=0)]
            # generate a new learning rate schedule by repeating each learning rate with the respective (average) number of epochs
            retrain_lr_schedule = sum([[lr]*epochs[lrs.index(lr)] for lr in lrs], [])

            # .. and the same for finetuning:
            lrs = sorted(max([fold_results[i]["finetune"]["lr_schedule"].keys() for i in fold_results], key=len), reverse=True)
            epochs = [round(e) for e in np.mean([[fold_result["finetune"]["lr_schedule"].get(lr, 0) for lr in lrs] for fold_result in fold_results.values()], axis=0)]
            finetune_lr_schedule = sum([[lr]*epochs[lrs.index(lr)] for lr in lrs], [])

            grid_search_results[start_ft_layer] = { 
                "retrain": {
                    "mAP": np.mean([fold_results[fold]["retrain"]["mAP"] for fold in fold_results]),
                    "lr_schedule": retrain_lr_schedule
                },
                "finetune": {
                    "mAP": np.mean([fold_results[fold]["finetune"]["mAP"] for fold in fold_results]),
                    "lr_schedule": finetune_lr_schedule
                }
            }

    log.debug("Grid search results: ")
    log.debug(json.dumps(grid_search_results, indent=2))
    # collect optimal parameters from grid search results
    # currently, decision is based on best results for fine-tuning (meaning that a potentially better retraining schedule found for a less optimal ft layer is ignored)
    best_retrain_mAP = 0.0
    best_retrain_schedule = None
    best_finetune_mAP = 0.0
    best_finetune_layer = None
    best_finetune_schedule = None

    for start_ft_layer in grid_search_results:
        if grid_search_results[start_ft_layer]["finetune"]["mAP"] > best_finetune_mAP:
            best_finetune_mAP=grid_search_results[start_ft_layer]["finetune"]["mAP"]
            best_retrain_schedule=grid_search_results[start_ft_layer]["retrain"]["lr_schedule"]
            best_finetune_layer=start_ft_layer
            best_finetune_schedule=grid_search_results[start_ft_layer]["finetune"]["lr_schedule"]

    hyper_params = {
        "retrain": {
            "lr_schedule": best_retrain_schedule
        },
        "finetune": {
            "lr_schedule": best_finetune_schedule,
            "start_layer": best_finetune_layer,
        }
    }

    return hyper_params


def retrain_and_finetune(deepvcd_ds:DatasetDescriptor, 
                         norm:str="lrn", 
                         model_weights:str="imagenet", 
                         seed=None,
                         lr_schedule_retrain=[INITIAL_LR_RETRAIN],
                         finetuning_layer="conv_1",
                         lr_schedule_finetune=[LR_FINETUNE]):

    eval_metrics = ["CategoricalAccuracy", AUC(curve="PR", multi_label=True, name="mAP")]

    num_classes = len(deepvcd_ds.get_labels())
    tf_train_ds, tf_val_ds, tf_test_ds = prepare_dataset(deepvcd_ds=deepvcd_ds, seed=seed)
   
    # load basemodel and setup transfer learning
    log.info(f"Retrain on full dataset")
    basemodel = AlexNetFlat(include_top=True, input_shape=(227, 227, 3), norm=norm, weights=model_weights, classes=1000)
    learner = TransferLearner(pretrained_model=basemodel, chop_off_layer=CHOP_OFF_LAYER, num_ext_units=NUM_UNITS_EXT_LAYER, num_classes=num_classes)

    # use optimized learning rate schedule for retraining:
    def lr_scheduler(epoch, cur_lr, schedule):
        return schedule[epoch]

    lr_schedule_cb = LearningRateScheduler(schedule=functools.partial(lr_scheduler, schedule=lr_schedule_retrain), verbose=1)
    callbacks = [lr_schedule_cb]
    history_rt = learner.retrain(train_ds=tf_train_ds, 
                                 val_ds=tf_val_ds, 
                                 lr=INITIAL_LR_RETRAIN, 
                                 epochs=len(lr_schedule_retrain), 
                                 metrics=eval_metrics, 
                                 callbacks=callbacks)
    model_retrain = clone_model(learner.model)  #FIXME: better encapsulate model access
    model_retrain.set_weights(learner.model.get_weights())
    scores, gt = learner.predict(ds=tf_test_ds)
    retrain_mAP = average_precision_score(y_score=scores, y_true=gt)

    # use optimized start ft layer value and num epochs for fine-tuning:
    lr_schedule_cb = LearningRateScheduler(schedule=functools.partial(lr_scheduler, schedule=lr_schedule_finetune), verbose=1)
    callbacks = [lr_schedule_cb]
    log.info("Finetune on full dataset")
    history_ft = learner.finetune(train_ds=tf_train_ds, 
                                  val_ds=tf_val_ds, 
                                  lr=LR_FINETUNE, 
                                  epochs=len(lr_schedule_finetune), 
                                  finetuning_layer=finetuning_layer, 
                                  metrics=eval_metrics, 
                                  callbacks=callbacks)
    model_finetune = clone_model(learner.model)  #FIXME: better encapsulate model access
    model_finetune.set_weights(learner.model.get_weights())
    scores, gt = learner.predict(ds=tf_test_ds)
    finetune_mAP = average_precision_score(y_score=scores, y_true=gt)

    log.info("mAP scores on test dataset (retrain stage, finetune stage):")
    log.info(f"{retrain_mAP}, {finetune_mAP}")

    return model_retrain, model_finetune


def main():
    # parse cli parameters
    #FIXME: missing: initial_lr_rt, lr_ft, max_epochs_rt, max_epochs_ft
    #FIXME: model hyperparameters `chop-off layer` is currently hard-coded!
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset",
                        dest="dataset",
                        type=str,
                        help="Path to dataset descriptor yaml or directory following a dataset structure (see documentation).",
                        default=None,
                        required=True)
    parser.add_argument("-w", "--weights",
                        dest="weights",
                        type=str,
                        help="Path to hdf5 file with pre-trained model weights. If none given, ImageNet weights are loaded.",
                        default="imagenet",
                        required=False)
    parser.add_argument("-n", "--norm",
                        dest="norm",
                        type=str,
                        help="Normalization to be used after first and second convolutional layer ('lrn' (default), 'tflrn', 'batch' or None). Must be the same used in the pre-trained model!",
                        default="lrn",
                        required=False)
    parser.add_argument("-s", "--seed", dest="seed", 
                        type=int, 
                        help="Set the random seed.", 
                        default=None,
                        required=False)
    parser.add_argument("--swap_val_test_data", dest="swap_val_test",
                        action="store_true",
                        default=False,
                        required=False,
                        help="Flag to indicate, whether validation and test subsets should be swapped in provided dataset (default: False).")
    parser.add_argument("--start_ft_layer", dest="start_ft_layer",
                        nargs='+', 
                        default=None,
                        required=False,
                        help="List of layers to test for fine-tuning. Each layer listed here will be used as start-layer for fine-tuning, i.e. all layers >= start-layer will be fine-tuned." \
                             "If multiple start-layers are provided, grid search is conducted to find best."
                        )

    args = parser.parse_args()
    deepvcd_ds = None
    if pathlib.Path(args.dataset).is_file():
        log.info("Loading dataset from descriptor file '{filename}'".format(filename=args.dataset))
        deepvcd_ds = YAMLLoader.read(yaml_file=args.dataset)
    elif pathlib.Path(args.dataset).is_dir():
        log.info("Loading dataset from directory '{directory}'".format(directory=args.dataset))
        deepvcd_ds = DirectoryLoader.load(dataset_dir=args.dataset)
    else:
        raise ArgumentError(argument=args.dataset, message=f"No dataset descriptor found!")
    
    # check if we need to swap val and test data:
    if args.swap_val_test:
        deepvcd_ds.ground_truth["val"], deepvcd_ds.ground_truth["test"] = deepvcd_ds.ground_truth["test"], deepvcd_ds.ground_truth["val"]

    norm=None if args.norm.lower()=="none" else args.norm

    hyper_params = hyperparameter_optimization(deepvcd_ds=deepvcd_ds, 
                                               norm=norm, 
                                               model_weights=args.weights, 
                                               num_xval_folds=5, 
                                               start_ft_layer=args.start_ft_layer,
                                               seed=args.seed)
    retrain_and_finetune(deepvcd_ds=deepvcd_ds,
                         norm=norm,
                         model_weights=args.weights,
                         seed=args.seed,
                         lr_schedule_retrain=hyper_params["retrain"]["lr_schedule"],
                         finetuning_layer=hyper_params["finetune"]["start_layer"],
                         lr_schedule_finetune=hyper_params["finetune"]["lr_schedule"])


if __name__ == "__main__":
    import argparse

    # configure logging
    log.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')                                                                                     # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    log.addHandler(ch)

    log.info("Tensorflow version: {ver}".format(ver=tf.__version__))
    log.info("Keras version: {ver}".format(ver=tf.keras.__version__))

    main()

