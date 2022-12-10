import os
import tensorflow

def strategy():
    if 'POPLAR_SDK_ENABLED' in os.environ:
        from tensorflow.python import ipu
        ipu_config = ipu.config.IPUConfig()
        ipu_config.auto_select_ipus = 1
        ipu_config.configure_ipu_system()
        strategy = ipu.ipu_strategy.IPUStrategy()
        strategy_scope = strategy.scope()
    else:
        gpu = tensorflow.config.list_logical_devices('GPU')
        if len(gpu) > 1:
            strategy = tensorflow.distribute.MirroredStrategy(gpu)
            strategy_scope = strategy.scope()
        elif len(gpu) == 1:
            strategy_scope = tensorflow.device('GPU')
        elif len(gpu) == 0:
            # from contextlib import nullcontext
            # strategy_scope = nullcontext()
            strategy_scope = tensorflow.device('CPU')

    return strategy_scope
