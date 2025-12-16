import functools
import time
import logging
import mlflow
from typing import Optional
from contextlib import contextmanager
import inspect


logger = logging.getLogger(__name__)

def autolog_params(exclude: Optional[list] = None):
    """
    Декоратор для автоматического логирования аргументов функции как параметров MLflow.
    """
    if exclude is None:
        exclude = ["self", "X_train", "y_train", "X_val", "y_val"]

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Получаем имена аргументов
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            params_to_log = {}
            for k, v in bound_args.arguments.items():
                if k not in exclude and not isinstance(v, (list, dict, object)):
                    # Логируем только простые типы, чтобы не засорять MLflow
                    params_to_log[k] = v

            # Логируем, если есть активный ран
            if mlflow.active_run():
                mlflow.log_params(params_to_log)
                logger.info(f"Autologged params for {func.__name__}: {params_to_log}")

            return func(*args, **kwargs)
        return wrapper
    return decorator

@contextmanager
def mlflow_experiment_context(experiment_name: str, run_name: str = None):
    """
    Контекстный менеджер для управления экспериментом.
    Автоматически считает время выполнения и ловит ошибки.
    """
    mlflow.set_experiment(experiment_name)

    start_time = time.time()
    run = mlflow.start_run(run_name=run_name)

    try:
        yield run
        # Логируем длительность успешного выполнения
        duration = time.time() - start_time
        mlflow.log_metric("execution_time_seconds", duration)

    except Exception as e:
        # Логируем ошибку и помечаем ран как FAILED
        logger.error(f"Run failed: {e}")
        mlflow.set_tag("status", "FAILED")
        mlflow.set_tag("error", str(e))
        raise e
    finally:
        mlflow.end_run()
