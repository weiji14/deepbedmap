from behave import given, when, then
import numpy as np
import optuna


@given("a prepared collection of tiled raster data")
def load_train_dev_datasets(context):
    dataset, _ = context.srgan_train.load_data_into_memory()
    _, _, context.test_iter, _ = context.srgan_train.get_train_dev_iterators(
        dataset=dataset, first_size=len(dataset) - 1, batch_size=1, seed=42
    )


@given(
    "some hyperparameter settings {num_residual_blocks} {residual_scaling} {learning_rate}"
)
def get_neural_network_hyperparameters(
    context, num_residual_blocks, residual_scaling, learning_rate
):
    context.num_residual_blocks = int(num_residual_blocks)
    context.residual_scaling = float(residual_scaling)
    context.learning_rate = float(learning_rate)


@given("a compiled neural network model")
def compile_neural_network_model_with_hyperparameter_settings(context):
    model = context.srgan_train.compile_srgan_model(
        num_residual_blocks=context.num_residual_blocks,
        residual_scaling=context.residual_scaling,
        learning_rate=context.learning_rate,
    )
    context.g_model, context.g_optimizer, context.d_model, context.d_optimizer = model


@when("the model is trained for a while")
def run_neural_network_model_training(context):
    metric_names = [
        "discriminator_loss",
        "discriminator_accu",
        "generator_loss",
        "generator_psnr",
    ]
    columns = metric_names + [f"val_{metric_name}" for metric_name in metric_names]

    metrics_dict = context.srgan_train.trainer(
        i=0,
        columns=columns,
        train_iter=context.test_iter,
        dev_iter=context.test_iter,
        g_model=context.g_model,
        g_optimizer=context.g_optimizer,
        d_model=context.d_model,
        d_optimizer=context.d_optimizer,
    )
    context.epoch_metrics = {
        metric: np.mean(metrics_dict[metric]) for metric in columns
    }


@then("we know how well the model performs on our test area")
def check_epoch_metrics_not_nan(context):
    for metric in context.epoch_metrics.keys():
        try:
            metric_val = context.epoch_metrics[metric]
            assert not np.isnan(metric_val)
        except AssertionError:
            print(f"{metric} has value: {metric_val}")
