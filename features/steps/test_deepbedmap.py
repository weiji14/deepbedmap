from behave import given, when, then
import rasterio


@given("some view of Antarctica {bounding_box}")
def window_view_of_Antarctica(context, bounding_box):
    xmin, ymin, xmax, ymax = [float(c) for c in bounding_box.split(",")]
    context.window_bound = rasterio.coords.BoundingBox(
        left=xmin, bottom=ymin, right=xmax, top=ymax
    )


@when("we gather low and high resolution images related to that view")
def get_model_input_raster_images(context):
    input_tiles = context.deepbedmap.get_deepbedmap_model_inputs(
        window_bound=context.window_bound
    )
    context.X_tile, context.W1_tile, context.W2_tile, context.W3_tile = input_tiles


@when("pass those images into our trained neural network model")
def predict_using_trained_neural_network(context):
    model, _ = context.deepbedmap.load_trained_model(experiment_key="latest")
    context.Y_hat = model.forward(
        x=context.X_tile, w1=context.W1_tile, w2=context.W2_tile, w3=context.W3_tile
    ).array


@then("a four times upsampled super resolution bed elevation map is returned")
def step_impl(context):
    # Ensure input (X_tile) and output (Y_hat) shape is like (1, 1, height, width)
    assert context.X_tile.ndim == 4
    assert context.Y_hat.ndim == 4

    # Check that High Resolution output shape (DeepBedMap) divided by
    # Low Resolution input shape (BEDMAP2) minus 2 pixels (1km on each side)
    # of padding is exactly equal to 4
    assert context.Y_hat.shape[2] / (context.X_tile.shape[2] - 2) == 4.0
    assert context.Y_hat.shape[3] / (context.X_tile.shape[3] - 2) == 4.0
